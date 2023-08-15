# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import logging
import os
import json
import time
import math
import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    get_scheduler,
)

from utils import parse_args

from models import ProtoNetworkWithPrompt
from dataloaders import FewRelDatasetWithPrompt, FewRelTestDatasetWithPrompt, DataCollatorForFewRelWithPrompt, collate_fn

# get logger
logger = get_logger(__name__)


def main():
    # parser arguments
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    # Handle the repository creation
    use_cp = 'CP' if args.use_cp else 'BERT'
    use_rel = '_useRel_' if args.use_rel else '_'
    # args.output_dir = os.path.join(args.output_dir, f'{args.task_name}_{args.model_name_or_path}_{args.proto_type}{use_rel}{use_cp}{args.seed}_{args.N_way}way_{args.K_shot}shot')
    args.output_dir = os.path.join(args.output_dir, f'{args.task_name}_{use_cp}{use_rel}{args.seed}_{args.N_way}way_{args.K_shot}shot')
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    log = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    logging.basicConfig(
        filename=os.path.join(args.output_dir, f'{log}.log'),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(args)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    special_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]', '[unused4]']
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, additional_special_tokens=special_tokens)
    model = ProtoNetworkWithPrompt(config=config,
                                   model_name_or_path=args.model_name_or_path,
                                   N_way=args.N_way,
                                   K_shot=args.K_shot,
                                   Q_num=args.Q_num,
                                   Q_na_rate=args.Q_na_rate,
                                   use_rel=args.use_rel,
                                   use_cp=args.use_cp,
                                   dot_dist=args.dot_dist)


    # padding
    padding = "max_length" if args.pad_to_max_length else True

    # load datasets
    with accelerator.main_process_first():
        # train dataset
        train_dataset = FewRelDatasetWithPrompt(data_dir=args.data_dir,
                                                data_file=args.train_file,
                                                tokenizer=tokenizer,
                                                padding=padding,
                                                max_length=args.max_length,
                                                n_way=args.N_way,
                                                k_shot=args.K_shot,
                                                q_num=args.Q_num,
                                                na_rate=args.Q_na_rate)
        
        # fewrel1.0 validation dataset
        valid_dataset = FewRelDatasetWithPrompt(data_dir=args.data_dir,
                                                data_file=args.valid_file,
                                                tokenizer=tokenizer,
                                                padding=padding,
                                                max_length=args.max_length,
                                                n_way=args.N_way,
                                                k_shot=args.K_shot,
                                                q_num=args.Q_num,
                                                na_rate=args.Q_na_rate)
        
        # fewrel2.0 validation dataset
        valid_dataset_da = FewRelDatasetWithPrompt(data_dir=args.data_dir,
                                                data_file=args.valid_file,
                                                tokenizer=tokenizer,
                                                padding=padding,
                                                max_length=args.max_length,
                                                n_way=args.N_way,
                                                k_shot=args.K_shot,
                                                q_num=args.Q_num,
                                                na_rate=args.Q_na_rate,
                                                is_da=True)
        
        # fewrel1.0 test dataset
        test_dataset = FewRelTestDatasetWithPrompt(data_dir=args.data_dir,
                                                   data_file=args.test_file,
                                                   tokenizer=tokenizer,
                                                   padding=padding,
                                                   max_length=args.max_length,)
        
        # fewrel1.0 test dataset
        test_dataset_da = FewRelTestDatasetWithPrompt(data_dir=args.data_dir,
                                                    data_file=args.test_file,
                                                    tokenizer=tokenizer,
                                                    padding=padding,
                                                    max_length=args.max_length,
                                                    is_da=True)

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = collate_fn
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        # TODO
        data_collator = DataCollatorForFewRelWithPrompt(tokenizer,
                                                        padding=padding,
                                                        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
                                                        return_tensors='pt')

    train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_valid_batch_size)
    valid_dataloader_da = DataLoader(valid_dataset_da, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_valid_batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_valid_batch_size)
    test_dataloader_da = DataLoader(test_dataset_da, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_valid_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_iters,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, valid_dataloader, valid_dataloader_da, test_dataloader, test_dataloader_da, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, valid_dataloader_da, test_dataloader, test_dataloader_da, lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(args.task_name, experiment_config)

    # Training
    logger.info("***** Running training *****")
    logger.info(f"  N-way = {args.N_way}")
    logger.info(f"  K-shot = {args.K_shot}")
    logger.info(f"  Q-num = {args.Q_num}")
    logger.info(f"  Train Iters = {args.num_train_iters}")
    logger.info(f"  Valid Steps = {args.num_valid_steps}")
    logger.info(f"  Valid Iters = {args.num_valid_iters}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    
    # record
    postfix = OrderedDict()
    train_iters = 0
    best_acc = 0.0
    best_acc_da = 0.0
    patience = 0
    patience_da = 0
    total_acc = 0.0
    total_loss = 0.0

    # Only show the progress bar once on each machine.
    train_bar = tqdm(range(args.num_train_iters), disable=not accelerator.is_local_main_process)

    for batch in train_dataloader:
        # train
        model.train()
        # batch
        batch_relation, batch_support, batch_query, batch_label = batch
        loss, acc = model(batch_relation, batch_support, batch_query, batch_label)

        # acc
        total_acc += acc.detach().cpu().item()
        total_loss += loss.detach().cpu().item()
        
        loss = loss / args.gradient_accumulation_steps
        accelerator.backward(loss)
        

        if train_iters % args.gradient_accumulation_steps == 0 or train_iters == args.num_train_iters - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_iters += 1
        postfix.update({
            'loss': f'{total_loss/train_iters:.3f}',
            'acc': f'{total_acc/train_iters:.3f}'
        })
        train_bar.update(1)
        train_bar.set_postfix(postfix)

        # validation
        if train_iters % args.num_valid_steps == 0 or train_iters == args.num_train_iters:
            # valid report
            valid_iters = 0
            total_valid_acc = 0.0
            model.eval()
            valid_bar = tqdm(range(args.num_valid_iters), disable=not accelerator.is_local_main_process)
            valid_bar.set_postfix(acc=f'{total_valid_acc:.4f}')
            for batch in valid_dataloader:
                with torch.no_grad():
                    # batch
                    batch_relation, batch_support, batch_query, batch_label = batch
                    _, acc = model(batch_relation, batch_support, batch_query, batch_label)

                    # acc
                    total_valid_acc += acc.detach().cpu().item()

                    # update
                    valid_iters += 1
                    current_acc = total_valid_acc/valid_iters
                    valid_bar.update(1)
                    valid_bar.set_postfix(acc=f'{current_acc:.4f}')
                    
                    # end validation
                    if valid_iters == args.num_valid_iters:
                        valid_epoch = math.ceil(train_iters/args.num_valid_steps)
                        logger.info(f'The {valid_epoch} validation epoch: {current_acc:.4f}')
                        if best_acc <= current_acc:
                            # save current model
                            best_acc = current_acc
                            patience = 0
                            save_ckpt = os.path.join(args.output_dir, args.save_ckpt)
                            logger.info(f'Save best model {train_iters} in {save_ckpt}')
                            torch.save(model.state_dict(), save_ckpt)
                        else:
                            patience += 1
                        break

            # da valid
            valid_iters = 0
            total_valid_acc = 0.0
            model.eval()
            valid_bar = tqdm(range(args.num_valid_iters), disable=not accelerator.is_local_main_process)
            valid_bar.set_postfix(acc=f'{total_valid_acc:.4f}')
            for batch in valid_dataloader_da:
                with torch.no_grad():
                    # batch
                    batch_relation, batch_support, batch_query, batch_label = batch
                    _, acc = model(batch_relation, batch_support, batch_query, batch_label)

                    # acc
                    total_valid_acc += acc.detach().cpu().item()

                    # update
                    valid_iters += 1
                    current_acc = total_valid_acc/valid_iters
                    valid_bar.update(1)
                    valid_bar.set_postfix(acc=f'{current_acc:.4f}')
                    
                    # end validation
                    if valid_iters == args.num_valid_iters:
                        valid_epoch = math.ceil(train_iters/args.num_valid_steps)
                        logger.info(f'The {valid_epoch} domain adaption validation epoch: {current_acc:.4f}')
                        if best_acc_da <= current_acc:
                            # save current model
                            best_acc_da = current_acc
                            patience_da = 0
                            save_ckpt = os.path.join(args.output_dir, 'best_model_da.pt')
                            logger.info(f'Save best model {train_iters} in {save_ckpt}')
                            torch.save(model.state_dict(), save_ckpt)
                        else:
                            patience_da += 1
                        break

            # save validation
            if args.with_tracking:
                accelerator.log(
                    {
                        "iters": train_iters,
                        "train_loss": total_loss / train_iters,
                        "train_acc": total_acc / train_iters,
                        "valid_acc": current_acc
                    },
                    step=train_iters,
                )
        
        # train finished!
        if train_iters == args.num_train_iters or (patience > 6 and patience_da > 6):
            break
    
    # save last model
    save_ckpt = os.path.join(args.output_dir, 'last_model.pt')
    logger.info(f'Save last model in {save_ckpt}')
    torch.save(model.state_dict(), save_ckpt)
    tokenizer.save_pretrained(args.output_dir)
    
    
    if args.do_test:
        # fewrel1.0 test
        logger.info('Start testing...')
        all_preds = []
        # load best model
        model.load_state_dict(torch.load(os.path.join(args.output_dir, args.save_ckpt)))
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                # batch
                batch_relation, batch_support, batch_query = batch
                _, preds = model(batch_relation, batch_support, batch_query)
                preds = preds.view(-1)
                preds = preds.detach().cpu().numpy().tolist()
                all_preds += preds
        with open(os.path.join(args.output_dir, f'pred-{args.N_way}-{args.K_shot}.json'), 'w') as f:
            all_preds = [int(pred) for pred in all_preds]
            json.dump(all_preds, f)

        # fewrel2.0 test
        logger.info('Start da testing...')
        all_preds = []
        # load best model
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model_da.pt')))
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader_da):
                # batch
                batch_relation, batch_support, batch_query = batch
                _, preds = model(batch_relation, batch_support, batch_query)
                preds = preds.view(-1)
                preds = preds.detach().cpu().numpy().tolist()
                all_preds += preds
        with open(os.path.join(args.output_dir, f'pred-da-{args.N_way}-{args.K_shot}.json'), 'w') as f:
            all_preds = [int(pred) for pred in all_preds]
            json.dump(all_preds, f)

    # end training
    if args.with_tracking:
        accelerator.end_training()

if __name__ == "__main__":
    main()