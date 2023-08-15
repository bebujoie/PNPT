import argparse
import torch
import torch.nn as nn

from transformers import SchedulerType

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a relation classification task")
    # task
    parser.add_argument(
        "--task_name",
        type=str,
        default='prompt-fsre',
        help="The name of the task to train on.",
    )
    parser.add_argument(
        "--N_way",
        type=int,
        default=10,
        help="The number of relation class.",
    )
    parser.add_argument(
        "--K_shot",
        type=int,
        default=5,
        help="The number of support sample for each class.",
    )
    parser.add_argument(
        "--Q_num",
        type=int,
        default=1,
        help="The number of query sample for each class.",
    )
    parser.add_argument(
        "--Q_na_rate",
        type=float,
        default=0.0,
        help="The rate of na-relation in query sets.",
    )
    parser.add_argument(
        "--use_cp",
        action='store_true',
        help="Using CP model as encoder.",
    )
    parser.add_argument(
        "--use_rel",
        action='store_true',
        help="Using external relation information for prototype.",
    )
    parser.add_argument(
        "--dot_dist",
        action='store_true',
        help="Using dot distance for metric.",
    )

    # train
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default='./data', 
        help="The data directory contains train file and test file etc."
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default='train_wiki.json', 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--valid_file", 
        type=str, 
        default='val_wiki.json', 
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--valid_file_da", 
        type=str, 
        default='val_pubmed.json', 
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default='test_wiki.json', 
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--test_file_da", 
        type=str, 
        default='test_wiki.json', 
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    
    # model
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_valid_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_iters", 
        type=int, 
        default=30000, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--num_valid_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--num_valid_iters",
        type=int,
        default=1000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--do_test", 
        action='store_true',
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--save_ckpt", 
        type=str,
        default='best_model.pt',
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.valid_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.valid_file is not None:
            extension = args.valid_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args

def calculate_acc(preds, labels):
    # preds: (batch_size, N_way * Q_total_num)
    # labels: (batch_size, N-way * (Q-num + Q-na-num))
    acc = torch.mean((preds.view(-1) == labels.view(-1)).type(torch.FloatTensor))
    return acc

def calculate_loss(logits, label):
    loss_func = nn.CrossEntropyLoss()
    N_way = logits.size(-1)
    loss = loss_func(logits.view(-1,N_way), label.view(-1))
    return loss
