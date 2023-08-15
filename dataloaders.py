import os
import json
import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import Union, Optional, List, Dict, Any
from transformers.utils import PaddingStrategy
from dataclasses import dataclass


# FSRE dataset with prompt
class FewRelDatasetWithPrompt(Dataset):
    def __init__(self,
                 data_dir: str, 
                 data_file: str, 
                 tokenizer: PreTrainedTokenizer, 
                 padding: Union[bool, str, PaddingStrategy] = True, 
                 max_length: Optional[int] = None,
                 n_way: int = 5, 
                 k_shot: int = 1, 
                 q_num: int = 1, 
                 na_rate: float = 0.0,
                 is_da: bool = False):
        # dataset
        self.pid2name = json.load(open(os.path.join(data_dir, 'pid2name.json')))
        self.datasets = json.load(open(os.path.join(data_dir, data_file)))
        self.relations = list(self.datasets.keys())
        self.is_da = is_da

        # tokenize
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

        # few shot
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_num = q_num
        self.na_rate = na_rate
    
    def __getrelset__(self, relation_set):
        if self.is_da:
            relations = [relation.split('_') for relation in relation_set]        
            rel_inputs = self.tokenizer(relations,
                                        padding=self.padding, 
                                        truncation=True, 
                                        max_length=self.max_length,
                                        return_token_type_ids=False,
                                        is_split_into_words=True)
        else:
            rel_name = [rel[0].lower() for rel in relation_set]
            rel_desc = [rel[1].lower() for rel in relation_set]
            rel_inputs = self.tokenizer(rel_name,
                                        rel_desc, 
                                        padding=self.padding, 
                                        truncation=True, 
                                        max_length=self.max_length,
                                        return_token_type_ids=False)
        return rel_inputs
    
    
    def __getprompt__(self, head, tail):
        # head and tail
        head = head.split(' ')
        tail = tail.split(' ')
        head = [h.lower() for h in head]
        tail = [t.lower() for t in tail]

        mask_token = self.tokenizer.mask_token

        # prompt: [sub] head entity [sub] [MASK] [obj] tail entity [obj]
        prompt = []
        prompt.append('[unused0]')
        prompt += head
        prompt.append('[unused1]')
        prompt.append(mask_token)
        prompt.append('[unused2]')
        prompt += tail
        prompt.append('[unused3]')

        return prompt


    def __getsamset__(self, sample_set):
        # special token
        head_b_token = '[unused0]'
        head_e_token = '[unused1]'
        tail_b_token = '[unused2]'
        tail_e_token = '[unused3]'

        # samples
        inputs = []
        prompts = []
        for sample in sample_set:
            # entity position
            head_bos = sample['h'][-1][0][0]
            head_eos = sample['h'][-1][0][-1]
            tail_bos = sample['t'][-1][0][0]
            tail_eos = sample['t'][-1][0][-1]

            tokens = []
            # insert special around entity
            for index, token in enumerate(sample['tokens']):
                token = token.lower()
                if index == head_bos:
                    tokens.append(head_b_token)
                if index == tail_bos:
                    tokens.append(tail_b_token)
                # tokens
                tokens.append(token)
                if index == head_eos:
                    tokens.append(head_e_token)
                if index == tail_eos:
                    tokens.append(tail_e_token)
            
            inputs.append(tokens)

            prompt = self.__getprompt__(sample['h'][0], sample['t'][0])
            prompts.append(prompt)
        
        # inputs tokenize
        inputs = self.tokenizer(inputs,
                                prompts,
                                padding=self.padding,
                                truncation=True,
                                max_length=self.max_length,
                                is_split_into_words=True,
                                return_token_type_ids=False)

        # head position and tail position
        head_b_token_id = self.tokenizer.convert_tokens_to_ids(head_b_token)
        head_e_token_id = self.tokenizer.convert_tokens_to_ids(head_e_token)
        tail_b_token_id = self.tokenizer.convert_tokens_to_ids(tail_b_token)
        tail_e_token_id = self.tokenizer.convert_tokens_to_ids(tail_e_token)
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        head_pos = []
        tail_pos = []
        mask_pos = []
        for input_ids in inputs['input_ids']:
            head_bos = input_ids.index(head_b_token_id) if head_b_token_id in input_ids else -1
            head_eos = input_ids.index(head_e_token_id) if head_e_token_id in input_ids else -1
            tail_bos = input_ids.index(tail_b_token_id) if tail_b_token_id in input_ids else -1
            tail_eos = input_ids.index(tail_e_token_id) if tail_e_token_id in input_ids else -1
            pos = input_ids.index(mask_token_id) if mask_token_id in input_ids else -1
            head_pos.append([head_bos, head_eos])
            tail_pos.append([tail_bos, tail_eos])
            mask_pos.append(pos)
        
        inputs.update({
            'head_pos': head_pos,
            'tail_pos': tail_pos,
            'mask_pos': mask_pos
        })

        return inputs


    def __getitem__(self, index):
        # In N-way K-shot, a dataloader includes support sets and query sets
        n_way_rels = random.sample(self.relations, self.n_way)
        na_rels = list(filter(lambda x: x not in n_way_rels, self.relations))
        query_na_num = int(self.q_num * self.na_rate)

        # relationsets
        relation_set = []
        # K-shot support sets
        support_set = []
        # Q-query sets
        query_set = []
        # Q-query labels
        query_label = []

        for i, rel in enumerate(n_way_rels):
            if self.is_da:
                relation_set.append(rel)
            else:
                relation_set.append(self.pid2name[rel])

            # random select K+Q samples as support and query sets
            indices = np.random.choice(np.arange(len(self.datasets[rel])), 
                                       size=self.k_shot + self.q_num,
                                       replace=False)
            
            count = 0
            for j in indices:
                if count < self.k_shot:
                    # support sets
                    support_set.append(self.datasets[rel][j])
                else:
                    # query set
                    query_set.append(self.datasets[rel][j])
                count += 1
            
            # label: i and number: q_num
            query_label += [i] * self.q_num
        
        # negtive samples in query sets
        na_rel = np.random.choice(na_rels, query_na_num, replace=True)
        for rel in na_rel:
            index = np.random.choice(np.arange(len(self.datasets[rel])),
                                     1,
                                     False)[0]
            query_set.append(self.datasets[rel][index])
        # negtive samples label = N
        query_label += [self.n_way] * query_na_num


        # tokenize: support_set, query_set, query_label
        # relation_set: (N, seq_len)
        relation_set = self.__getrelset__(relation_set)
        # support_set: (N*K, seq_len)
        support_set = self.__getsamset__(support_set)
        # query_set: (N*Q, seq_len)
        query_set = self.__getsamset__(query_set)

        item = {}
        item.update({
            'relation': relation_set,
            'support': support_set,
            'query': query_set,
            'label': query_label
        })

        return item

    def __len__(self):
        return int(1e8)


# FSRE test dataset with prompt
class FewRelTestDatasetWithPrompt(Dataset):
    def __init__(self,
                 data_dir: str, 
                 data_file: str, 
                 tokenizer: PreTrainedTokenizer, 
                 padding: Union[bool, str, PaddingStrategy] = True, 
                 max_length: Optional[int] = None,
                 is_da: bool = False):
        # dataset
        self.pid2name = json.load(open(os.path.join(data_dir, 'pid2name.json')))
        self.datasets = json.load(open(os.path.join(data_dir, data_file)))
        self.is_da = is_da

        # tokenize
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
    
    def __getrelset__(self, relation_set):
        if self.is_da:
            relations = [relation.split('_') for relation in relation_set]
            rel_inputs = self.tokenizer(relations,
                                        padding=self.padding, 
                                        truncation=True, 
                                        max_length=self.max_length,
                                        return_token_type_ids=False,
                                        is_split_into_words=True)
        else:
            rel_name = [rel[0].lower() for rel in relation_set]
            rel_desc = [rel[1].lower() for rel in relation_set]
            rel_inputs = self.tokenizer(rel_name,
                                        rel_desc, 
                                        padding=self.padding, 
                                        truncation=True, 
                                        max_length=self.max_length,
                                        return_token_type_ids=False)
        return rel_inputs
    
    
    def __getprompt__(self, head, tail):
        # head and tail
        head = head.split(' ')
        tail = tail.split(' ')
        head = [h.lower() for h in head]
        tail = [t.lower() for t in tail]

        # special token
        mask_token = self.tokenizer.mask_token

        # prompt: [sub] head entity [sub] [MASK] [obj] tail entity [obj]
        prompt = []
        prompt.append('[unused0]')
        prompt += head
        prompt.append('[unused1]')
        prompt.append(mask_token)
        prompt.append('[unused2]')
        prompt += tail
        prompt.append('[unused3]')

        return prompt


    def __getsamset__(self, sample_set):
        # special token
        head_b_token = '[unused0]'
        head_e_token = '[unused1]'
        tail_b_token = '[unused2]'
        tail_e_token = '[unused3]'

        # samples
        inputs = []
        prompts = []
        for sample in sample_set:
            # entity position
            head_bos = sample['h'][-1][0][0]
            head_eos = sample['h'][-1][0][-1]
            tail_bos = sample['t'][-1][0][0]
            tail_eos = sample['t'][-1][0][-1]

            tokens = []
            # insert special around entity
            for index, token in enumerate(sample['tokens']):
                token = token.lower()
                if index == head_bos:
                    tokens.append(head_b_token)
                if index == tail_bos:
                    tokens.append(tail_b_token)
                # tokens
                tokens.append(token)
                if index == head_eos:
                    tokens.append(head_e_token)
                if index == tail_eos:
                    tokens.append(tail_e_token)
            
            inputs.append(tokens)

            prompt = self.__getprompt__(sample['h'][0], sample['t'][0])
            prompts.append(prompt)
        
        # inputs tokenize
        inputs = self.tokenizer(inputs,
                                prompts,
                                padding=self.padding,
                                truncation=True,
                                max_length=self.max_length,
                                is_split_into_words=True,
                                return_token_type_ids=False)

        # head position and tail position
        head_b_token_id = self.tokenizer.convert_tokens_to_ids(head_b_token)
        head_e_token_id = self.tokenizer.convert_tokens_to_ids(head_e_token)
        tail_b_token_id = self.tokenizer.convert_tokens_to_ids(tail_b_token)
        tail_e_token_id = self.tokenizer.convert_tokens_to_ids(tail_e_token)
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        head_pos = []
        tail_pos = []
        mask_pos = []
        for input_ids in inputs['input_ids']:
            head_bos = input_ids.index(head_b_token_id) if head_b_token_id in input_ids else -1
            head_eos = input_ids.index(head_e_token_id) if head_e_token_id in input_ids else -1
            tail_bos = input_ids.index(tail_b_token_id) if tail_b_token_id in input_ids else -1
            tail_eos = input_ids.index(tail_e_token_id) if tail_e_token_id in input_ids else -1
            pos = input_ids.index(mask_token_id) if mask_token_id in input_ids else -1
            head_pos.append([head_bos, head_eos])
            tail_pos.append([tail_bos, tail_eos])
            mask_pos.append(pos)
        
        inputs.update({
            'head_pos': head_pos,
            'tail_pos': tail_pos,
            'mask_pos': mask_pos
        })

        return inputs


    def __getitem__(self, index):
        data = self.datasets[index]

        relation_set = []
        support_set = []
        query_set = []

        # N
        for relation in data['relation']:
            if self.is_da:
                relation_set.append(relation)
            else:
                relation_set.append(self.pid2name[relation])

        # N * K
        for support in data['meta_train']:
            support_set += support
        
        # N * 1
        query_set.append(data['meta_test'])

        # tokenize: support_set, query_set, query_label
        # relation set
        relation_set = self.__getrelset__(relation_set)

        # support_set: (N*K, seq_len)
        support_set = self.__getsamset__(support_set)

        # query_set: (N*1, seq_len)
        query_set = self.__getsamset__(query_set)

        item = {}
        item.update({
            'relation': relation_set,
            'support': support_set,
            'query': query_set,
        })

        return item

    def __len__(self):
        return len(self.datasets)

# datacollator
@dataclass
class DataCollatorForFewRelWithPrompt():
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = 'pt'

    def __call__(self, features:List[Dict[str, Any]]) -> Dict[str, Any]:
        # (batch_size, N-way, seq_len)
        batch_relation = {}
        for k in features[0]['relation'].keys():
            batch_relation.setdefault(k, [])
            for item in features:
                batch_relation[k] += item['relation'][k]
        
        # (batch_size, N-way * K-shot, seq_len)
        batch_support = {}
        for k in features[0]['support'].keys():
            batch_support.setdefault(k, [])
            for item in features:
                batch_support[k] += item['support'][k]

        # (batch_size, N-way * (Q-num + Q-na-num), seq_len)
        batch_query = {}
        for k in features[0]['query'].keys():
            batch_query.setdefault(k, [])
            for item in features:
                batch_query[k] += item['query'][k]

        batch_label = [item['label'] for item in features] if 'label' in features[0].keys() else None

        batch_relation = self.tokenizer.pad(batch_relation,
                                            padding=self.padding,
                                            max_length=self.max_length,
                                            pad_to_multiple_of=self.pad_to_multiple_of,
                                            return_tensors=self.return_tensors)

        batch_support = self.tokenizer.pad(batch_support,
                                           padding=self.padding,
                                           max_length=self.max_length,
                                           pad_to_multiple_of=self.pad_to_multiple_of,
                                           return_tensors=self.return_tensors)
        
        batch_query = self.tokenizer.pad(batch_query,
                                         padding=self.padding,
                                         max_length=self.max_length,
                                         pad_to_multiple_of=self.pad_to_multiple_of,
                                         return_tensors=self.return_tensors)
        
        if batch_label is None:
            return batch_relation, batch_support, batch_query
        else:
            batch_label = torch.tensor(batch_label)
            return batch_relation, batch_support, batch_query, batch_label

def collate_fn(data):
    # (batch_size * N-way, seq_len)
    batch_relation = {}
    for k in data[0]['relation'].keys():
        batch_relation.setdefault(k, [])
        for item in data:
            batch_relation[k] += item['relation'][k]
    batch_relation = {k:torch.tensor(v) for k, v in batch_relation.items()}

    # (batch_size, N-way * K-shot, seq_len)
    batch_support = {}
    for k in data[0]['support'].keys():
        batch_support.setdefault(k, [])
        for item in data:
            batch_support[k] += item['support'][k]
    batch_support = {k:torch.tensor(v) for k, v in batch_support.items()}

    # (batch_size, N-way * (Q-num + Q-na-num), seq_len)
    batch_query = {}
    for k in data[0]['query'].keys():
        batch_query.setdefault(k, [])
        for item in data:
            batch_query[k] += item['query'][k]
    batch_query = {k:torch.tensor(v) for k, v in batch_query.items()}

    is_test = False if 'label' in data[0].keys() else True

    if is_test:
        return batch_relation, batch_support, batch_query
    else:
        # (batch_size, N-way * (Q-num + Q-na-num))
        batch_label = [item['label'] for item in data]
        batch_label = torch.tensor(batch_label)

        return batch_relation, batch_support, batch_query, batch_label

# debug
def main():
    data_dir = './data'
    valid_file = 'val_pubmed.json'
    test_file = 'test_pubmed_input-5-1.json'

    from transformers import AutoTokenizer
    special_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]']
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', additional_special_tokens=special_tokens)

    valid_dataset = FewRelDatasetWithPromptDA(data_dir, valid_file, tokenizer, True, 128)
    test_dataset = FewRelTestDatasetWithPromptDA(data_dir, test_file, tokenizer, True, 128)
    collate_fn = DataCollatorForFewRelWithPrompt(tokenizer, max_length=128)

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=4,
                                  shuffle=False,
                                  collate_fn=collate_fn)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=4,
                                 shuffle=False,
                                 collate_fn=collate_fn)
    
    for index, batch in enumerate(valid_dataloader):
        batch_relation, batch_support, batch_query, batch_label = batch
        if index >= 3:
            break
    
    for batch in test_dataloader:
        batch_relation, batch_support, batch_query = batch


if __name__ == '__main__':
    main()