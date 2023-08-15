import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from transformers import AutoModel

from utils import calculate_acc, calculate_loss

class GateControl(nn.Module):
    def __init__(self, hidden_size: int, bias: bool):
        nn.Module.__init__(self)
        self.pro_gate = nn.Linear(hidden_size, hidden_size, bias)
        self.rel_gate = nn.Linear(hidden_size, hidden_size, bias)
    
    def forward(self, protonet, relation):
        outputs = self.pro_gate(protonet) + self.rel_gate(relation)
        outputs = F.sigmoid(outputs)
        return outputs
       

class ProtoNetworkWithPrompt(nn.Module):
    def __init__(self, config, model_name_or_path, N_way, K_shot, Q_num, Q_na_rate, use_rel=False, use_cp=False, dot_dist=False):
        nn.Module.__init__(self)
        self.config = config
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.use_rel = use_rel
        self.dot = dot_dist
        self.N_way = N_way
        self.K_shot = K_shot
        self.Q_num = Q_num
        self.Q_na_num = Q_num * Q_na_rate
        self.Q_total_num = self.Q_num + self.Q_na_num

        if use_cp:
            ckpt = torch.load("./CP_model/CP")
            #import pdb
            #pdb.set_trace()
            temp = OrderedDict()
            ori_dict = self.encoder.state_dict()
            for name, parameter in ckpt["bert-base"].items():
                if name in ori_dict:
                    temp[name] = parameter
            
            ori_dict.update(temp)
            self.encoder.load_state_dict(ori_dict)
        
        if use_rel:
            # self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
            self.gate = GateControl(config.hidden_size * 2, True)
    
    def __mean_pooling__(self, outputs, attention_mask):
        # outputs: (batch_size * N_way, seq_len, hidden_size)
        attention_mask = attention_mask.unsqueeze(-1).expand(outputs.shape)
        sum_embeddings = torch.sum(outputs * attention_mask, 1)
        sum_mask = torch.sum(attention_mask, dim=1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        mean_embeddings = sum_embeddings/sum_mask
        return mean_embeddings
    
    def __dist__(self, x, y, dim):
        if self.dot:
            dist = torch.cosine_similarity(x, y, dim=dim)
        else:
            dist_func = nn.PairwiseDistance(p=2)
            dist = -dist_func(x, y)
        return dist
    
    def __batch_dist__(self, support, query):
        # support: (batch_size, N_way, hidden_size * 2)
        # query: (batch_size, N_way * Q_total_num, hidden_size * 2)
        # support: (batch_size, N_way * Q_total_num, N_way, hidden_size * 2)
        # query: (batch_size, N_way * Q_total_num, N_way, hidden_size * 2)
        dist = self.__dist__(support.unsqueeze(1), query.unsqueeze(2), -1)
        return dist
    
    def forward(self, relation_set, support_set, query_set, query_label=None):
        hidden_size = self.config.hidden_size
        # relation encoder
        # relation: (batch_size * N_way, hidden_size)
        relation_outputs = self.encoder(**relation_set)
        rel_global = relation_outputs.pooler_output
        rel_local = relation_outputs.last_hidden_state
        batch_size, seq_len, hidden_size = rel_local.shape
        # mean
        rel_local = torch.mean(rel_local, dim=1)
        # rel_local = self.__mean_pooling__(rel_local, relation_set['attention_mask'])
        # relation: (batch_size * N_way, hidden_size * 2)
        relation = torch.cat((rel_global, rel_local), dim=-1)
        # relation: (batch_size, N_way, hidden_size * 2)
        relation = relation.view(-1, self.N_way, hidden_size * 2)

        # support encoder
        # head_pos: (batch_size * N_way * K_shot, 2)
        # tail_pos: (batch_size * N_way * K_shot, 2)
        # mask_pos: (batch_size * N_way * K_shot, 1)
        head_pos = support_set.pop('head_pos')
        tail_pos = support_set.pop('tail_pos')
        mask_pos = support_set.pop('mask_pos')

        sentence_outputs = self.encoder(**support_set)
        sample_num, seq_len, hidden_size = sentence_outputs.last_hidden_state.shape
        # (batch_size * N_way * K_shot, hidden_size)
        sentence_global = sentence_outputs.pooler_output
        sentence_local = sentence_outputs.last_hidden_state[np.arange(sample_num), mask_pos]
        # (batch_size * N_way * K_shot, hidden_size * 2)
        sentence = torch.cat((sentence_global, sentence_local), dim=-1)
        # (batch_size, N_way, K_shot, hidden_size * 2)
        sentence = sentence.view(-1, self.N_way, self.K_shot, hidden_size * 2)

        # (batch_size * N_way * K_shot, hidden_size)
        entity_h = sentence_outputs.last_hidden_state[np.arange(sample_num),head_pos[:,0]]
        entity_t = sentence_outputs.last_hidden_state[np.arange(sample_num),tail_pos[:,0]]
        # (batch_size * N_way * K_shot, hidden_size * 2)
        entity = torch.cat((entity_h, entity_t), dim=-1)
        # (batch_size, N_way, K_shot, hidden_size * 2)
        entity = entity.view(-1, self.N_way, self.K_shot, hidden_size * 2)

        # ablation exp1: only global representation
        # support = sentence_global.view(-1, self.N_way, self.K_shot, hidden_size)
        # ablation exp2: only mask representation
        # support = sentence_local.view(-1, self.N_way, self.K_shot, hidden_size)
        # normal
        support = sentence


        # (batch_size, N_way, hidden_size)
        support = torch.mean(support, dim=2)

        if self.use_rel:
            # (batch_size, N_way, hidden_size)
            assert support.shape == relation.shape
            # gate = self.gate(torch.abs(support - relation))
            # gate = F.sigmoid(gate)
            gate = self.gate(support, relation)
            support = gate * support + (1 - gate) * relation


            # Gate Mechanism
        # if self.use_rel:
        #     # add
        #     # support = torch.where(support > relation, support, relation)
        #     # max pool
        #     # support = support + relation
        #     # mean
        #     # support = support + relation
        #     rel_loss = self.rel_loss(support, relation)

        # query encoder
        # head_pos: (batch_size * N_way * Q_total_num, 2)
        # tail_pos: (batch_size * N_way * Q_total_num, 2)
        # mask_pos: (batch_size * N_way * Q_total_num, 1)
        head_pos = query_set.pop('head_pos')
        tail_pos = query_set.pop('tail_pos')
        mask_pos = query_set.pop('mask_pos')

        query_outputs = self.encoder(**query_set)
        sample_num, seq_len, hidden_size = query_outputs.last_hidden_state.shape

        # (batch_size * N_way * Q_total_num, hidden_size)
        sentence_global = query_outputs.pooler_output
        sentence_local = query_outputs.last_hidden_state[np.arange(sample_num), mask_pos]
        sentence = torch.cat((sentence_global,sentence_local), dim=-1)

        entity_h = query_outputs.last_hidden_state[np.arange(sample_num),head_pos[:,0]]
        entity_t = query_outputs.last_hidden_state[np.arange(sample_num),tail_pos[:,0]]
        # (batch_size * N_way * Q_total_num, hidden_size * 2)
        entity = torch.cat((entity_h, entity_t), dim=-1)

        # ablation exp1: only global representation
        # query = sentence_global
        # ablation exp2: only mask representation
        # query = sentence_local
        # normal
        query = sentence

        if query_label is not None:
            query = query.view(-1, int(self.N_way * self.Q_total_num), hidden_size * 2)
            logits = self.__batch_dist__(support, query)
            preds = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            loss = calculate_loss(logits, query_label)
            acc = calculate_acc(preds, query_label)
            return loss, acc
        else:
            # support: (batch_size, N_way, hidden_size * 2)
            # query: (batch_size, hidden_size * 2)
            logits = self.__dist__(support, query.unsqueeze(1), dim=-1)
            # preds: (batch_size, N_way)
            preds = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            return logits, preds
