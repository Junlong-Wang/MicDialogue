import os
import json
from itertools import chain
import sys
import torch
from torch.utils.data import (Dataset, DataLoader)
from torch.nn.utils.rnn import pad_sequence
import networkx as nx
import numpy as np
from tqdm import tqdm
sys.path.append('..')

# 读取json
def load_json2data(file_path):
    '''
    把数据从json中读取
    :param file_path:
    :return:
    '''
    with open(file_path,mode='r',encoding='utf-8') as f:
        data = json.load(f)
    return data

class WBDataset(Dataset):

    def __init__(self, data, tokenizer, raw_entity_embeds,entities,edges,batch_first=True, lm_labels=True):
        self.data = data
        self.tokenizer = tokenizer
        # 这里注意了，两个tokenizer的pad符号都是第一个字符，即id=0
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        # 这是一个List[tensor]
        self.raw_entity_embeds = raw_entity_embeds
        # 结点[[],[[[],[]],[]],...,[]]
        self.entities = entities
        # 边
        self.edges = edges

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.lm_labels:
            history = self.data[index]['dialog'][:-1]
            response = self.data[index]['dialog'][-1]
        else:
            history = self.data[index][:-1]
            response = []
        # 当前对话检索到的多个子图的实体初始嵌入
        dialog_entity_embeds = self.raw_entity_embeds[index]
        # 当前对话检索到的子图的结点、边
        # 这里可能有空的地方
        dialog_nodes = self.entities[index]
        dialog_edges = self.edges[index]

        # graph_embeds:List[tensor(n_entity,768)]:[[n_entity,768]]
        graph_embeds = []
        for utter_entity_embeds in dialog_entity_embeds:
            for subgraph_embeds in utter_entity_embeds:
                graph_embeds.append(subgraph_embeds)

        # graph_adjs:List[np.]:[[n_entity,n_entity]]
        graph_adjs = []
        graph_entity_mask = []

        total_nodes = 0
        for utter_nodes in dialog_nodes:
            total_nodes += sum(len(subgraph) for subgraph in utter_nodes)

        if total_nodes==1:
            # 生成一个邻接矩阵[1.]
            graph_adjs.append(np.zeros([1]))
            # 生成实体mask[1.]
            graph_entity_mask.append(np.ones([1]))
        else:
            for utter_nodes,utter_edges in zip(dialog_nodes,dialog_edges):
                for subgraph_nodes,subgraph_edges in zip(utter_nodes,utter_edges):
                    adj = self.get_adj(subgraph_nodes, subgraph_edges)
                    graph_adjs.append(adj)
                    graph_entity_mask.append([1]*len(subgraph_nodes))



        return self.process(history, response, graph_embeds, graph_adjs,graph_entity_mask)

    def process(self, history, response, graph_embeds, graph_adjs, graph_entity_mask):
        bos, eos = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])

        encoder_input = [bos]
        for his in history:
            encoder_input += his + [eos]
        decoder_input = [bos] + response + [eos]

        instance = {}
        instance["input_ids"] = encoder_input
        instance["output_ids"] = decoder_input
        instance["input_mask"] = [1] * len(encoder_input)
        instance["output_mask"] = [1] * len(decoder_input)

        # 这里的batch_size实际上是子图数量：n_subgraph
        # 填充邻接矩阵和实体嵌入,[n_graph,num_eneities,768]
        entity_embeds = pad_sequence(graph_embeds,batch_first=self.batch_first, padding_value=self.pad)
        # [n_graph,mask]
        graph_entity_mask = pad_sequence([torch.tensor(subgraph_entities_mask, dtype=torch.long)
                                          for subgraph_entities_mask in graph_entity_mask],
                                         batch_first=self.batch_first, padding_value=self.pad)

        # 填充邻接矩阵 graph_adjs:List[np.]:[[n_entity,n_entity]]
        # 实体数
        entities_num = entity_embeds.size(1)
        new_graph_adj = []
        for subgraph_adj in graph_adjs:
            old_entities_num = subgraph_adj.shape[0]
            num_subgraph_adj = np.zeros((entities_num, entities_num))
            num_subgraph_adj[:old_entities_num, :old_entities_num] = subgraph_adj
            adj = torch.from_numpy(num_subgraph_adj).long()
            new_graph_adj.append(adj)
        new_graph_adj = pad_sequence(new_graph_adj, batch_first=self.batch_first)

        instance['entity_embeds'] = entity_embeds
        # instance['entities_mask'] = [1] * (entities_embed.size(0))
        instance['graph_adjs'] = new_graph_adj
        instance['entity_mask'] = graph_entity_mask

        return instance

    def get_adj(self,nodes,edges):
        '''
        构建邻接矩阵
        '''
        # 有结点情况
        new_graph = nx.Graph()
        new_graph.add_nodes_from(nodes)
        new_graph.add_edges_from(edges)
        adj = nx.to_numpy_matrix(new_graph)
        # 填充对角
        # print(f'G的邻接矩阵为：\n {A}')
        #
        # np.fill_diagonal(adj, 1)
        return adj



    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        output_ids = pad_sequence(
            [torch.tensor(instance["output_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        input_mask = pad_sequence(
            [torch.tensor(instance["input_mask"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        output_mask = pad_sequence(
            [torch.tensor(instance["output_mask"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        # 填充邻接矩阵和实体嵌入,[bz,num_eneities,768]
        # entities_embed = pad_sequence(
        #     [instance["entities_embed"] for instance in batch],
        #     batch_first=self.batch_first, padding_value=self.pad)
        # entities_num = entities_embed.size(1)
        # new_adj_batch = []
        # for instance in batch:
        #     old_entities_num = instance['adj'].shape[0]
        #     new_adj_matrix = np.zeros((entities_num, entities_num))
        #     new_adj_matrix[:old_entities_num, :old_entities_num] = instance['adj']
        #     adj = torch.from_numpy(new_adj_matrix).long()
        #     new_adj_batch.append(adj)
        # new_adj_batch = pad_sequence(new_adj_batch,batch_first=self.batch_first)
        #
        # entities_mask = pad_sequence(
        #     [torch.tensor(instance["entities_mask"], dtype=torch.long) for instance in batch],
        #     batch_first=self.batch_first, padding_value=self.pad)

        return {
                "input_ids":input_ids,
                "output_ids":output_ids,
                "input_mask":input_mask,
                "output_mask":output_mask,
                "entity_embeds":[instance['entity_embeds'] for instance in batch],
                "graph_adjs":[instance['graph_adjs'] for instance in batch],
                "entity_mask":[instance['entity_mask'] for instance in batch]
                }



def get_data(tokenizer, dataset_path):
    with open(dataset_path, "r", encoding='utf-8') as f:
        data = json.loads(f.read())

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    
    data = tokenize(data)
    return data

def build_train_dataloaders(tokenizer,batch_size,dataset_path):
    datasets = get_data(tokenizer,dataset_path)
    train_entity_embed = torch.load('/root/autodl-tmp/8kfeature/train_entity_embeds.pt')
    # train_graph = load_json2data('./data/dial_kng_join.json')['train']
    train_nodes = load_json2data('./data/dialog_subgraph/train_nodes.json')
    train_edges = load_json2data('./data/dialog_subgraph/train_edges.json')
    train_dataset= WBDataset(datasets["train"], tokenizer,train_entity_embed,train_nodes,train_edges)
    train_loader = DataLoader(train_dataset, sampler=None, collate_fn=train_dataset.collate,
                              num_workers=1, batch_size=batch_size, shuffle=True)
    return train_dataset, train_loader


def build_valid_dataloaders(tokenizer,batch_size,dataset_path):
    datasets = get_data(tokenizer,dataset_path)
    valid_enetity_embed = torch.load('/root/autodl-tmp/8kfeature/valid_entity_embeds.pt')
    # valid_graph = load_json2data('./data/dial_kng_join.json')['valid']
    valid_nodes = load_json2data('./data/dialog_subgraph/valid_nodes.json')
    valid_edges = load_json2data('./data/dialog_subgraph/valid_edges.json')
    valid_dataset = WBDataset(datasets["valid"], tokenizer,valid_enetity_embed,valid_nodes,valid_edges)
    valid_loader = DataLoader(valid_dataset, sampler=None, collate_fn=valid_dataset.collate,
                              num_workers=1, batch_size=batch_size, shuffle=False)
    return valid_dataset, valid_loader


def build_test_dataloaders(tokenizer, batch_size,dataset_path):
    datasets = get_data(tokenizer,dataset_path)
    test_enetity_embed = torch.load('/root/autodl-tmp/8kfeature/test_entity_embeds.pt')
    # test_graph = load_json2data('./data/dial_kng_join.json')['test']
    test_nodes = load_json2data('./data/dialog_subgraph/test_nodes.json')
    test_edges = load_json2data('./data/dialog_subgraph/test_edges.json')
    test_dataset = WBDataset(datasets["test"], tokenizer,test_enetity_embed,test_nodes,test_edges)

    test_loader = DataLoader(test_dataset, sampler=None, collate_fn=test_dataset.collate,
                              num_workers=1, batch_size=batch_size, shuffle=True)
    return test_dataset, test_loader
if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(r"G:\bart-base-chinese")
    train_dataset, train_dataloader = build_valid_dataloaders(tokenizer, batch_size=2, dataset_path='./uni_dataset.json')
    device = "cuda:0"
    print(len(train_dataloader))
    for batch in tqdm(train_dataloader):
        encoder_input = batch['input_ids'].to(device)
        decoder_input = batch['output_ids'].to(device)
        mask_encoder_input = batch['input_mask'].to(device)
        mask_decoder_input = batch['output_mask'].to(device)
        entity_embeds = batch['entity_embeds']
        graph_adjs = batch['graph_adjs']
        entity_mask = batch['entity_mask']
        assert len(entity_embeds)==len(graph_adjs) and len(entity_embeds)==len(entity_mask)
        print(len(entity_embeds),len(graph_adjs),len(entity_mask))
        # assert entity_embeds.size(1)==adj.size(1) and entities_embed.size(1)==entities_mask.size(1)
        # net = GAT(768,384,768,2,0.3)
        # output = net(entities_embed,adj,entities_mask)
        # print(output.shape)