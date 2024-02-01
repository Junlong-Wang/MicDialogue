# -*- coding: utf-8 -*-
import os
import json

import torch
from torch.utils.data import DataLoader
from transformers import cached_path

import preprocess
from od.inputters.dataset_wb import WBDataset, WBdistDataset

LCCC_URL = "https://coai-dataset.oss-cn-beijing.aliyuncs.com/CleanWB.zip"
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]","[MED]","[/MED]"]


def get_data(tokenizer, dataset_path, dataset_cache, logger):
    """ Get tokenized dataset from COTK or cache."""
    dataset_path = dataset_path or LCCC_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
        samples = None
    else:
        logger.info("Download dataset from %s", dataset_path)
        cache_file = cached_path(dataset_path)
        dataset = {"train":[],"valid":[],"test":[]}
        with open(cache_file, "r", encoding="utf-8") as f:
            raw_data = json.loads(f.read())
            samples = [{k: v[:5]} for k, v in raw_data.items()]
        for key,value in raw_data.items():
            for dialog in value:
                dataset[key].append(dialog['dialog'])

        # print(dialog_dataset["train"][0])

        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset, samples

def get_knowledge(train_know_path,valid_know_path,tokenizer):
    train_know = preprocess.load_json2data(train_know_path)
    valid_know = preprocess.load_json2data(valid_know_path)
    new_train_know = []
    new_valid_know = []
    for dialog in train_know:
        new_dialog_know = []
        for utterance in dialog:
            new_dialog_know.extend(utterance)
        new_train_know.append(new_dialog_know)
    for dialog in valid_know:
        new_dialog_know = []
        for utterance in dialog:
            new_dialog_know.extend(utterance)
        new_valid_know.append(new_dialog_know)

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    train_know = tokenize(new_train_know)
    valid_know = tokenize(new_valid_know)
    return train_know,valid_know

def build_dataloaders(args, tokenizer, logger):
    logger.info("Build train and validation dataloaders")
    train_know,valid_know = get_knowledge(args.train_know_path,args.valid_know_path,tokenizer)
    logger.info("Load Knowledge Successfully!")
    datasets, raw_samples = get_data(tokenizer, args.data_path, args.dataset_cache, logger)
    train_dataset, valid_dataset = WBDataset(datasets["train"],train_know,tokenizer), WBDataset(datasets["valid"],valid_know,tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              collate_fn=train_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler,
                              collate_fn=valid_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.valid_batch_size,
                              shuffle=False)

    return train_loader, valid_loader, train_sampler, valid_sampler


def build_dist_loaders(args, tokenizer, logger):
    logger.info("Build train and validation dataloaders")

    train_dataset = WBdistDataset(tokenizer, data_path=args.train_path)
    valid_dataset = WBdistDataset(tokenizer, data_path=args.valid_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset,
                              collate_fn=valid_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=valid_sampler,
                              batch_size=args.valid_batch_size,
                              shuffle=False)
    return train_loader, valid_loader, train_sampler, valid_sampler

if __name__ == '__main__':
    from transformers import BertTokenizer
    import logging
    logger = logging.getLogger(__file__)
    model_check_point = "../../pretrained/CDial-GPT"
    tokenizer_class = BertTokenizer
    # train_know = preprocess.load_json2data('../../data/train_knowledge_join.json')
    # valid_know = preprocess.load_json2data('../../data/valid_knowledge_join.json')
    tokenizer = tokenizer_class.from_pretrained(model_check_point, do_lower_case=True,never_split=["[speaker1]", "[speaker2]"])
    tokenizer.add_special_tokens({'additional_special_tokens':['[MED]','[/MED]']})
    train_know,valid_know = get_knowledge('../../data/train_knowledge_join.json','../../data/valid_knowledge_join.json',tokenizer)


    datasets, raw_samples = get_data(tokenizer, '../../data/dataset.json', "dataset_cache", logger)

    train_dataset, valid_dataset = WBDataset(datasets["train"], train_know,tokenizer), WBDataset(datasets["valid"], valid_know,tokenizer)

    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate,
                              num_workers=8,
                              batch_size=1,
                              shuffle=False)
    valid_loader = DataLoader(valid_dataset,
                              collate_fn=valid_dataset.collate,
                              num_workers=8,
                              batch_size=1,
                              shuffle=False)

    for idx,batch in enumerate(train_loader):
        input_ids, token_type_ids, lm_labels = batch
        # input = input_ids[0].numpy().tolist()
        lens = input_ids.shape[1]
        if lens>512:
            print("大！！")

        # lable = lm_labels[0].numpy().tolist()
        # text = tokenizer.decode(input)
        # response = tokenizer.decode(lable)
        # print(text)
        # print(response)
        # print(token_type_ids)
        # break