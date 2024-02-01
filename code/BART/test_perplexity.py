import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer,BartConfig
import sys
sys.path.append('..')
from argparse import ArgumentParser
# imports chinese gpt

from inputter import build_test_dataloaders


def calculate_perplexity(args):

    device = torch.device(args.device)

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    config = BartConfig().from_pretrained(args.config_path)
    if args.type=="base":
        from transformers import BartForConditionalGeneration
        model = BartForConditionalGeneration(config)
    else:
        from modeling_shebart import BartForConditionalGeneration
        model = BartForConditionalGeneration(config)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(device)
    model.eval()
    # ernie_tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-health-zh")
    # ernie = AutoModel.from_pretrained("nghuyong/ernie-health-zh")
    # ernie = ernie.to(device)
    # ernie.eval()
    # print('load success')
    #------------------------END LOAD MODEL--------------
    tokenizer = BertTokenizer.from_pretrained(args.config_path)
    _, test_dataloader = build_test_dataloaders(tokenizer,batch_size=args.batch_size,dataset_path=args.data_path)


    #------------------------END LOAD VAL DATA--------------

    perplexity = 0
    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # batch = [item.to(device) for item in batch]
            # input_ids, input_mask, output_ids, output_mask = batch
            # output = model(input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=output_ids,
            #                decoder_attention_mask=output_mask)
            input_ids = batch['input_ids'].to(device)
            output_ids = batch['output_ids'].to(device)
            input_mask = batch['input_mask'].to(device)
            output_mask = batch['output_mask'].to(device)
            if args.type=="kng":

                entity_embeds = [instance.to(device) for instance in batch['entity_embeds']]
                graph_adjs = [instance.to(device) for instance in batch['graph_adjs']]
                entity_mask = [instance.to(device) for instance in batch['entity_mask']]
                output = model(input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=output_ids,
                               decoder_attention_mask=output_mask, entity_embeds=entity_embeds, entity_mask=entity_mask,
                               graph_adjs=graph_adjs)
            else:
                output = model(input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=output_ids,
                               decoder_attention_mask=output_mask)

            target_mask = output_mask[:, 1:].contiguous().reshape(-1).bool()
            logits = output['logits'][:, :-1].contiguous()
            target = output_ids[:, 1:].contiguous()
            probs = logits.view(-1, logits.size(-1))[target_mask]
            labels = target.view(-1)[target_mask]
            criterion = nn.CrossEntropyLoss()
            loss = criterion(probs, labels)
            perplexity += np.exp(loss.item())
            batch_count += 1
    print(f'valid perplexity: {perplexity / batch_count}')

    #------------------------END VAL-------------------


if __name__ == '__main__':
    '''
    计算测试集上困惑度
    '''
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./pretrained/original_pretrained_model_for_bertGPT.pth",
                        help="Pretrained Model Name")
    parser.add_argument("--config_path", type=str, default="/root/autodl-tmp/bart-base-chinese",
                        help="Pretrained Model Name")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for testing")
    parser.add_argument("--data_path", type=str, default="./data/dataset.json",
                        help="Dataset path")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--type", type=str, default="base")
    print('Calculate PPL on the test set......')
    args = parser.parse_args()
    calculate_perplexity(args)

