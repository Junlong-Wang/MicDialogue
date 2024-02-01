import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer,BartConfig
import sys
sys.path.append('..')
from argparse import ArgumentParser
# imports chinese gpt
from Loss import sequence_cross_entropy_with_logits
from inputter import build_test_dataloaders
from model import DialogTransformer

def calculate_perplexity(args):
    print('Start Validating......')
    tokenizer = BertTokenizer.from_pretrained(args.config_path)
    test_dataset, test_dataloader = build_test_dataloaders(tokenizer, args.batch_size, args.data_path)

    model = DialogTransformer()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    perplexity = 0
    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch['input_ids'].to(args.device)
            output_ids = batch['output_ids'].to(args.device)
            input_mask = batch['input_mask'].to(args.device)
            output_mask = batch['output_mask'].to(args.device)

            logits = model(input_ids, input_mask, output_ids, output_mask)

            out = logits[:, :-1].contiguous()
            target = output_ids[:, 1:].contiguous()
            target_mask = output_mask[:, 1:].contiguous()

            loss = sequence_cross_entropy_with_logits(out, target, target_mask)
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
                        help="Trained Model")
    parser.add_argument("--config_path", type=str, default="/root/autodl-tmp/bart-base-chinese",
                        help="Pretrained Model Name")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for testing")
    parser.add_argument("--data_path", type=str, default="./dataset.json",
                        help="Dataset path")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--type", type=str, default="base")
    print('Calculate PPL on the test set......')
    args = parser.parse_args()
    calculate_perplexity(args)

