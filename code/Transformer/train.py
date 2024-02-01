import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup,set_seed
from inputter import build_train_dataloaders,build_valid_dataloaders,build_test_dataloaders
from tqdm import tqdm
import numpy as np
import os
from argparse import ArgumentParser
from model import DialogTransformer
from Loss import sequence_cross_entropy_with_logits
def train(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    train_dataset, train_dataloader = build_train_dataloaders(tokenizer, args.batch_size,args.dataset_path)
    # test_dataset,test_dataloader = build_test_dataloaders(tokenizer, args.batch_size,args.dataset_path)
    valid_dataset,valid_dataloader = build_valid_dataloaders(tokenizer, args.batch_size,args.dataset_path)
    # 模型
    model = DialogTransformer()
    model = model.to(args.device)

    num_train_optimization_steps = int(
        len(train_dataset) / args.batch_size / args.num_gradients_accumulation) * args.n_epochs
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                num_training_steps=num_train_optimization_steps)
    update_count = 0
    print('start training...')

    for epoch in range(args.n_epochs):
        model.train()
        train_losses = 0
        batch_count = 0
        pbar_train = tqdm(train_dataloader, desc=f'epoch {epoch}')
        for batch in pbar_train:

            input_ids = batch['input_ids'].to(args.device)
            output_ids = batch['output_ids'].to(args.device)
            input_mask = batch['input_mask'].to(args.device)
            output_mask = batch['output_mask'].to(args.device)

            logits = model(input_ids, input_mask, output_ids, output_mask)

            out = logits[:, :-1].contiguous()
            target = output_ids[:, 1:].contiguous()
            target_mask = output_mask[:, 1:].contiguous()

            loss = sequence_cross_entropy_with_logits(out, target, target_mask)
            loss = loss / args.num_gradients_accumulation
            loss.backward()

            update_count += 1

            if update_count % args.num_gradients_accumulation == args.num_gradients_accumulation - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            pbar_train.set_postfix(loss='{:.4f}'.format(loss.item()), lr=optimizer.param_groups[0]['lr'])
            batch_count += 1
            train_losses += loss.item()
        train_losses = train_losses / batch_count

        print("Epoch: %d, Average training loss: %.4f" % (epoch, train_losses))
        model.eval()
        valid_one_epoch(model,valid_dataloader,args.device)




def valid_one_epoch(model,valid_dataloader,device):
    print('Start Validating......')
    perplexity = 0
    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            input_ids = batch['input_ids'].to(device)
            output_ids = batch['output_ids'].to(device)
            input_mask = batch['input_mask'].to(device)
            output_mask = batch['output_mask'].to(device)

            logits = model(input_ids, input_mask, output_ids, output_mask)

            out = logits[:, :-1].contiguous()
            target = output_ids[:, 1:].contiguous()
            target_mask = output_mask[:, 1:].contiguous()

            loss = sequence_cross_entropy_with_logits(out, target, target_mask)
            perplexity += np.exp(loss.item())
            batch_count += 1
    print(f'valid perplexity: {perplexity / batch_count}')

    # 保存模型
    PATH = os.path.join('./checkpoint', "{:.4f}".format(perplexity / batch_count)+"transformer.pth")
    torch.save(model.state_dict(), PATH)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, default=r"G:\huggingface-hub\bert-base-chinese",
                        help="The pretrained model and tokenizer path or name.")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/",
                        help="Saving path of the checkpoint model")
    parser.add_argument("--dataset_path", type=str, default="./dataset.json",
                        help="Path of the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validing")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_gradients_accumulation", type=int, default=8, help="Accumulate gradients on several steps")

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()
    # 固定随机种子
    set_seed(args.seed)

    train(args)
