import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from transformers import BertTokenizer
from model import DialogTransformer
import fire
from collections import defaultdict
from inputter import build_test_dataloaders
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from nltk.util import ngrams
from argparse import ArgumentParser
from tqdm import tqdm

def bleu(predict, target, n):
    return sentence_bleu([target], predict, weights=tuple(1 / n for i in range(n)))

def nist(predict, target, n):
    if len(predict) < n or len(target) < n:
        return 0
    return sentence_nist([target], predict, n)

def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score

def cal_length(sentences):
    sen_length = [len(s) for s in sentences]
    return np.mean(sen_length), np.var(sen_length)

def calculate_metrics(predict, reference):
    reference_len = len(reference)
    predict_len = len(predict)

    #-------------------bleu----------
    bleu_1 = bleu(predict, reference, 1)
    bleu_2 = bleu(predict, reference, 2)
    bleu_3 = bleu(predict, reference, 3)
    bleu_4 = bleu(predict, reference, 4)
    #-------------------nist----------
    nist_2 = nist(predict, reference, 2)
    nist_4 = nist(predict, reference, 4)
    #-------------------meteor----------
    # 这里如果这么写就变成str类型了，新版nltk要求meteor_score传入的对象是可迭代对象
    # predict = " ".join(predict)
    # reference = " ".join(reference)
    meteor_scores = meteor_score([reference], predict)
    return bleu_1, bleu_2, bleu_3, bleu_4, nist_2, nist_4, meteor_scores

def top_k_logits(logits, k):
    """Mask logits so that only top-k logits remain
    """
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_generate(
    top_k = 50,
    temperature = 1.0,
    decoder_path='./transformer.pth',
    config_path="bert-base-chinese"
    ):
    # make sure your model is on GPU

    #------------------------LOAD MODEL-----------------
    print('load the model....')
    tokenizer = BertTokenizer.from_pretrained(config_path)

    model = DialogTransformer()
    model.load_state_dict(torch.load(decoder_path))

    device = torch.device(f"cuda:0")
    model.to(device)
    model.eval()

    print('load success')
    #------------------------END LOAD MODEL--------------

    #------------------------LOAD VALIDATE DATA------------------
    test_dataset,test_dataloader = build_test_dataloaders(tokenizer, batch_size=1,dataset_path='./dataset.json')
    #------------------------END LOAD VALIDATE DATA--------------

    #------------------------START GENERETE-------------------
    update_count = 0
    bleu_1scores = 0
    bleu_2scores = 0
    bleu_3scores = 0
    bleu_4scores = 0
    nist_2scores = 0
    nist_4scores = 0
    
    meteor_scores = 0
    sentences = []
    print('start generating....')
    f = open("trans_dialogue.txt", "w")
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            encoder_input = batch['input_ids'].to(device)
            decoder_input = batch['output_ids'].to(device)
            encoder_mask = batch['input_mask'].to(device)

            outputs = model.encoder(encoder_input, encoder_mask)
            encoder_hidden_states = outputs.last_hidden_state
            prev_pred = decoder_input[:, :1]
            sentence = prev_pred

            # decoding loop
            for i in range(100):
                out = model.decoder(sentence, encoder_hidden_states=encoder_hidden_states, )
                logits = model.linear(out.last_hidden_state)
#                print (logits.size())
                
                logits = logits[:, -1]
                logits = logits.squeeze(1) / temperature
                
                logits = top_k_logits(logits, k=top_k)
                probs = F.softmax(logits, dim=-1)
                prev_pred = torch.multinomial(probs, num_samples=1)
                sentence= torch.cat([sentence, prev_pred], dim=-1)
                if prev_pred[0][0] == 102:
                    break

            predict = tokenizer.convert_ids_to_tokens(sentence[0].tolist())

            encoder_input = encoder_input.squeeze(dim=0)
            encoder_input_num = (encoder_input != 0).sum()
            inputs = tokenizer.convert_ids_to_tokens(encoder_input[:encoder_input_num].tolist())

            decoder_input = decoder_input.squeeze(dim=0)
            decoder_input_num = (decoder_input != 0).sum()

            reference = tokenizer.convert_ids_to_tokens(decoder_input[:decoder_input_num].tolist())
#            print('-'*20 + f"example {update_count}" + '-'*20)
#            print(f"input: {''.join(inputs)}")
#            print(f"output: {''.join(reference)}")
#            print(f"predict: {''.join(predict)}")
            f.write('-'*20 + f"example {update_count}" + '-'*20 + '\n')
            f.write(f"input: {''.join(inputs)}" + "\n")
            f.write(f"output: {''.join(reference)}" + "\n")
            f.write(f"predict: {''.join(predict)}" + "\n\n")

            temp_bleu_1, \
            temp_bleu_2, \
            temp_bleu_3, \
            temp_bleu_4, \
            temp_nist_2, \
            temp_nist_4, \
            temp_meteor_scores = calculate_metrics(predict[1:-1], reference[1:-1])

            bleu_1scores += temp_bleu_1
            bleu_2scores += temp_bleu_2
            bleu_3scores += temp_bleu_3
            bleu_4scores += temp_bleu_4
            nist_2scores += temp_nist_2
            nist_4scores += temp_nist_4

            meteor_scores += temp_meteor_scores
            sentences.append(" ".join(predict[1:-1]))
            update_count += 1

    entro, dist = cal_entropy(sentences)
    mean_len, var_len = cal_length(sentences)

    print(f'avg: {mean_len}, var: {var_len}')
    print(f'entro: {entro}')
    print(f'dist: {dist}')
    print(f'test bleu_1scores: {bleu_1scores / update_count}')
    print(f'test bleu_2scores: {bleu_2scores / update_count}')
    print(f'test bleu_3scores: {bleu_3scores / update_count}')
    print(f'test bleu_4scores: {bleu_4scores / update_count}')
    print(f'test nist_2scores: {nist_2scores / update_count}')
    print(f'test nist_4scores: {nist_4scores / update_count}')
    print(f'test meteor_scores: {meteor_scores / update_count}')


if __name__ == '__main__':
    '''
        计算测试集上困惑度
        '''
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./pretrained/original_pretrained_model_for_bertGPT.pth",
                        help="Trained Model")
    parser.add_argument("--config_path", type=str, default="/root/autodl-tmp/bart-base-chinese",
                        help="Pretrained Model Name")
    parser.add_argument("--data_path", type=str, default="./dataset.json",
                        help="Dataset path")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    print('Generate the response on the test set and caculate the metric......')
    args = parser.parse_args()
    sample_generate(decoder_path=args.model_path,config_path=args.config_path)
