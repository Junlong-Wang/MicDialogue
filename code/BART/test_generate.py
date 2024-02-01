import torch
import torch.nn as nn
from transformers import BertTokenizer,BartConfig
from modeling_shebart import BartForConditionalGeneration
import json
import sys
sys.path.append('..')
from tqdm import tqdm
from inputter import build_test_dataloaders
from argparse import ArgumentParser

# 加载已经训练好的模型
def get_model(config_path,model_path,type):
    config = BartConfig.from_pretrained(config_path)
    if type=="base":
        from transformers import BartForConditionalGeneration
        model = BartForConditionalGeneration(config)
    else:
        from modeling_shebart import BartForConditionalGeneration
        model = BartForConditionalGeneration(config)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    return model

def top_k_logits(logits, k):
    """Mask logits so that only top-k logits remain
    """
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def demo_generate(args):
    tokenizer = BertTokenizer.from_pretrained(args.config_path)
    # make sure your model is on GPU
    device = torch.device(args.device)

    # ------------------------LOAD MODEL-----------------
    model = get_model(args.config_path, args.model_path, args.type)
    model = model.to(device)
    model.eval()

    _, test_dataloader = build_test_dataloaders(tokenizer, batch_size=4, dataset_path=args.data_path)
    # ------------------------END LOAD VALIDATE DATA--------------

    # ------------------------START SAMPLE GENERETE-------------------
    pred_seqs = []
    ctx_seqs = []
    ref_seqs = []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['input_mask'].to(device)
            output_ids = batch['output_ids']
            if args.type=="kng":
                entity_embeds = [instance.to(device) for instance in batch['entity_embeds']]
                graph_adjs = [instance.to(device) for instance in batch['graph_adjs']]
                entity_mask = [instance.to(device) for instance in batch['entity_mask']]

                predict = model.generate(input_ids=input_ids, attention_mask=input_mask,
                                         entity_embeds=entity_embeds,graph_adjs=graph_adjs,entity_mask=entity_mask,
                                         max_length=args.max_length,early_stopping=True,no_repeat_ngram_size=2,
                                         num_beams=2, do_sample=True,top_k=args.top_k)
            else:
                predict = model.generate(input_ids=input_ids, attention_mask=input_mask,
                                         max_length=args.max_length, early_stopping=True,
                                         num_beams=5, do_sample=True, top_k=args.top_k)

            pred_txt = tokenizer.batch_decode(predict, skip_special_tokens=True)
            ref_txt = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            ctx_txt = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            # 去掉PAD符
            # for ctx, mask in zip(ctx_txt, input_mask):
            #     input_num = (mask != 0).sum()
            #     ctx = ctx[:input_num]

            pred_seqs.extend(pred_txt)
            ref_seqs.extend(ref_txt)
            ctx_seqs.extend(ctx_txt)
            # reference = tokenizer.convert_ids_to_tokens(target[:target_num].tolist())

            # encoder_input = encoder_input.squeeze(dim=0)
            # encoder_input_num = (encoder_input != 0).sum()
            # inputs = tokenizer.convert_ids_to_tokens(encoder_input[:encoder_input_num].tolist())
            # pred_seqs.append(["".join(inputs[1:-1]), "".join(predict[1:-1]), "".join(reference[1:-1])])
            assert len(ctx_seqs) == len(ref_seqs) == len(pred_seqs)
    return ctx_seqs, ref_seqs, pred_seqs
def kng_generate_sentence(args):
    tokenizer = BertTokenizer.from_pretrained(args.config_path)
    # make sure your model is on GPU
    device = torch.device(args.device)

    # ------------------------LOAD MODEL-----------------
    model = get_model(args.config_path,args.model_path,args.type)
    model = model.to(device)
    model.eval()

    _, test_dataloader = build_test_dataloaders(tokenizer, batch_size=1, dataset_path=args.data_path)
    # ------------------------END LOAD VALIDATE DATA--------------

    # ------------------------START SAMPLE GENERETE-------------------
    pred_seqs = []
    ctx_seqs = []
    ref_seqs = []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['input_mask'].to(device)
            output_ids = batch['output_ids']
            entity_embeds = [instance.to(device) for instance in batch['entity_embeds']]
            graph_adjs = [instance.to(device) for instance in batch['graph_adjs']]
            entity_mask = [instance.to(device) for instance in batch['entity_mask']]
            # 开始符
            predict = []
            prev_pred = tokenizer.bos_token_id
            predict.append(prev_pred)
            predict_ids = torch.tensor(predict,dtype=torch.long,device=args.device).unsqueeze(0)
            predict_mask= torch.ones_like(predict_ids,device=args.device)
            # print(predict_ids.shape)
            # print(predict_mask.shape)
            for i in range(args.max_length):
                outputs = model(input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=predict_ids,
                               decoder_attention_mask=predict_mask, entity_embeds=entity_embeds, entity_mask=entity_mask,
                               graph_adjs=graph_adjs)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_scores = top_k_logits(next_token_logits,args.top_k)
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                if next_token == tokenizer.eos_token_id:
                    break
                predict_ids = torch.cat([predict_ids,next_token],dim=1)
                predict_mask = torch.ones_like(predict_ids)

            pred_txt = tokenizer.batch_decode(predict_ids, skip_special_tokens=True)
            # print(pred_txt)
            ref_txt = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            ctx_txt = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            # 去掉PAD符
            # for ctx, mask in zip(ctx_txt, input_mask):
            #     input_num = (mask != 0).sum()
            #     ctx = ctx[:input_num]

            pred_seqs.extend(pred_txt)
            ref_seqs.extend(ref_txt)
            ctx_seqs.extend(ctx_txt)
            # reference = tokenizer.convert_ids_to_tokens(target[:target_num].tolist())

            # encoder_input = encoder_input.squeeze(dim=0)
            # encoder_input_num = (encoder_input != 0).sum()
            # inputs = tokenizer.convert_ids_to_tokens(encoder_input[:encoder_input_num].tolist())
            # pred_seqs.append(["".join(inputs[1:-1]), "".join(predict[1:-1]), "".join(reference[1:-1])])
            assert len(ctx_seqs) == len(ref_seqs) == len(pred_seqs)
    return ctx_seqs, ref_seqs, pred_seqs



def generate_sentences(args):
    tokenizer = BertTokenizer.from_pretrained(args.config_path)
    # make sure your model is on GPU
    device = torch.device(args.device)

    #------------------------LOAD MODEL-----------------
    model = get_model(args.config_path,args.model_path,args.type)
    model = model.to(device)
    model.eval()

    _, test_dataloader = build_test_dataloaders(tokenizer,batch_size=1,dataset_path=args.data_path)
    #------------------------END LOAD VALIDATE DATA--------------


    #------------------------START SAMPLE GENERETE-------------------
    pred_seqs = []
    ctx_seqs = []
    ref_seqs = []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            batch = [item.to(device) for item in batch]
            input_ids,input_mask,output_ids,output_mask = batch

            predict = model.generate(input_ids=input_ids, attention_mask=input_mask, max_length=args.max_length,
                                    early_stopping=True,no_repeat_ngram_size=2, num_beams=5, do_sample=True, top_k=args.top_k)

            pred_txt = tokenizer.batch_decode(predict,skip_special_tokens=True)
            ref_txt = tokenizer.batch_decode(output_ids,skip_special_tokens=True)
            ctx_txt = tokenizer.batch_decode(input_ids,skip_special_tokens=False)
            # 去掉PAD符
            for ctx,mask in zip(ctx_txt,input_mask):
                input_num = (mask!=0).sum()
                ctx = ctx[:input_num]

            pred_seqs.extend(pred_txt)
            ref_seqs.extend(ref_txt)
            ctx_seqs.extend(ctx_txt)
            # reference = tokenizer.convert_ids_to_tokens(target[:target_num].tolist())

            # encoder_input = encoder_input.squeeze(dim=0)
            # encoder_input_num = (encoder_input != 0).sum()
            # inputs = tokenizer.convert_ids_to_tokens(encoder_input[:encoder_input_num].tolist())
            # pred_seqs.append(["".join(inputs[1:-1]), "".join(predict[1:-1]), "".join(reference[1:-1])])
            assert len(ctx_seqs)==len(ref_seqs)==len(pred_seqs)
    return ctx_seqs,ref_seqs,pred_seqs

    #------------------------END SAMPLE GENERETE-------------------


def sample_generate(args):

    ctx_seqs,ref_seqs,pred_seqs = demo_generate(args=args)

    Dialog_list = []
    index = 0
    with open('./generate_sentences.json', 'w', encoding='utf-8') as f:
        for ctx,ref,pred in zip(ctx_seqs,ref_seqs,pred_seqs):
            cases = dict()
            cases['id'] = index
            index += 1
            cases['input'] = ctx
            cases['predict'] = pred
            cases['reference'] = ref
            Dialog_list.append(cases)
        json.dump(Dialog_list, f, ensure_ascii = False, indent = 4)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoint/31.8362bart.pth",
                        help="Pretrained Model Name or Path")
    parser.add_argument("--config_path", type=str, default=r"G:\bart-base-chinese",
                        help="Pretrained Model Name or Path")
    parser.add_argument("--data_path",type=str,default="./data/dataset.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top_k",type=int,default="50")
    parser.add_argument("--max_length", type=int, default="150")
    parser.add_argument("--type", type=str,default="base")
    print('Inference on the test set......')
    args = parser.parse_args()
    sample_generate(args)

