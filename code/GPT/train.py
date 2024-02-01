# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import json
import os
import math
import shutil
import logging
import random
from pprint import pformat
from argparse import ArgumentParser
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear, LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (OpenAIGPTLMHeadModel, OpenAIGPTConfig, GPT2LMHeadModel, GPT2Config,
                          WEIGHTS_NAME, CONFIG_NAME, AdamW, BertTokenizer)

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.nist_score import sentence_nist
from collections import defaultdict

from od.inputters.inputter import build_dataloaders, build_dist_loaders

logger = logging.getLogger(__file__)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)


# 分布式训练和评估
def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()



def train():
    # if os.path.exists('compare_txt'):
    #     shutil.rmtree('compare_txt')
    # os.mkdir('compare_txt')
    
    parser = ArgumentParser()
    parser.add_argument('--gpt2', action='store_true', default=True, help="use gpt2")
    parser.add_argument("--model_checkpoint", type=str, default="./pretrained/CDial-GPT", help="Path or URL of the model")
    parser.add_argument("--from_step", type=int, default=-1, help="Init learning rate from this step")
    parser.add_argument('--pretrained', action='store_true', default=True, help="If False train from scratch")
    parser.add_argument("--data_path", type=str, default="./data/dataset.json",
                        help="Path or url of the dataset. ")
    parser.add_argument("--train_know_path", type=str, default="./data/train_knowledge_join.json",
                        help="Path or url of the knowledge dataset. ")
    parser.add_argument("--valid_know_path", type=str, default="./data/valid_knowledge_join.json",
                        help="Path or url of the knowledge dataset. ")

    parser.add_argument("--train_path", type=str, default="data/toy_train.txt",
                        help="Path of the train dataset for dist dataset. ")
    parser.add_argument("--valid_path", type=str, default="data/toy_valid.txt",
                        help="Path of the valid dataset for dist dataset. ")
    parser.add_argument("--dataset_cache", type=str, default="dataset_cache",
                        help="Path or url of the dataset cache")
    parser.add_argument('--log_file', '-log_file', type=str, default="", help="Output logs to a file under this path")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of subprocesses for data loading")
    parser.add_argument("--n_epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--max_history", type=int, default=15, help="Number of previous exchanges to keep in history")
    parser.add_argument("--scheduler", type=str, default="linear", choices=['noam', 'linear'], help="method of optim")
    parser.add_argument("--n_emd", type=int, default=768, help="Number of n_emd in config file (for noam)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--eval_before_start", action='store_true', default=True,
                        help="If true start with a first evaluation before training")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
    parser.add_argument("--valid_steps", type=int, default=5000, help="Perfom validation every X steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process.
    # logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
    config_class = OpenAIGPTConfig if not args.gpt2 else GPT2Config
    tokenizer_class = BertTokenizer
    if args.pretrained:
        tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=True, never_split=["[speaker1]", "[speaker2]"])
        model = model_class.from_pretrained(args.model_checkpoint)
    else:
        tokenizer = tokenizer_class(os.path.join(args.model_checkpoint, "vocab.txt"), do_lower_case=True, never_split=["[speaker1]", "[speaker2]"])
        config = config_class.from_json_file(os.path.join(args.model_checkpoint, CONFIG_NAME))
        model = model_class(config)

    tokenizer.add_special_tokens({'additional_special_tokens':['[MDK]',"[/MED]"] })
    model.resize_token_embeddings(len(tokenizer))

    # 计算网络参数
    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1e6))

    model.to(args.device)

    optimizer = AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, correct_bias=True)

    logger.info("Prepare datasets")
    loader_class = build_dist_loaders if not args.data_path else build_dataloaders
    train_loader, val_loader, train_sampler, valid_sampler = loader_class(args, tokenizer, logger)
    print(len(train_loader))
    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Training function and trainer
    def update(engine, batch):
        input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
        model.train()

        (lm_loss), *_ = model(input_ids, labels=lm_labels, token_type_ids=token_type_ids)
        loss = lm_loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item(), optimizer.param_groups[0]['lr']

    trainer = Engine(update)

    # Evaluation function and eval (eval output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
            # input_ids, token_type_ids, lm_labels: torch.Size([4, 270])
            lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)  # torch.Size([4, 270, 13088])
            # lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))  # torch.Size([1076, 13088])
            # lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)  # torch.Size([1076])
            # return lm_logits_flat_shifted, lm_labels_flat_shifted
            shift_logits = lm_logits[..., :-1, :].contiguous()  # torch.Size([4, 269, 13088])
            shift_labels = lm_labels[..., 1:].contiguous()  # torch.Size([4, 269])
            return shift_logits, shift_labels

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Evaluation during training
    @trainer.on(Events.ITERATION_STARTED)
    def log_iterations(engine):
        # if engine.state.iteration % max(int(0.1 * len(train_loader)), 1) == 0:
        if engine.state.iteration % args.valid_steps == 0:
            evaluator.run(val_loader)

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # noam decrease the learning rate
    # model_size = model.config.n_embd
    model_size = args.n_emd
    noam_lambda = lambda step: (
            model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * args.warmup_steps ** (-1.5)))
    noam_scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda, last_epoch=args.from_step)
    scheduler = LRScheduler(noam_scheduler)
    if args.scheduler == "linear":
        scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    def cal_metrics(logits, labels, metric_):
        metrics_ = ['BLEU-4', 'BLEU-2', 'NIST-4', 'NIST-2', 'METEOR']
        _, preds = logits.max(dim=-1)
        turns = len(labels)
        bleu_2 = 0
        bleu_4 = 0
        meteor = 0
        nist_2 = 0
        nist_4 = 0
        for index in range(turns):
            pred_utt_ = preds[index]
            target_utt_ = labels[index]

            mask = target_utt_.ne(-1)
            pred_utt = torch.masked_select(pred_utt_, mask).tolist()
            target_utt = torch.masked_select(target_utt_, mask).tolist()

            pred_utt = tokenizer.convert_ids_to_tokens(pred_utt)
            target_utt = tokenizer.convert_ids_to_tokens(target_utt)
            min_len = min(len(pred_utt), len(target_utt))
            lens = min(min_len, 4)
            if lens == 0:
                continue
            if lens >= 4:
                bleu_4_utt = sentence_bleu([target_utt], pred_utt, weights = (0.25, 0.25, 0.25, 0.25), smoothing_function = SmoothingFunction().method1)
                nist_4_utt = sentence_nist([target_utt], pred_utt, 4)
            else:
                bleu_4_utt = 0
                nist_4_utt = 0
            if lens >= 2:
                bleu_2_utt = sentence_bleu([target_utt], pred_utt, weights = (0.5, 0.5), smoothing_function = SmoothingFunction().method1)
                nist_2_utt = sentence_nist([target_utt], pred_utt, 2)
            else:
                bleu_2_utt = 0
                nist_2_utt = 0
            bleu_2 += bleu_2_utt
            bleu_4 += bleu_4_utt
            meteor += meteor_score([target_utt], pred_utt)
            nist_2 += nist_2_utt
            nist_4 += nist_4_utt
        res_list = [bleu_4, bleu_2, nist_4, nist_2, meteor]
        res = res_list[metrics_.index(metric_)]
        return torch.tensor(res)

    def cal_entropy(logits, metric_):
        metrics_ = ['Entropy-4', 'Dist-1', 'Dist-2']
        _, preds = logits.max(dim=-1)
        etp_score = [0.0, 0.0, 0.0, 0.0]
        div_score = [0.0, 0.0, 0.0, 0.0]
        counter = [defaultdict(int), defaultdict(int),
                defaultdict(int), defaultdict(int)]
        for index in range(len(preds)):
            generated_ = tokenizer.convert_ids_to_tokens(preds[index].tolist())
            generated = " ".join(generated_)
            g = generated.rstrip().split()
            for n in range(4):
                for idx in range(len(g)-n):
                    ngram = ' '.join(g[idx:idx+n+1])
                    counter[n][ngram] += 1
        for n in range(4):
            total = sum(counter[n].values()) + 1e-10
            for v in counter[n].values():
                etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
            div_score[n] = (len(counter[n].values())+0.0) / total
        res_list = [etp_score[3], div_score[0], div_score[1]]
        res = res_list[metrics_.index(metric_)]
        return torch.tensor(res)


    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "lr")

    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), 
                           output_transform=lambda x: (x[0].view(-1, x[0].size(-1)), x[1].view(-1)))}  # modified
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["Perplexity"] = MetricsLambda(math.exp, metrics["average_nll"])

    metrics["BLEU-2"] = Loss(cal_metrics, output_transform=lambda x: (x[0], x[1], {"metric_": "BLEU-2"}))
    metrics["BLEU-4"] = Loss(cal_metrics, output_transform=lambda x: (x[0], x[1], {"metric_": "BLEU-4"}))
    metrics["NIST-4"] = Loss(cal_metrics, output_transform=lambda x: (x[0], x[1], {"metric_": "NIST-4"}))
    metrics["NIST-2"] = Loss(cal_metrics, output_transform=lambda x: (x[0], x[1], {"metric_": "NIST-2"}))
    metrics["METEOR"] = Loss(cal_metrics, output_transform=lambda x: (x[0], x[1], {"metric_": "METEOR"}))

    metrics["Entropy-4"] = Loss(cal_entropy, output_transform=lambda x: (x[0], "Entropy-4"))
    metrics["Dist-1"] = Loss(cal_entropy, output_transform=lambda x: (x[0], "Dist-1"))
    metrics["Dist-2"] = Loss(cal_entropy, output_transform=lambda x: (x[0], "Dist-2"))

    for name, metric in metrics.items():
        metric.attach(evaluator, name)




    # On the main process: add progress bar, tensorboard, checkpoints
    # And save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True, mininterval=2)
        pbar.attach(trainer, metric_names=["loss", "lr"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: \n%s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=None)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys())),
                         event_name=Events.EPOCH_COMPLETED)  # OutputHandler原先有another_engine=trainer，现在取消了该参数another_engine

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=15)
        # save model after evaluation
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, tb_logger.writer.logdir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.logdir, CONFIG_NAME))
        tokenizer.save_vocabulary(tb_logger.writer.logdir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint
    # (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    # print(tb_logger.writer.logdir)
    # print(checkpoint_handler._saved)
    print(os.path.join(tb_logger.writer.logdir, checkpoint_handler._saved[-1][1][-1]))
    print(os.path.join(tb_logger.writer.logdir, WEIGHTS_NAME))

    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1],os.path.join(tb_logger.writer.logdir, WEIGHTS_NAME))  # _saved[-1][1][-1]

        tb_logger.close()


if __name__ == "__main__":
    train()
    # with open('./data/data.json','r',encoding='utf-8') as f:
    #     data = json.load(f)
    #     train_data = data['train']
    #     print(len(train_data))
    # # print(len(data))