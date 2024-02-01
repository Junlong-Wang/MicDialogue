from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist

from nltk.util import ngrams
import numpy as np

# 定义文本生成的指标
# 及，‘单个样本’的指标计算方法

def bleu(predict, target, n):
    chencherry = SmoothingFunction()
    return sentence_bleu([target], predict, weights=tuple(1 / n for i in range(n)),smoothing_function=chencherry.method1)


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
    sen_length = [len(s.split()) for s in sentences]
    return np.mean(sen_length), np.var(sen_length)


def calculate_metrics(predict, reference):
    # input: List[str],List[str]
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
    return bleu_1,bleu_2, bleu_3,bleu_4, nist_2, nist_4, meteor_scores
