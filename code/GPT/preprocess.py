import os
from sklearn.model_selection import train_test_split
import json
import jieba
MAX_HISTORY = 15
MAX_TOKENS = 470

DISEASE_REL = ['检查', "影像学检查", "辅助治疗", "筛查", "临床症状及体征", "相关症状", "药物治疗", "病因", "辅助检查",
               "手术治疗",
               "诱发因素", "侵及周围组织或转移的症状", "诊断依据", "发病部位", "外侵部位", "治疗方案", "所属科室",
               "预防"]
MEDICINE_REL = ["入药部位", "适应证", "适应症", "功能主治"]
CHECK_REL = ['适应症', "相关症状", "相关疾病", "检查科目", "发病部位"]
SYMPTOM_REL = ['检查', "相关疾病", "所属科室"]


def load_json2data(file_path):
    '''
    把数据从json中读取
    :param file_path:
    :return:
    '''
    with open(file_path,mode='r',encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_data2json(newfile_path,data):
    '''
    把数据保存为json文件
    :param newfile_path:
    :param data:
    :return:
    '''
    data = json.dumps(data,ensure_ascii=False)
    with open(newfile_path, 'w', encoding='utf-8') as f:
        f.write(data)


def read_wb_emotion_data():
    root_path = r'G:\MeetYouData\DataFilterTask\情感星座类_二次过滤后'
    wb_emotion_data = list()
    for path in os.listdir(root_path):

        for file in os.listdir(os.path.join(root_path, path)):
            dialog_path = os.path.join(root_path, path, file)
            data = load_json2data(dialog_path)
            for d in data:
                d['dialogs'] = d['dialogs'][:5]
            dialogs = get_wb_dialog(data)
            wb_emotion_data.extend(dialogs)

    return wb_emotion_data



def read_wb_health_data():
    '''
    读取微博数据，转成list格式
    '''
    root_path = r'G:\MeetYouData\DataFilterTask\妇科母婴类'
    wb_health_data = list()
    total = 0
    for fst_dir in os.listdir(root_path):
        # print(fst_dir)
        for sed_dir in os.listdir(os.path.join(root_path,fst_dir)):
            print(sed_dir)
            data = load_json2data(os.path.join(root_path,fst_dir,sed_dir))
            dialogs = get_wb_dialog(data)
            total = total + count_wb_dialog(data)
            wb_health_data.extend(dialogs)

    print(total)

    return wb_health_data

def count_wb_dialog(data):
    '''
    统计微博数据
    '''
    c = 0
    for d in data:
        dialogs = d['dialogs']
        for dialog in dialogs:
            valid = dialog['valid']
            if valid == '1':
                c += 1
    return c

def get_wb_dialog(data):
    '''
    获取微博部分的对话数据
    '''
    result = list()
    for d in data:
        dialogs = d['dialogs']

        category = d['category']
        for idx,dialog in enumerate(dialogs):
            valid = dialog['valid']
            if valid == '1':
                content = dialog['content']
                # 对话轮次
                turn = len(content)
                # 总词数，记得去空格
                tokens = sum([len(u.replace(" ","")) for u in content])
                # if (tokens+turn+2) <= MAX_TOKENS:
                #     sample = {}
                #     sample['category'] = category
                #     sample['dialog'] = content
                #     result.append(sample)
                # else:
                    # 1.直接分成两个对话
                    # result.append({"category":category,"dialog":content[:(turn//2)]})
                    # result.append({"category": category, "dialog": content[(turn//2):]})
                    # 2.删除句子
                while content and (turn+tokens+2) > MAX_TOKENS:
                    # 如果这个根评论下的对话集合只有一个，或者这是第一个对话
                    if idx == 0:
                        # 从末尾开始删除
                        content = content[:-1]
                    else:
                        # 否则，从头开始删除
                        content = content[1:]
                    # 对话轮次
                    turn = len(content)
                    # 总词数
                    tokens = sum([len(u.replace(" ","")) for u in content])
                result.append({"category": category, "dialog": content})

    return result
'''
读取一个根评论，查看是否有多个子对话
对于超过30轮和总长度大于480的对话，可以分割为两个对话

'''
def read_dy_dialog():
    ''''
    读取抖音数据，转成list格式
    '''
    # total = 0
    result = list()
    dy_path = r'G:\MeetYouData\抖音数据\抖音数据\Processed'
    for file in os.listdir(dy_path):
        data = load_json2data(os.path.join(dy_path,file))
        for d in data:
            dialog = d['dialog']
            category = d['category']
            if d['valid'] == '1':
                dialog = [u[3:] for u in dialog]
                turn = len(dialog)
                tokens = sum([len(u.replace(" ","")) for u in dialog])
                # if tokens+turn+2 <= MAX_TOKENS:
                #     result.append(dialog)
                while dialog and (turn+tokens+2) > MAX_TOKENS:
                    # 从末尾开始删除
                    dialog = dialog[:-1]
                    # 对话轮次
                    turn = len(dialog)
                    # 总词数
                    tokens = sum([len(u.replace(" ","")) for u in dialog])
                result.append({"category": category, "dialog": dialog})
        # print(len(data))
        # total += len(data)
    # print(total)
    return result

def get_data():

    wb_health_data = read_wb_health_data()
    print(len(wb_health_data))
    wb_emotion_data = read_wb_emotion_data()
    print(len(wb_emotion_data))
    dy_data = read_dy_dialog()
    print(len(dy_data))
    # 数据长度限制和截断

    health_data = wb_health_data + dy_data
    emotion_data = wb_emotion_data
    print("一共",len(health_data+emotion_data))
    return health_data,emotion_data

def split_data(health_data,emotion_data):
    # 划分数据集
    dataset = {}

    health_train_set, health_test_set = train_test_split(health_data, test_size=0.2, shuffle=True)
    health_valid_set, health_test_set = train_test_split(health_test_set, test_size=0.5, shuffle=True)

    emotion_train_set, emotion_test_set = train_test_split(emotion_data, test_size=0.2, shuffle=True)
    emotion_valid_set, emotion_test_set = train_test_split(emotion_test_set, test_size=0.5, shuffle=True)


    dataset['train'] = health_train_set + emotion_train_set
    dataset['valid'] = health_valid_set + emotion_valid_set
    dataset['test'] = health_test_set + emotion_test_set
    print("train size:",len(dataset['train']))
    print("valid size",len(dataset['valid']))
    print("test size",len(dataset['test']))
    return dataset



# 分词后的数据集
def cut_data():
    dataset = load_json2data('./data/dataset.json')
    jieba.load_userdict(r"./data/medical_vocab.txt")
    data = {}
    for key,value in dataset.items():
        #
        data[key] = []
        for item in value:
            dialog = item['dialog']
            dialog = [" ".join(jieba.cut(utterance)) for utterance in dialog]
            item['dialog'] = dialog
            data[key].append(item)
    save_data2json('./data/cut_dataset.json',data)



# 获取停用词表
def get_stop_words(stop_words_path='./data/hit_stopwords.txt'):
    stopwords = [line.strip() for line in open(stop_words_path, encoding='UTF-8').readlines()]
    return stopwords




# 获取知识
def get_knowledge(knowledge_path):
    '''
    disease
    medicine
    check_item
    symptom
    '''
    knowledge = load_json2data(knowledge_path)
    for type,data in knowledge.items():
        relation = set()
        for triplet in knowledge[type]:
            relation.add(triplet[1])
    return knowledge

def dialog_join_knowledge():
    # 加载停用词表
    stop_words = get_stop_words()
    # 加载知识
    knowledge = get_knowledge('./data/knowledge.json')

    from tqdm import tqdm
    # 加载分词后的数据集
    data = load_json2data('./data/cut_dataset.json')
    train_set = data['train']
    konwledge_join_dataset=[]
    for item in tqdm(train_set):
        dialog = item['dialog']
        # 按对话分割的知识实体
        dialog_knowledge = []
        for utterance in dialog:
            # 一条话语的知识实体
            utter_knowledge = set()

            utter_list = utterance.split()
            for word in utter_list:
                # 非停用词
                if word not in stop_words:
                    # 检索实体
                    for type in knowledge:

                        for head, relation, tail in knowledge[type]:
                            if word == head and relation in (DISEASE_REL+MEDICINE_REL+CHECK_REL+SYMPTOM_REL):
                                utter_knowledge.add(tail)
                            elif word == tail and relation in (DISEASE_REL+MEDICINE_REL+CHECK_REL+SYMPTOM_REL):
                                utter_knowledge.add(head)


            dialog_knowledge.append(list(utter_knowledge))

        konwledge_join_dataset.append(dialog_knowledge)
    save_data2json('data/train_knowledge_join.json', konwledge_join_dataset)

def generate_medical_vocab():
    '''
    根据CMeKG，生成实体词语词典
    '''
    knowledge = get_knowledge('./data/knowledge.json')
    entities = set()
    for type, data in knowledge.items():
        for triplet in knowledge[type]:
            head = triplet[0]
            relation = triplet[1]
            tail = triplet[2]
            if relation in DISEASE_REL or relation in MEDICINE_REL or relation in SYMPTOM_REL or relation in CHECK_REL:
                entities.add(head)
                entities.add(tail)
    print(len(entities))
    with open('./data/medical_vocab.txt','a',encoding='utf-8') as f:
        for entity in entities:
            f.write(entity)
            f.write('\n')

if __name__ == '__main__':
    # dialog_join_knowledge()
    count = {}
    dataset = load_json2data('./data/dataset.json')
    for key,value in dataset.items():
        data = dataset[key]
        for dialog in data:
            category = dialog['category']
            count[category] = count.get(category,0)+1
    print(count)

