# --------------------------------------------------------
# -----------数据处理工具在data_loader中被调用--------------
# --------------------------------------------------------

from core.data.ans_punct import prep_ans
import numpy as np
import en_core_web_sm
# import en_vectors_web_lg
import random, re, json


def shuffle_list(ans_list):
    random.shuffle(ans_list)


# ------------------------------
# ----初始化工具文件-------------
# ------------------------------
#图像特征目录加载
def img_feat_path_load(path_list):
    '''
    :param path_list: 图像特征文件目录列表:['./datasets/coco_extract/val2014/']
    :return: 返回{iid：图像名}类型 例如：{'9':'COCO_train2015_00000000009.jpg.npz'}
    '''
    # pathimg = path_list[0]
    # pathx = os.listdir(pathimg)
    iid_to_path = {}
    # for ix, path in enumerate(pathx):
    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0])) #把图像名字中的id取出来
        iid_to_path[iid] = path
    return iid_to_path

#图像特征加载
def img_feat_load(path_list):
    """
    :param path_list: 目录列表
    :return: {iid:feat}
    """
    iid_to_feat = {}
    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        img_feat = np.load(path)
        img_feat_x = img_feat['x'].transpose((1, 0))
        iid_to_feat[iid] = img_feat_x
        print('\rPre-Loading: [{} | {}] '.format(ix, path_list.__len__()), end='          ')

    return iid_to_feat

#问题加载
def ques_load(ques_list):
    '''
    :param ques_list: 输入问题列表：[{'image_id':'458752','question':'what...','question_id':'458752000'}...]
    :return: 返回字典类型---{'458752000':{'image_id':'','question':'','question_id':'458752000'}...}
    '''
    qid_to_ques = {}

    for ques in ques_list:
        qid = str(ques['question_id'])
        qid_to_ques[qid] = ques

    return qid_to_ques


def tokenize(stat_ques_list, use_glove):
    '''
    :param stat_ques_list: 输入问题列表：[{'image_id':'458752','question':'what...','question_id':'458752000'}...]
    :param use_glove: 是否使用glove
    :return: 返回{单词：索引},pretrain_emb训练前的单词嵌入(18405,96),词嵌入大小96
    '''
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        # spacy_tool = en_vectors_web_lg.load()
        spacy_tool = en_core_web_sm.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)

    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                #token_to_ix:把问题出现的词写入，如果重复出现不写，例如每个问题都有what，则只写一次
                #共18405个词
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb

def ans_stat(json_file):
    '''
    :param json_file: json文件
    :return: 输出{"答案"：索引}，{"ix":"ans"}
    '''
    ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))
    return ans_to_ix, ix_to_ans

#图像特征处理
def proc_img_feat(img_feat, img_feat_pad_size):
    '''
    :param img_feat:图像特征，
    :param img_feat_pad_size:填充大小100
    :return:返回图像特征
    '''
    if img_feat.shape[0] > img_feat_pad_size:
        img_feat = img_feat[:img_feat_pad_size]

    img_feat = np.pad(
        img_feat,
        ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return img_feat

#问题处理函数
def proc_ques(ques, token_to_ix, max_token):
    """
    :param ques: 输入的问题 {image_id:47391,question:what color is the boy,question_id:47391000}
    :param token_to_ix: {单词：索引}字典
    :param max_token: 最多单词个数
    :return: 返回{问题：问题长度}键值字典
    """
    ques_ix = np.zeros(max_token, np.int64)

    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques['question'].lower()
    ).replace('-', ' ').replace('/', ' ').split()

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


def get_score(occur):
    if occur == 0:
        return .0
    elif occur == 1:
        return .3
    elif occur == 2:
        return .6
    elif occur == 3:
        return .9
    else:
        return 1.

#答案处理函数
def proc_ans(ans, ans_to_ix):
    '''
    :param ans: 输入的答案
    :param ans_to_ix: {"ans":'ix'} 共3129个词，答案出现频率高的前3129个，
    :return: ans_core:答案的分数
    '''
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    ans_prob_dict = {} #答案单词字典

    for ans_ in ans['answers']:
        ans_proc = prep_ans(ans_['answer'])
        if ans_proc not in ans_prob_dict:
            ans_prob_dict[ans_proc] = 1
        else:
            ans_prob_dict[ans_proc] += 1

    for ans_ in ans_prob_dict:
        if ans_ in ans_to_ix:
            ans_score[ans_to_ix[ans_]] = get_score(ans_prob_dict[ans_])

    return ans_score

