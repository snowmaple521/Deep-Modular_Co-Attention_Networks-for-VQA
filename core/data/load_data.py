# --------------------------------------------------------#
#--------------------数据加载------------------------------#
# -------------------注释：SnowMaple------------------- --#
# --------------------------------------------------------#


from core.data.data_utils import img_feat_path_load, img_feat_load, ques_load, tokenize, ans_stat
from core.data.data_utils import proc_img_feat, proc_ques, proc_ans
import numpy as np
import glob, json, torch, time
import torch.utils.data as Data

class DataSet(Data.Dataset):
    def __init__(self, __C):
        """
        :param __C: 配置信息
        """
        self.__C = __C
        #加载原始数据
        self.img_feat_path_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        #split_list ={'train','val','vg'}
        for split in split_list:
            if split in ['train']:
                self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')

        self.stat_ques_list = json.load(open(__C.QUESTION_PATH['train'], 'r'))['questions']
            # json.load(open(__C.QUESTION_PATH['train'], 'r'))['questions'] + \
            #json.load(open(__C.QUESTION_PATH['val'], 'r'))['questions']
            # json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
            # json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions']

        self.ques_list = []
        self.ans_list = [] #annotations文件内容读取到ans_list列表中

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:

            self.ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))['questions']
            if __C.RUN_MODE in ['train']:
                self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']

        # 定义运行数据大小 = ans_list_len
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print('== Dataset size:', self.data_size)

        # {image id} -> {image feature absolutely path}
        if self.__C.PRELOAD: #PRELOAD=False 执行else
            print('==== Pre-Loading features ...')
            time_start = time.time()
            self.iid_to_img_feat = img_feat_load(self.img_feat_path_list)
            time_end = time.time()
            print('==== Finished in {}s'.format(int(time_end-time_start)))
        else:
#调用data_utils.py中img_feat_path_load函数加载图像特征文件
            #图像特征{'9':'COCO_train2015_00000000009.jpg.npz',....}
            self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)
#调用data_utils ques_load函数加载问题
        # 问题加载：{'458752000':{'image_id':'','question':'','question_id':'458752000'}...}
        self.qid_to_ques = ques_load(self.ques_list)

#调用data_utils.py的tokenize的函数
        #token_to_ix:把问题出现的词写入，如果重复出现不写，例如每个问题都有what，则只写一次
        #pretrained_emb:词嵌入
        #token_size : 大小18405
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)
#调用 data_utils.py的ans_stat函数
        self.ans_to_ix, self.ix_to_ans = ans_stat('core/data/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')


    def __getitem__(self, idx):
        '''
        self:ans_list,ans_toix,..
        :param idx: idx=0
        :return:torch类型的：img_feat_iter，ques_feat_iter,ans_iter
        '''

        # For code safety
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train']:
            #加载答案数据，每次加载一个annotation数据包含answers:10个答案,image_id,question_id，
            # {'answers':[{'answer':'skatebodarding'},...,{'answer':'skatebodarding'}],"image_id":139831,"question_id"='VG_1293929'
            ans = self.ans_list[idx]
            #加载问题数据，每次加载一个question如下：{'image_id': 139831, 'question': "What's the man doing?", 'question_id': 'VG_1293929'}
            ques = self.qid_to_ques[str(ans['question_id'])]

            # Process image feature from (.npz) file
            if self.__C.PRELOAD: #如果为真，返回image_id的npz文件
                img_feat_x = self.iid_to_img_feat[str(ans['image_id'])]
            else:
                #之间load image_id的npz文件
                img_feat = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
                #将数据转换维度
                img_feat_x = img_feat['x'].transpose((1, 0))
            #图像特征迭代器，图像特征输入x，特征填充大小：100
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            #问题特征迭代器，调用data_utils的proc_ques函数，输入ques,token_to_ix,max_token
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            #答案迭代器，调用data_utils的proc_ans函数,传入ans，ans_to_ix数据，输出答案分数矩阵
            ans_iter = proc_ans(ans, self.ans_to_ix)

        else:
            # Load the run data from list
            ques = self.ques_list[idx]
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

        return torch.from_numpy(img_feat_iter), torch.from_numpy(ques_ix_iter), torch.from_numpy(ans_iter)

    #统计数据长度
    def __len__(self):
        return self.data_size


