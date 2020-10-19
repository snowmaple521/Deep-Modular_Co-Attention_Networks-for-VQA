#--------------------------------
#MCAN-vqa:Deep Modular Co-Attention Networks
#Copy by SnowMaple
#用于文件位置配置文件
#--------------------------------
import os
class PATH:
    def __init__(self):
        # vqa2 dataset root path
        self.DATASET_PATH = './datasets/vqa/'
        #图像特征位置
        self.FEATURE_PATH = './datasets/coco_extract/'
        self.init_path()
    #图像数据全改为一个val2014
    def init_path(self):
        #图像特征目录
        self.IMG_FEAT_PATH = {
            'train':self.FEATURE_PATH +'train2014/',
            # 'val':self.FEATURE_PATH + 'val2014/'
        }
        #VQA2+VG问题目录
        self.QUESTION_PATH={
            'train':self.DATASET_PATH + 'v2_OpenEnded_mscoco_val2014_questions.json',
            # 'val':self.DATASET_PATH + 'v2_OpenEnded_mscoco_val2014_questions.json',
            # 'vg':self.DATASET_PATH+'VG_questions.json'
        }
        #VQA2+VG答案目录
        self.ANSWER_PATH={
            'train':self.DATASET_PATH+'v2_mscoco_val2014_annotations.json',
            # 'val':self.DATASET_PATH+'v2_mscoco_val2014_annotations.json',
            # 'vg':self.DATASET_PATH+'VG_annotations.json'
        }

        self.RESULT_PATH = './results/result_test/'
        self.PRED_PATH = './results/pred/'
        self.CACHE_PATH = './results/cache/'
        self.LOG_PATH = './results/log/'
        self.CKPTS_PATH = './ckpts/'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')
        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')
        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')
        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')
        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')


    def check_path(self):
        print("Checking dataset....")
        for mode in self.IMG_FEAT_PATH:
            if not os.path.exists(self.IMG_FEAT_PATH[mode]):
                print(self.IMG_FEAT_PATH[mode] +'NOT EXIT')
                exit(-1)
        for mode in self.QUESTION_PATH:
            if not os.path.exists(self.QUESTION_PATH[mode]):
                print(self.QUESTION_PATH[mode]+'NOT EXIT')
                exit(-1)
        for mode in self.ANSWER_PATH:
            if not os.path.exists(self.ANSWER_PATH[mode]):
                print(self.ANSWER_PATH[mode]+'NOT EXIT')
                exit(-1)

        print("数据目录检查完毕")







