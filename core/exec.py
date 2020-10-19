# --------------------------------------------------------
# -----------------执行文件--------------------------------
# -----------------Snow Maple 注释-------------------------
# --------------------------------------------------------

from core.data.load_data import DataSet
from core.model.net import Net
from core.model.optim import get_optim, adjust_lr
from core.data.data_utils import shuffle_list
from core.data.vqa import VQA
from core.data.vqaEval import VQAEval

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
#代码注释部分，1.*的是对训练模型进行的注释，2.*是对评估模型进行的注释

class Execution:
    def __init__(self, __C):
        self.__C = __C
        print('Loading training set ........')
        #1.1加载训练数据集
        self.dataset = DataSet(__C) #1. 执行完load_data，返回此处
        #2.1 评估数据集设为None
        self.dataset_eval = None
        #EVAL_EVERY_EPOCH:设置为true进行脱机评估
        if __C.EVAL_EVERY_EPOCH:
            __C_eval = copy.deepcopy(__C)
            #设置RUN_MODE = train
            setattr(__C_eval, 'RUN_MODE', 'train')
            #为每个epoch评估加载验证集,因为内存有限，此处我将val改成了train
            print('Loading validation set for per-epoch evaluation ........')
            #2.2 加载评估数据集（验证集，在评估方法上调用验证集数据集）
            self.dataset_eval = DataSet(__C_eval)
    #1.2开始训练模型，此处评估数据集为None,只训练不作评估
    def train(self, dataset, dataset_eval=None):

        #1.3 首先训练的开始前要获取需要的信息，数据集大小，数据集的问题嵌入大小，答案大小，文本嵌入向量
        data_size = dataset.data_size
        token_size = dataset.token_size #18405
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        #1.4 需要的信息获取后，开始定义MCAN模型，传入需要的参数，输出：多模态融合特征 proj_feat
        net = Net(self.__C,pretrained_emb, token_size,ans_size )
        net.cuda()
        net.train() # 1.5 调用train进行训练，这步的前一步是？后一步是？

        # 如果需要的话，定义多gpu训练
        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()

        # 如果恢复训练，则加载检查点
        if self.__C.RESUME:
            print(' ========== 恢复性训练 ==========')

            if self.__C.CKPT_PATH is not None:
                print('警告:您现在正在使用CKPT_PATH参数，CKPT_VERSION和CKPT_EPOCH不能工作')

                path = self.__C.CKPT_PATH #此处要设置ckpt_path的目录，不能为None
            else:
                path = self.__C.CKPTS_PATH + \
                       'ckpt_' + self.__C.CKPT_VERSION + \
                       '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

            # 加载模型网络参数
            print('加载ckpt {} 文件'.format(path))
            ckpt = torch.load(path)
            print('参数加载完成!')
            #...............state_dict这里存什么数据.............
            net.load_state_dict(ckpt['state_dict'])

            # 加载优化器参数
            optim = get_optim(self.__C, net, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.__C.BATCH_SIZE * self.__C.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])
            # epoch
            start_epoch = self.__C.CKPT_EPOCH

        # 如果不恢复训练，则重新训练
        else:
            if ('ckpt_' + self.__C.VERSION) in os.listdir(self.__C.CKPTS_PATH):
                shutil.rmtree(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            optim = get_optim(self.__C, net, data_size)
            start_epoch = 0

        loss_sum = 0
        named_params = list(net.named_parameters()) # 参数名称
        grad_norm = np.zeros(len(named_params)) # 梯度规范化

        # 定义多线程数据加载 dataloader
        if self.__C.SHUFFLE_MODE in ['external']:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=False,
                num_workers=self.__C.NUM_WORKERS, # 进程数
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )
        else:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=True,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )

        # 训练过程 这里max_epoch我设置为1
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):

            # 保存日志信息
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            # 写入日志信息
            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            # 学习率衰减
            if epoch in self.__C.LR_DECAY_LIST:
                adjust_lr(optim, self.__C.LR_DECAY_R)

            # Externally shuffle
            if self.__C.SHUFFLE_MODE == 'external':
                shuffle_list(dataset.ans_list)

            time_start = time.time()

            # 迭代的加载 图像特征迭代器，问题特征迭代器，答案迭代器
            for step, ( img_feat_iter,  ques_ix_iter, ans_iter ) in enumerate(dataloader):
                optim.zero_grad() # 梯度清零
                img_feat_iter = img_feat_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                ans_iter = ans_iter.cuda()
                # grad_accu_steps:累计梯度，来解决本地显存不足的问题，
                # 其是变相扩大batchsize，如果batch_size=6,样本总量为24，grad_acc_steps=2
                # 那么参数更新次数为24/6=4，如果减小batch_size = 6/2=3，则参数更新次数不变
                for accu_step in range(self.__C.GRAD_ACCU_STEPS):

                    sub_img_feat_iter = \
                        img_feat_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                     (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ans_iter = \
                        ans_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * self.__C.SUB_BATCH_SIZE]


                    pred = net(
                        sub_img_feat_iter,#[5,100,2048]
                        sub_ques_ix_iter # [5,14]
                    )

                    loss = loss_fn(pred, sub_ans_iter)
                    # 只有平均减少需要被grad_accu_steps划分
                    loss.backward() # 反向传播，计算当前梯度
                    loss_sum += loss.cpu().data.numpy() * self.__C.GRAD_ACCU_STEPS
                    # 输出每个train的loss
                    if self.__C.VERBOSE:
                        if dataset_eval is not None:
                            mode_str = self.__C.SPLIT['train']
                        else:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['train']

                        print("\r[version %s][epoch %2d][step %4d/%4d][%s] loss: %.4f, lr: %.2e" % (
                            self.__C.VERSION,
                            epoch + 1,
                            step,
                            int(data_size / self.__C.BATCH_SIZE),
                            mode_str,
                            loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                            optim._rate
                        ), end='          ')

                # Gradient norm clipping 梯度标准剪裁
                if self.__C.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.__C.GRAD_NORM_CLIP
                    )

                # 保存梯度下降信息
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.__C.GRAD_ACCU_STEPS
                optim.step()

                # with open('One_epoch_data.txt','w') as F:
                #     F.write(net.state_dict()+optim.optimizer.state_dict()+optim.lr_base)

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            # print('')
            epoch_finish = epoch + 1

            # 保存检查点
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base
            }

            print("===========训练模型的state=====")
            print(state)
            torch.save(
                state,
                self.__C.CKPTS_PATH +
                'ckpt_' + self.__C.VERSION +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

            # 打开日志文件
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / data_size) +
                '\n' +
                'lr = ' + str(optim._rate) +
                '\n\n'
            )
            logfile.close()

            # 每个epoch后，进行模型评估，调用评估函数
            if dataset_eval is not None:
                self.eval(
                    dataset_eval,
                    state_dict=net.state_dict(),
                    valid=True
                )

            loss_sum = 0
            grad_norm = np.zeros(len(named_params))


    #评估函数
    def eval(self, dataset, state_dict=None, valid=False):
        # 评估模型，传入的数据集是验证集，主要是利用epoch1的检查点.pkl文件进行评估验证集
        # 的准确度，从而获取每种答案类型的精确度，

        # 加载模型参数
        if self.__C.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = self.__C.CKPT_PATH
        else:
            path = self.__C.CKPTS_PATH + \
                   'ckpt_' + self.__C.CKPT_VERSION + \
                   '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('加载 ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print("评估模型网络的state_dict:", state_dict)
            print('完成!')

        # 存储预测列表 问题id列表，答案列表
        qid_list = [ques['question_id'] for ques in dataset.ques_list]
        ans_ix_list = []
        pred_list = []

        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb
        # 和train一样调用网络
        net = Net(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        # 评估
        net.eval()

        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)
        net.load_state_dict(state_dict)

        # 数据集加载，此处传入的是验证集，
        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=True
        )

        for step, ( img_feat_iter, ques_ix_iter,ans_iter ) in enumerate(dataloader):
            print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / self.__C.EVAL_BATCH_SIZE),
            ), end='          ')

            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()
            # 输出预测的特征向量
            pred = net(
                img_feat_iter,
                ques_ix_iter
            )
            # 转换成cpu的数据的numpy类型
            pred_np = pred.cpu().data.numpy()
            # 沿着一个轴返回最大值:取出一个向量中预测值最大的，最有可能接近真实答案
            pred_argmax = np.argmax(pred_np, axis=1)
            np.savetxt("pre_np.txt", pred_np)
            np.savetxt("pre_argmax.txt", pred_argmax)
            # 保存最接近真实答案的索引
            if pred_argmax.shape[0] != self.__C.EVAL_BATCH_SIZE:
                pred_argmax = np.pad(
                    pred_argmax,
                    (0, self.__C.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                    mode='constant',
                    constant_values=-1
                )

            # 将这个最接近真实答案的预测答案pre_argmax加入到ans_ix_list中，
            ans_ix_list.append(pred_argmax)

            file = open('ans_ix_list.txt', 'w')
            file.write(str(ans_ix_list))
            file.close()
            # 保存整个预测向量
            if self.__C.TEST_SAVE_PRED:
                if pred_np.shape[0] != self.__C.EVAL_BATCH_SIZE:
                    pred_np = np.pad(
                        pred_np,
                        ((0, self.__C.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                        mode='constant',
                        constant_values=-1
                    )
                pred_list.append(pred_np)

        print('')
        ans_ix_list = np.array(ans_ix_list).reshape(-1)

        result = [{
            'answer': dataset.ix_to_ans[str(ans_ix_list[qix])],  # ix_to_ans(load with json) keys are type of string
            'question_id': int(qid_list[qix])
        }for qix in range(qid_list.__len__())]

        # 将结果写入结果文件：问题id：对应预测答案
        if valid:
            if val_ckpt_flag:
                result_eval_file = \
                    self.__C.CACHE_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__C.CACHE_PATH + \
                    'result_run_' + self.__C.VERSION + \
                    '.json'

        else:
            if self.__C.CKPT_PATH is not None:
                result_eval_file = \
                    self.__C.RESULT_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                result_eval_file = \
                    self.__C.RESULT_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '_epoch' + str(self.__C.CKPT_EPOCH) + \
                    '.json'

            print('Save the result to file: {}'.format(result_eval_file))

        json.dump(result, open(result_eval_file, 'w'))

        # 保存整个预测向量
        if self.__C.TEST_SAVE_PRED:

            if self.__C.CKPT_PATH is not None:
                ensemble_file = \
                    self.__C.PRED_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '.json'
            else:
                ensemble_file = \
                    self.__C.PRED_PATH + \
                    'result_run_' + self.__C.CKPT_VERSION + \
                    '_epoch' + str(self.__C.CKPT_EPOCH) + \
                    '.json'

            print('Save the prediction vector to file: {}'.format(ensemble_file))

            pred_list = np.array(pred_list).reshape(-1, ans_size)
            file = open('pred_list.txt', 'w')
            file.write(str(pred_list))
            file.close()
            result_pred = [{
                'pred': pred_list[qix],
                'question_id': int(qid_list[qix])
            }for qix in range(qid_list.__len__())]

            pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)


        # 运行验证脚本
        if valid:
            # 创建vqa对象和vqaRes对象
            ques_file_path = self.__C.QUESTION_PATH['train']
            ans_file_path = self.__C.ANSWER_PATH['train']

            vqa = VQA(ans_file_path, ques_file_path)
            vqaRes = vqa.loadRes(result_eval_file, ques_file_path)

            # 通过获取vqa和vqaRes创建vqaEval对象
            vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

            # 评估结果
            """
            如果您有一个问题id列表，希望对其进行结果评估，请将其作为列表传递给下面的函数
            默认情况下，它使用注释文件中的所有问题id
            """
            vqaEval.evaluate()

            # print accuracies
            print("\n")
            #计算全部准确率
            print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            # 计算每种答案类型准确率，yes/no,number,other
            print("Per Answer Type Accuracy is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print("\n")
            # 将评估结果写入log文件
            if val_ckpt_flag:
                print('Write to log file: {}'.format(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.CKPT_VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.CKPT_VERSION + '.txt',
                    'a+'
                )

            else:
                print('写入日志文件: {}'.format(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.VERSION + '.txt',
                    'a+')
                )

                logfile = open(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.VERSION + '.txt',
                    'a+'
                )

            logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            for ansType in vqaEval.accuracy['perAnswerType']:
                logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            logfile.write("\n\n")
            logfile.close()

    #运行函数
    def run(self, run_mode):
        #如果是训练集就执行训练函数
        if run_mode == 'train':
            self.empty_log(self.__C.VERSION)
            self.train(self.dataset, self.dataset_eval)
        #如果是验证集就执行评估函数
        elif run_mode == 'val':
            self.eval(self.dataset, valid=True)
        #如果是测试集就执行测试函数
        elif run_mode == 'test':
            self.eval(self.dataset)

        else:
            exit(-1)

    #初始化空log数据文件
    def empty_log(self, version):
        print('初始化日志文件........')
        if (os.path.exists(self.__C.LOG_PATH + 'log_run_' + version + '.txt')):
            os.remove(self.__C.LOG_PATH + 'log_run_' + version + '.txt')
        print('初始化日志文件完成!')
        print('')




