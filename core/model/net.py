# --------------------------------------------------------
# ------------------------神经网络-------------------------
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch
# AttFlat  ： 注意力扁平化网络
class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C
        # 感知器网络
        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        # 线性网络
        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )
    # 前向传播，模型计算
    def forward(self, x, x_mask):
        '''
        :param x: 输入 需要加注意力的特征向量x
        :param x_mask: 输入 需要遮挡的张量 x_mask
        :return: 返回带有注意力的特征x_atted
        '''
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- 主要的MCAN模块-------
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        '''
        :param __C: 配置信息
        :param pretrained_emb: 语训练的词嵌入向量
        :param token_size: 18405个词
        :param answer_size:
        '''
        super(Net, self).__init__()
        #调用 Embedding进行词向量的嵌入
        self.embedding = nn.Embedding(
            num_embeddings=token_size, # 18405
            embedding_dim=__C.WORD_EMBED_SIZE # 96
        )

        # 加载Glove嵌入权重
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        # 采用LSTM网络，进行特征提取
        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE, # 96
            hidden_size=__C.HIDDEN_SIZE, # 256
            num_layers=1, # 层数：1
            batch_first=True
        )
        # 图像特征线性变换
        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE, #2048
            __C.HIDDEN_SIZE #256
        )
        # 主干网络
        self.backbone = MCA_ED(__C)
        #
        self.attflat_img = AttFlat(__C) # 求图像特征的自注意力
        self.attflat_lang = AttFlat(__C)# 求文本特征的注意力

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE) # 输出层的标准化 ？？ 为啥来这层
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size) # 预测答案的线性变换层 ？？为啥来这层


    # 前向传播，进行模型的计算
    def forward(self, img_feat, ques_ix):
        '''
        :param img_feat:  输入图像特征 （10,1024）
        :param ques_ix:  输入问题特征 （10,14）
        :return: 融合特征
        '''

        # 设置遮挡，不给模型看未来的信息
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2)) #（10,1,1,14）
        img_feat_mask = self.make_mask(img_feat) #（10,1,1,100）

        # 预处理语言特征
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # 预处理图像特征
        img_feat = self.img_feat_linear(img_feat) #[5,100,2048]

        # 主干网络框架 调用MCA_ED网络模型，
        # 输入：语言特征，图像特征，带有遮挡的语言图像矩阵，带有遮挡的图像特征矩阵
        # 输出：由MCA_ED处理后的语言特征，图像特征，
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )
        # 加入注意力网络 输出带有注意力的语言特征
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )
        # 加入注意力网络 输出带有注意力的图像特征
        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )
        # 将两者特征融合
        proj_feat = lang_feat + img_feat
        # 通过标准化网络进行标准化
        proj_feat = self.proj_norm(proj_feat)
        # 通过sigmoid网络进行分类
        proj_feat = torch.sigmoid(self.proj(proj_feat))
        # 返回融合后的特征
        return proj_feat


    #遮挡对角线上一部分信息，不给模型看未来的数据
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature),dim=-1) == 0).unsqueeze(1).unsqueeze(2)
