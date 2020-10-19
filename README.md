# Deep-Modular_Co-Attention_Networks-for-VQA
> 本项目主要是参考原作者代码然后进行了注释，放在自己代码库，以备以后的调用
> 本项目基于《Deep-Modular Co-Attention Networks for VQA》这篇论文
## 文件目录结构
```angular2html
-|-cfgs
|---base_cfgs.py 用于设置实验的一些配置参数 
|---path_cfgs.py 加载数据的一些目录参数
|==================================================|
|-core
|---data
|-----ans_punct.py 答案数据预处理文件
|-----answer_dict.json 答案json文件
|-----data_utils.py ：加载数据集需要的工具文件
|-----load_data.py ：加载数据集的函数
|-----vqa.py : vqa数据类
|-----vqaEval.py : 用于数据评估时类
|---model
|------mca.py 包含mac-ed级联encoder-decoder模型 包括MHAtt多头注意力类，FFN前馈神经网络，SA自我注意力，SGA自我指导注意力
|------net.py 网络模型
|------net_utils.py 网络模型的工具包括，标准化，感知器，全连接层
|------optim.py 优化
|---exec.py 模型执行文件，实现数据加载，模型训练，模型评估
|==================================================|

|-datasets
|---coco_extract ： fast-R-CNN提取好的特征，npy文件
|---vqa : 一些annotation，question文件
|-run.py : 项目入口
|==================================================|
```
