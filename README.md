# Introduction

This repository contains the datasets and associated code.
In this work, we propose a novel Parallelized User Representation Framework (PURF), which decouples the learning of global and local user representations by introducing a dual-channel architecture. Specifically, the global channel employs a flexible and pluggable graph encoder that captures user interaction patterns across different cascades, while the local channel utilizes a lightweight Transformer to model the cascade-specific user dynamics within each diffusion instance. These two types of user representations are then integrated through a dynamic gating mechanism, which adaptively balances their contributions based on the context. This design not only addresses the representational conflict inherent in sequential structures but also provides a modular and extensible solution that can be combined with a wide range of existing models. Our framework is designed to be simple yet effective, requiring no complex architectural modifications or additional computational overhead. 

Extensive experiments conducted on three real-world datasets—Twitter, Weibo, and Douban—demonstrate that PURF consistently enhances the performance of four state of art baselines, achieving improvement on both MAP and Hits metrics. Further studies confirm that dual-channel design mitigates performance degradation in the event of single-channel failure, and dynamically adjusts feature importance based on dataset characteristics and user behavior patterns.

# Dataset

The Dataset contains the Twitter and the Weibo that we constructed, and the public dataset Douban. Each dataset contains two types of formats that were used in the experiments to test the baseline and PURF framework, respectively. The basic information of the dataset is as follows.

|             | Twitter | Weibo | Douban  |
| ----------- | ------- | ----- | ------  |
| #user       | 8,292   | 6,512 | 11,899  |
| #cascades   | 2,798   | 2,267 | 3,485   |
| #Links      | 86,768  | -     | 194,685 |
| Avg. Length | 3.39    | 43.73 | 20.79   |

In each dataset folder, the origin directory contains all the original cascade information. The BaselineFormat directory includes the training, validation, and test data formatted for use with baseline models, while the ModelFormat directory contains the corresponding data formatted for use with the PURF framework. The data_process.py file provides the code for converting the data in the origin folder into these two formats.

# CODE
The PURF framework is implemented based on the overall Rechorus architecture and currently includes the code for using four baseline models as the global encoder.

## Getting Start

```python
python main.py --model_name DyHGCN --emb_size 64 --num_layers 1 --num_heads 1 --dropout 0.2 --lr 1e-3 --l2 0 --history_max 500 --dataset ModelFormat --path ./Dataset/Twitter --batch_size 128 --eval_batch_size 128 --test_all 1 --epoch 200 --metric 'HR,NDCG,MAP' --topk 5,10,20,50,100 --gpu 0 --random_seed 42
```

## Folder Structure

src/
├── main.py                # 主程序入口，负责参数解析、模型训练与评估等流程
├── models/                # 各类模型及其组件
│   ├── BaseModel.py           # 模型基类
│   ├── GraphEncoder.py        # 图编码器（全局通道核心）
│   ├── TransformerBlock.py    # Transformer模块相关代码
│   ├── DyHGCN.py              # PURF框架+DyHGCN模型
│   ├── DisenIDP.py            # PURF框架+DisenIDP模型
│   ├── MINDS.py               # PURF框架+MINDS模型
│   ├── MSHGAT.py              # PURF框架+MS-HGAT模型
│   └── __init__.py
├── utils/                 # 工具函数与通用组件
│   ├── load.py                # 数据加载与预处理
│   ├── utils.py               # 常用工具函数
│   ├── layers.py              # 神经网络层实现
│   └── __init__.py
├── helper/                # 训练与数据读取辅助模块
│   ├── BaseRunner.py          # 训练与评估流程控制
│   ├── BaseReader.py          # 数据读取基类
│   ├── DyHGCNReader.py        # DyHGCN数据读取器
│   ├── DisenIDPReader.py      # DisenIDP数据读取器
│   ├── MINDSReader.py         # MINDS数据读取器
│   ├── MSHGATReader.py        # MSHGAT数据读取器
│   └── __init__.py

## Citation
```bibtex
@inproceedings{yuan_dyhgcn_2021,
	title = {{DyHGCN}: {A} {Dynamic} {Heterogeneous} {Graph} {Convolutional} {Network} to {Learn} {Users}’ {Dynamic} {Preferences} for {Information} {Diffusion} {Prediction}},
	booktitle = {Machine {Learning} and {Knowledge} {Discovery} in {Databases}},
	author = {Yuan, Chunyuan and Li, Jiacheng and Zhou, Wei and Lu, Yijun and Zhang, Xiaodan and Hu, Songlin},
	year = {2021},
	pages = {347--363}
}

@article{sun_ms-hgat_2022,
	title = {{MS}-{HGAT}: {Memory}-{Enhanced} {Sequential} {Hypergraph} {Attention} {Network} for {Information} {Diffusion} {Prediction}},
	journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
	author = {Sun, Ling and Rao, Yuan and Zhang, Xiangbo and Lan, Yuqian and Yu, Shuanghe},
	year = {2022},
	pages = {4156--4164}
}

@article{jiao_enhancing_2024,
	title = {Enhancing {Multi}-{Scale} {Diffusion} {Prediction} via {Sequential} {Hypergraphs} and {Adversarial} {Learning}},
	journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
	author = {Jiao, Pengfei and Chen, Hongqian and Bao, Qing and Zhang, Wang and Wu, Huaming},
	year = {2024},
	pages = {8571--8581}
}

@inproceedings{cheng2023enhancing,
  title={Enhancing Information Diffusion Prediction with Self-Supervised Disentangled User and Cascade Representations},
  author={Cheng, Zhangtao and Ye, Wenxue and Liu, Leyuan and Tai, Wenxin and Zhou, Fan},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={3808--3812},
  year={2023}
}

@inproceedings{li2024rechorus2,
  title={ReChorus2. 0: A Modular and Task-Flexible Recommendation Library},
  author={Li, Jiayu and Li, Hanyu and He, Zhiyu and Ma, Weizhi and Sun, Peijie and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the 18th ACM Conference on Recommender Systems},
  pages={454--464},
  year={2024}
}

@inproceedings{wang2020make,
  title={Make it a chorus: knowledge-and time-aware item modeling for sequential recommendation},
  author={Wang, Chenyang and Zhang, Min and Ma, Weizhi and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={109--118},
  year={2020}
}

@article{王晨阳2021rechorus,
  title={ReChorus: 一个综合, 高效, 易扩展的轻量级推荐算法框架},
  author={王晨阳 and 任一 and 马为之 and 张敏 and 刘奕群 and 马少平},
  journal={软件学报},
  volume={33},
  number={4},
  pages={0--0},
  year={2021}
}
```