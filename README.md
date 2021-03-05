# PyTorch-ShuffleNet V2

**测试环境：Ubuntu-20.04，1050Ti**  

**基于PyTorch的ShuffleNetV2模型实现**    

**论文地址**:[https://arxiv.org/abs/1807.11164](https://arxiv.org/abs/1807.11164)  

**写了一篇关于ShuffleNetV2-Block的文章，欢迎查阅[https://zhuanlan.zhihu.com/p/354530881](https://zhuanlan.zhihu.com/p/354530881)**  

## 项目目录  

```
.  
├── class_indices.json  
├── code-test.py  
├── data_preparation.py  
├── model_data.pth  
├── model.py  
├── model_weight/  
├── my_dataset.py  
├── README.md  
├── test_net.py  
└── train_net.py  
```

## 文件说明  

**model**.py:ShuffleNetV2模型  

**train_net**.py:训练脚本  

**test_net**.py:测试脚本  

**code_test**.py:代码片段测试脚本（无关紧要）  

**class_indices**.json:自动生成的目标索引文件  

**data_perparation**.py:用于分割训练集和测试集  

**my_dataset**.py:继承DataSet类，用于实现DataLoader  

**.pth**:权重文件  

