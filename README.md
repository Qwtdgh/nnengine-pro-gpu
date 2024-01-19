# nnengine-pro-gpu

BUAA 2023 硕士高等软工课程设计 GPU版

## 项目结构

```
.
├── src
│   ├── core                     张量、模型等核心接口
│   │   ├── nn.py
│   │   └── tensor.py
│   ├── data                     数据
│   │   └── dataloader.py
│   └── utils                    训练、测试工具
│       ├── evaluator.py         评估器
│       ├── optimizer.py         优化器
│       ├── scheduler.py         学习率管理器
│       └── trainer.py           训练器
├── test.py                      测试
└── usr                          自订模型、数据集
    └── mnist_dataset.py
```



## 项目运行

在A100上进行transformer机器翻译任务训练，大约占用10G GPU内存

```shell
python wmt18_translation.py
```

推理

```shell
python trans_pred.py
```

## Transformer机器翻译任务

![image-20240118094034892](https://typoraqlh.oss-cn-beijing.aliyuncs.com/qlh/typora/image-20240118094034892.png)


## gitlab协作

### 1. 暂存本地修改

```shell
git stash
```

### 2. 拉取远程代码

将origin main分支的代码到本地的origin/main分支

```shell
git fetch origin main
```

### 3. 合并分支

将origin/main分支的代码合并到本地的main分支

```shell
git rebase origin/main
```

### 4. 将本地修改出栈

```shell
git stash pop
```

