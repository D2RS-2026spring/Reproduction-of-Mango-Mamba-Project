# Reproduction-of-Mango-Mamba-Project
Mango-Mamba芒果叶病害轻量级检测模型结果复现仓库
# 复现步骤



## 一、创建Colaboratory（需使用GPU，普通PC无法直接复现）

1.在 Google 云端硬盘(https://drive.google.com/drive/home)新建一个Colaboratory；

2.点击修改-笔记本设置，运行时类型选择python 3，硬件加速器选择T4 GPU。



## 二、python环境配置

1.新建Colab单元格，执行

```
!pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

如有报错，则重复执行，直至无报错；



2.新建Colab单元格，执行

```
!pip install einops timm

!pip install "mamba-ssm<2.3" "transformers<4.36" --no-build-isolation -v
```

如有报错，则重复执行，直至无报错。



## 三、仓库克隆

新建Colab单元格，执行

```
!git clone https://github.com/hzau-wujing/Reproduction-of-Mango-Mamba-Project
```



## 四、训练代码复现

新建Colab单元格内，执行

```
!python /content/Reproduction-of-Mango-Mamba-Project/train.py
```



## 五、混淆矩阵与热力图代码复现

新建Colab单元格，执行

```
!python /content/Reproduction-of-Mango-Mamba-Project/pic.py
```

