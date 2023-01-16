# paper_translate

## Installation
* 创建环境
```
conda create --name layoutlmv3 python=3.7
conda activate layoutlmv3
```

* 首先完成LayoutLMv3的安装
```
git clone https://github.com/microsoft/unilm.git
cd unilm/layoutlmv3
pip install -r requirements.txt
# install pytorch, torchvision refer to https://pytorch.org/get-started/locally/
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# install detectron2 refer to https://detectron2.readthedocs.io/en/latest/tutorials/install.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -e .
```

* 然后完成vila的安装
```
# 回到初始目录
git clone git@github.com:allenai/VILA.git
cd VILA 
pip install -e . # Install the `vila` library 
pip install -r requirements.txt # Only install the dependencies 
```

* 完成本项目代码的拷贝与依赖库安装
```
# 回到初始目录
git clone https://github.com/lazyc81/paper_translate.git
cd paper_translate
pip install -r requirements.txt
```

## 权重文件下载

* 该项目由4个模型构成，其中2个模型会在初次使用时自动加载权重，2个模型则需要手动下载权重文件。
1. 将 [layoutlmv3-base-finetuned-publaynet](https://huggingface.co/HYPJUDY/layoutlmv3-base-finetuned-publaynet/tree/main) 目录下的所有文件下载于`/path/to/layoutlmv3-base-finetuned-publaynet`中，同时修改[cascade_layoutlmv3.yaml](cascade_layoutlmv3.yaml)文件中的`WEIGHTS: "/path/to/layoutlmv3-base-finetuned-publaynet/model_final.pth"`。此处的`path/to`可修改为本地想要存放权重的路径。
2. 下载 [Math Formula Detection(MFD)](https://www.dropbox.com/s/7xel0i3iqpm2p8y/model_final.pth?dl=1) 文件（可能需要在外网VPN环境下载），放置于`path/to/MFD/`路径下，同时修改[pdf_parse_new.py](pdf_parse_new.py)中的第79行`/path/to/MFD/model_final.pth`。此处的`path/to`可修改为本地想要存放权重的路径。

## 运行

在命令行运行 `uvicorn main:app --host '0.0.0.0' --port 8080 --reload`，其中8080可为想要使用的端口号，运行成功即可访问`http://IP地址:端口号/docs`。
