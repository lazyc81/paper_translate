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

