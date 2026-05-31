# FaceNet 人脸验证项目

这是一个基于 PyTorch 的 FaceNet 人脸验证项目，用于判断两张人脸图片是否来自同一个人。项目默认提供 `mobilenet` 与 `inception_resnetv1` 两种主干网络配置，并保留了预测、训练和 LFW 数据集评估流程。

## 目录结构

```text
facenet-pytorch-main/
|-- datasets/              # 训练图片目录，每个子文件夹代表一个身份类别
|-- img/                   # 示例图片
|-- lfw/                   # LFW 评估数据集目录
|-- logs/                  # 训练日志和训练得到的权重
|-- model_data/            # 预训练权重、LFW pairs 文件和评估结果图
|-- nets/                  # 模型结构与训练损失
|-- utils/                 # 数据读取、训练循环、评估指标等工具
|-- eval_LFW.py            # 在 LFW 上评估模型
|-- facenet.py             # 预测封装类，推理时主要修改这里的配置
|-- predict.py             # 交互式图片相似度预测入口
|-- train.py               # 模型训练入口
|-- txt_annotation.py      # 根据 datasets 生成 cls_train.txt
`-- requirements.txt       # Python 依赖
```

## 环境安装

建议在虚拟环境中安装依赖，避免影响电脑里的其它 Python 项目。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果使用 GPU，请确认本机的 CUDA、显卡驱动和 PyTorch 版本匹配。项目代码中默认开启 CUDA，没有 GPU 时需要把以下文件里的 CUDA 开关改成 `False`：

- `facenet.py` 中的 `"cuda": True`
- `train.py` 中的 `Cuda = True`
- `eval_LFW.py` 中的 `cuda = True`

## 预训练权重

当前项目的 `model_data/` 目录已经包含常用权重文件，代码会优先从本地读取这些文件，不再依赖外部仓库下载链接：

- `facenet_mobilenet.pth`
- `facenet_inception_resnetv1.pth`
- `backbone_weights_of_mobilenetv1.pth`
- `backbone_weights_of_inception_resnetv1.pth`

预测或评估时，需要保证 `model_path` 和 `backbone` 对应。例如：

```python
"model_path": "model_data/facenet_mobilenet.pth"
"backbone": "mobilenet"
```

如果换成 Inception-ResNetV1 权重，也要同步改成：

```python
"model_path": "model_data/facenet_inception_resnetv1.pth"
"backbone": "inception_resnetv1"
```

## 图片预测

默认预测入口是 `predict.py`，运行后会依次输入两张图片路径，并输出两张图片特征向量之间的距离。

```powershell
python predict.py
```

示例输入：

```text
img\1_001.jpg
img\1_002.jpg
```

距离越小，表示两张图片越相似；距离越大，表示差异越明显。实际使用时建议结合自己的业务数据确定阈值，不要只依赖固定经验值。

如果要使用自己训练得到的权重，主要修改 `facenet.py` 中的配置：

```python
_defaults = {
    "model_path": "model_data/facenet_mobilenet.pth",
    "input_shape": [160, 160, 3],
    "backbone": "mobilenet",
    "letterbox_image": True,
    "cuda": True,
}
```

## 训练数据格式

训练集放在 `datasets/` 目录下，每个人一个文件夹，文件夹内放该人的人脸图片。

```text
datasets/
|-- person_001/
|   |-- 001.jpg
|   `-- 002.jpg
|-- person_002/
|   |-- 001.jpg
|   `-- 002.jpg
`-- ...
```

建议每个类别准备尽量多、质量稳定的人脸图片。图片数量过少时，训练过程容易不稳定，评估结果也不可靠。

## 训练步骤

1. 准备训练图片，按上面的目录格式放到 `datasets/`。
2. 生成训练标注文件：

```powershell
python txt_annotation.py
```

该命令会在根目录生成 `cls_train.txt`，内容是类别编号和图片绝对路径。

3. 打开 `train.py`，根据实际情况修改常用参数：

```python
Cuda = True
annotation_path = "cls_train.txt"
input_shape = [160, 160, 3]
backbone = "mobilenet"
model_path = "model_data/facenet_mobilenet.pth"
batch_size = 96
Epoch = 100
save_dir = "logs"
lfw_eval_flag = True
```

训练时需要注意：

- `batch_size` 必须是 3 的倍数。
- 显存不足时，优先调小 `batch_size`。
- 没有 LFW 数据集时，可以先把 `lfw_eval_flag` 改为 `False`。
- 继续训练已有模型时，把 `model_path` 指向已有权重。
- 从主干网络预训练权重开始训练时，可以把 `model_path` 设为空字符串，并把 `pretrained` 设为 `True`。
- 从零开始训练时，把 `model_path` 设为空字符串，并把 `pretrained` 设为 `False`。

4. 开始训练：

```powershell
python train.py
```

训练产生的权重和日志会保存到 `logs/` 目录。

## LFW 评估

LFW 评估入口是 `eval_LFW.py`。运行前确认：

- `lfw/` 目录中已经放入 LFW 图片。
- `model_data/lfw_pair.txt` 存在。
- `eval_LFW.py` 中的 `model_path` 和 `backbone` 与当前权重对应。

运行评估：

```powershell
python eval_LFW.py
```

评估完成后，ROC 图会保存到：

```text
model_data/roc_test.png
```

## 常见调整点

`facenet.py`：预测阶段的权重、主干网络、输入尺寸、CUDA 开关。

`train.py`：训练阶段的主干网络、初始权重、学习率、优化器、批大小、训练轮数、日志目录和 LFW 评估开关。

`eval_LFW.py`：评估阶段的权重、主干网络、LFW 路径、batch size 和 ROC 保存路径。

`txt_annotation.py`：训练集目录默认是 `datasets`，如果训练图片放在其它位置，需要同步修改 `datasets_path`。

## 使用建议

- 训练前先用少量图片跑通 `txt_annotation.py` 和 `train.py`，确认环境没有问题后再换成完整数据集。
- 预测前先确认权重文件存在，并且 `model_path` 与 `backbone` 对应。
- 人脸识别效果受图片质量影响很大，建议尽量使用清晰、正脸、光照稳定的图片。
- 如果只是做图片相似度验证，优先从 `predict.py` 和 `facenet.py` 开始看；如果要重新训练，再看 `train.py` 和 `utils/` 下的数据处理逻辑。
