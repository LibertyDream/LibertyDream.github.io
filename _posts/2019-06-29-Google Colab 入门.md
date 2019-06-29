---
layout:     post
title:      启程
subtitle:   答疑解惑的简明教程
date:       2019-06-29
author:     一轩明月
header-img: img/post-bg-coffee.jpeg
catalog: 	 true
tags:
    - Colab
    - Data Science
---

# Google Colab 入门

> 编译自:
>
> https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c

你知道世上有一个地方，那里有免费的 GPU 资源。但就像肥美多汁的熟黑莓，挂在枝头却又有些遥不可及。

快如闪电的处理速度只待君采撷。到底如何挥动这利剑，来这里你已经拜对了山门。

![1561798172868](..\img\chapter\2019-06-29\2019-6-29-lightning.png)

有些人或许不知道，Google 做了一件很酷的事，它提供基于 Jupyter Notebook 的免费云服务并白送 GPU 资源。这对于你改善编程技能是一个利好，更重要的是它让任何人都有机会使用流行库，比如 **TensorFlow**，**PyTorch**，**Keras** 和 **OpenCV** 等，开发属于自己的深度学习应用。

> Colab 提供的 GPU 是免费的!
>
> Colab 提供的 GPU 是免费的!
>
> Colab 提供的 GPU 是免费的!

既然已经免费，有些限制是正常的（详细的内容可以到官方 FAQ 页面查看）。Colab 支持 **Python 2.7**、**Python 3.6**，暂不支持 R 和 Scala。计算任务不能超过 4 小时，计算资源一般在 12G 内存，50G 硬盘左右。如果你爱折腾，不介意重新上传文件，这些限制也就无所谓了。

无论是想锻炼 Python 编码功底还是深入操练深度学习库（TensorFlow**，**PyTorch**，**Keras 和 OpenCV）Colab 都是理想的选择。Colab 上你可以上传、创建、编辑、存储和分享 notebook，挂载谷歌云盘（Google Drive）并使用任意你已存入的内容，导入你想加载的目录。无论是上传本地个人 Jupyter Notebook，还是直接从 GitHub 加载 notebook，亦或导入 Kaggle 文件，下载编辑好的 notebook，你可以在 Colab 上做你想做的一切。

棒极了，不是吗？

初次使用 Colab 你会因其非凡和易用而惊叹，但也会遇到一些小问题。如果你对 Jupyter Notebook 了如指掌，在使用 Colab 时体验会如丝般顺滑，可一些细小的差异决定了你是乘免费 GPU 自由飞翔还是坐在电脑前头撞南墙……

![1561801196001](..\img\chapter\2019-06-29\2019-6-29-crap-head.png)

这篇文章赠与那些困惑、沮丧，只是想跑跑模型而不能的人们。

## 网盘设置

### 为你的 notebook 建一个文件夹

> 技术上讲，如果你只是想玩一玩 Colab，这一步并不是必须的。但鉴于 Colab 工作在你的 Google 网盘之上，专设一个文件夹总是好的。你可以前往 Google Drive 点击“新建/New"创建一个新文件夹。我之所以提这点，是因为我的网盘上目前零散飘落着一地 notebook，而我现在必须要处理这个问题。

![1561801945051](..\img\chapter\2019-06-29\2019-6-29-new-folder.png)

如果你愿意，在 Google 网盘内创建一个新的 Colab notebook。点击 “新建/New"并下拉菜单到“更多/More"，然后选择“Colaboratory”。

![1561802458940](..\img\chapter\2019-06-29\2019-6-29-new-colab-file.png)

其他情况下你都可以直接前往 Google Colab。

### 开始吧

重命名 notebook 方式：

1. 点击 notebook 文件名并修改
2. 点击“文件/File"下拉菜单选择”重命名/Rename“

![1561804860746](..\img\chapter\2019-06-29\2019-6-29-rename-file.png)

### 启用你的免费GPU

想要使用 GPU ? 操作很简单，在“代码执行/runtime”下拉框中选择“更改运行时类型/ change runtime type”，并在硬件加速下拉菜单中选择 GPU 即可。

![1561814538398](..\img\chapter\2019-06-29\2019-6-29-set-up-GPU.png)

### 开始敲代码吧

现在你可以随时开始运行代码了，就像在任何地方的 Jupyter Notebook 中一样。

![run_code](..\img\chapter\2019-06-29\2019-6-29-run-code.png)

### 再高级点

想要挂载 Google 网盘？使用

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

之后你会看到一个链接。点击链接，最下方允许访问，复制给出的序列码，粘贴到 Colab 单元格下方的输入框中，按下回车，即可开始访问你的云盘啦！如果你在左侧侧框中没有看到你的云盘，点击侧框上方的“刷新/refresh"，云盘就会出现了。

（运行单元格，点击链接，复制页面给出的序列码，粘贴到输入框中，按回车，云盘挂载成功在左侧边框显示）：

![1561816105809](..\img\chapter\2019-06-29\2019-6-29-drive-gd.png)

此外，你可以随时可以通过下列代码访问云盘：

```python
!ls "content/gdrive/My Drive"
```

如果你想下载一个共享压缩文件，可以使用：

```python
!wget
!unzip
```

比如：

```python
!wget -cq https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
!unzip -qq flower_data.zip
```

执行上述命令可以获得优达学城的花朵数据。

如果你要上传小文件，当然可以使用一些简单语句完成。但是你还可以在左边侧框内点击“上传文件/upload files"达到同样的效果，如果你不想编辑代码来获取本地文件的话。

![1561816676866](..\img\chapter\2019-06-29\2019-6-29-upload-files.png)

Google Colab 对任何水准人士而言都是易用的，特别是当你熟悉 Jupyter Notebooks的时候。 但是想要获取大文件和指定的若干目录着实废了我一些功夫。

我在另一篇文章中单独介绍了通过 Google Colab 入门 Kaggle，如果你对此感兴趣，欢迎阅读！

## 导入库

导入要使用的库这再普通不过了，但有些例外。

多数时候你可以通过`import`像在平常 notebook 中一样导入你的库。

![1561818196208](..\img\chapter\2019-06-29\2019-6-29-import-normal.png)

**PyTorch 不同，在你运行任何其他 Torch  导入项前，你需要运行**

> 目前 Colab 已原生支持 PyTorch ！你已经不再需要运行下列代码，但是我选择保留以防一些人遇到问题

```python
# http://pytorch.org/
from os.path import exists
from wheel.pep425tags import get_abbr_impl,
get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(),
get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e
's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0')
else 'cpu'
!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-
0.4.1-{platform}-linux_x86_64.whl torchvision
import torch
```

之后就可以照常导入了。如果你直接尝试运行`import torch`会得到报错信息。个人十分推荐点击弹出的链接，这样你可以立刻得到这些代码，之后你只需要点击“INSTALL TORCH"来将 torch 导入你的 notebook。代码会在屏幕左侧弹出，然后点击“INSERT"即可。

![1561819341382](..\img\chapter\2019-06-29\2019-6-29-import-torch-1.png)

![1561819381404](..\img\chapter\2019-06-29\2019-6-29-import-torch-2.png)

import 语句导入不了了？尝试一下 pip 安装吧！记得在 Google Colab 中多数命令前需要感叹号。

```python
!pip install -q keras
import keras
```

或者：

```python
!pip3 install torch torchvision
```

还可以：

```python
!apt-get install
```

我确实发现 Pillow 有些问题，但你可以通过执行下列代码解决：

```python
import PIL
print(PIL.PILLOW_VERSION)
```

如果你想使用 5.3 版本以下内容，在“代码执行/runtime”下拉菜单下选择重启代码执行程序，并重新运行单元格，你的目的就达到了。

创建新的 notebook 很简单，“文件/File"下拉菜单内选择”新建 Python3 Notebook“。如果你想打开指定内容，”文件/File“菜单内选择”打开 Notebook...“

![1561820361347](..\img\chapter\2019-06-29\2019-6-29-new-book.png)

然后你的屏幕会是这样：

![1561820542947](..\img\chapter\2019-06-29\2019-6-29-new-book-1.png)

如你所见你可以打开近期文件，无论是 Google 网盘、GitHub 文件，亦或自己上传一个notebook

GitHub 选项很棒，你可以通过组织或者用户来方便的检索文件。如果找不到自己想要的内容，检查仓库菜单试试看。

![1561820768034](..\img\chapter\2019-06-29\2019-6-29-GitHub.png)

![1561820909631](..\img\chapter\2019-06-29\2019-6-29-GitHub-1.png)

![1561820988253](..\img\chapter\2019-06-29\2019-6-29-GitHub-2.png)

## 随时存储

保存你的作品很简单，可以执行“命令键 + s”或者“文件/File"下列菜单内选择保存。”文件/File"下的“Save a Copy in Drive"可以帮助你在网盘内存储一份notebook备份。你还可以选择”文件/File"->"download .ipyb"或者"download .py"下载当前notebook

![1561821352069](..\img\chapter\2019-06-29\2019-6-29-save-book.png)

入门教程难免挂一漏万，但这些应该已经可以帮你起步并使用免费的 GPU 了。