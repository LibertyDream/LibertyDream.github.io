---
layout:     post
title:      Google Colab 入门
subtitle:   答疑解惑的简明教程
date:       2019-06-29
author:     一轩明月
header-img: img/post-bg-2015.jpg
catalog: 	 true
tags:
    - Google Colab
excerpt:    Google Colab 提供了学生、研究者实践 ML/DL 所需的编码环境与硬件资源，特别是可以白嫖的 GPU，本文从网盘设置、新建文档起，手把手教你入门 Colab
---

> 编译自: [getting started with google colab](https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c)
>
> 2020-07-03 更新：添加快捷键、加载数据、外部数据导入等内容

Google 做了一件很酷的事，它提供基于 Jupyter Notebook 的免费云服务并白送 GPU 资源。这对于改进编程技能是一个利好，更重要的是它让任何人都有机会使用流行库，比如 **TensorFlow**，**PyTorch**，**Keras** 和 **OpenCV** 等，开发属于自己的深度学习应用。Colab 提供的 GPU 是免费的！Colab 提供的 GPU 是免费的！Colab 提供的 GPU 是免费的！

既然已经免费，有些限制是正常的（详细的内容可以到官方 FAQ 页面查看）。Colab 支持 **Python 2.7**、**Python 3.6**，暂不支持 R 和 Scala。计算任务不能超过 4 小时，计算资源一般在 12G 内存，50G 硬盘左右。如果你爱折腾，不介意重新上传文件，这些限制也就无所谓了。

无论是想提升 Python 编码功底还是深入操练深度学习库（TensorFlow**，**PyTorch**，**Keras 和 OpenCV）Colab 都是理想的选择。Colab 上你可以上传、创建、编辑、存储和分享 notebook，挂载谷歌云盘（Google Drive）并使用任意你已存入的内容，导入你想加载的目录。无论是上传本地个人 Jupyter Notebook，还是直接从 GitHub 加载 notebook，亦或导入 Kaggle 文件，下载编辑好的 notebook，你可以在 Colab 上做你想做的一切。

## 网盘设置

### 为你的 notebook 建一个文件夹

> 技术上讲，如果你只是想玩一玩 Colab，这一步并不是必须的。但鉴于 Colab 工作在你的 Google 网盘之上，专设一个文件夹总是好的。你可以前往 Google Drive 点击“新建/New"创建一个新文件夹。我之所以提这点，是因为我的网盘上目前零散飘落着一地 notebook，而我现在必须要处理这个问题。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-new-folder.png)

如果你愿意，在 Google 网盘内创建一个新的 Colab notebook。点击 “新建/New"并下拉菜单到“更多/More"，然后选择“Colaboratory”。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-new-colab-file.png)

相似的，如果是 Colab 中的 notebook 想要确认其在云盘中的位置，只需循着 “**File > Locate to Drive**” 路径，就能重定向到云盘了

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_locate_drive.png)

### 新建

如果不采用上述云盘启动方式，可以选择打开 [Colab 主页](https://colab.research.google.com/)，它会自动展示你早先的 notebook 并给出新建 notebook 的选项

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_index.png)

点击 `NEW  NOTEBOOK` 开启新笔记本，默认是 Python 3 环境

如果没看见该提示或是取消掉了，可以按 “**File > New Notebook**” 路径新建 notebook

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_new_notebook.png)

### 导入

想必你也看到了这两个标着 GitHub 和 Upload 的选项卡

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_import.png)

对 GitHub，需要 GitHub 对 Colab 的授权，然后会看到可访问的库，这样就能从中新建笔记本了

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_github.png)

GitHub 选项很棒，你可以通过组织或者用户来方便的检索文件。如果找不到自己想要的内容，检查仓库菜单试试看。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-GitHub.png)

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-GitHub-1.png)

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-GitHub-2.png)

如果要上传本地内容，只需按提示上传到 Colab 运行即可

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_upload.png)

### 重命名

重命名 notebook 方式：

1. 点击 notebook 文件名并修改
2. 点击 “文件/File" 下拉菜单选择 ”重命名/Rename“

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-rename-file.png)

### 键盘快捷键

Colab 绝大多数快捷键都和 Jupyter Notebook 相似，下面举一些重要的

- 显示所有快捷键 `ctrl+M+H`

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_shortkey.png)

- 新代码块 `ctrl+M+B`
- 单元格转换为 Markdown 文本 `ctrl+M+M`
- Markdown 转换为代码单元格 `ctrl+M` 后快速轻按 `Y`

### 换用 Python 2

随着 Python2 官方支持的终结，Colab 上 Python2 已不可用，如果有人发你 Python2 代码，可以前往下列地址

- http://bit.ly/colabpy2
- http://colab.fan/py2

最终会重定向到 https://colab.research.google.com/notebook#create=true&language=python2，这样就能调试 Python 2 代码了

### 启用免费GPU/TPU

想要使用 GPU ? 操作很简单，在  “代码执行/runtime”  下拉框中选择  “更改运行时类型/ change runtime type”，并在硬件加速下拉菜单中选择 GPU/TPU 即可。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-set-up-GPU.png)

如果想要用 Colab 接口调用本地 GPU，遵循以下步骤

1. 开启本地 Jupyter notebook 实例
2. 复制带有口令的 URL
3. 点击此处箭头（有时可能显示的是“连接”）

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_arrow.png)

4. 点击“连接到本地环境”

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_local_runtime.png)

5. 之后会看到下列提示

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_local_prompt.png)

6. 将链接粘贴到这，点击“连接”

### 网盘导入

想要挂载 Google 网盘？使用

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

之后你会看到一个链接。点击链接，最下方允许访问，复制给出的序列码，粘贴到 Colab 单元格下方的输入框中，按下回车，即可开始访问你的云盘啦！如果你在左侧侧框中没有看到你的云盘，点击侧框上方的“刷新/refresh"，云盘就会出现了。

（运行单元格，点击链接，复制页面给出的序列码，粘贴到输入框中，按回车，云盘挂载成功在左侧边框显示）：

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-drive-gd.png)

此外，你可以随时可以通过下列代码访问云盘：

```python
!ls "content/gdrive/My Drive"
```

此时左侧 “**Files**” 部分会看到 “**gdrive**”

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_gdrive.png)

设想你将文件上传到了 “Untitled folder”

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_gdrive_dir.png)

可以使用下列命令访问文件

```
myfile = open('gdrive/My Drive/Untitled folder/dataset1.csv')
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

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-upload-files.png)

Google Colab 对任何水准人士而言都是易用的，特别是当你熟悉 Jupyter Notebooks的时候。 但是想要获取大文件和指定的若干目录着实废了我一些功夫。

我在另一篇文章中单独介绍了通过 Google Colab 入门 Kaggle，如果你对此感兴趣，[欢迎阅读](https://libertydream.github.io/2019/07/13/在-Google-Colab上玩-Kaggle/)！

### Github 数据导入

如果数据在 Github 上，可以使用下列命令

```python
!git clone REPOLINK
%cd REPONAME
```

如果是压缩格式，就要解压使用

```python
!unzip GRP_radargrams.zip
%cd GRP_radargrams
```

稍后会介绍命令行的使用，这样可以更方便些。

## 导入库

导入要使用的库这再普通不过了，多数时候你可以通过`import`像在平常 notebook 中一样导入你的库。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-import-normal.png)

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

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-import-torch-1.png)

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-import-torch-2.png)

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

如果要下载包，比如 pillow，到 Colab，运行下列命令

```python
!pip install pillow
```

下载完成会显示

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_download_package.png)

更新包则用

```python
!pip install tensorflow --upgrade
```

包更新后可能需要重启运行环境

我确实发现 Pillow 有些问题，但你可以通过执行下列代码解决：

```python
import PIL
print(PIL.PILLOW_VERSION)
```

如果你想使用 5.3 版本以下内容，在“代码执行/runtime”下拉菜单下选择重启代码执行程序，并重新运行单元格，你的目的就达到了。

### 命令行

Colab 下的 bash 命令多是以 `!` 开头的，比如想要浏览目录，要用

```python
!dir
```

检查 CUDA 和 CUDNN，要用

```python
!nvcc --version
```

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-07-03_colab_bash_cuda.png)

## 随时存储

保存你的作品很简单，可以执行“命令键 + s”或者“文件/File"下列菜单内选择保存。”文件/File"下的“Save a Copy in Drive"可以帮助你在网盘内存储一份notebook备份。你还可以选择”文件/File"->"download .ipyb"或者"download .py"下载当前notebook

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-save-book.png)

### 开始敲代码吧

现在你可以随时开始运行代码了，就像在任何地方的 Jupyter Notebook 中一样。入门教程难免挂一漏万，但这些应该已经可以帮你起步并使用免费的 GPU 了。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-6-29-run-code.png)