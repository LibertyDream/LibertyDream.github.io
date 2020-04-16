---
layout:     post
title:      Colab + Kaggle 入门基础之数据导入
subtitle:   纯新手教程
date:       2019-07-13
author:     一轩明月
header-img: img/post-bg-code.jpg
catalog: 	 true
tags:
    - trick
    - Google Colab
    - Kaggle
excerpt:   知道哪里获取数据集，也知道自己希望它们在哪里进行运算处理。但是如何简单便捷的将数据从 Kaggle 导入 Google Colab呢？ 
---

> 文章编译自：
>
> https://towardsdatascience.com/setting-up-kaggle-in-google-colab-ebb281b61463

你已经知道哪里获取数据集，也知道自己希望它们在哪里进行运算处理。但是如何简单便捷的将数据从 Kaggle 导入 Google Colab呢？

遇到Google Colab绝对是入行AI、Deep Learning和Machine Learning以来最令人欣喜的事情。它免费向人们提供GPU资源，如果你刚入行，就从Colab开始吧，我在[另一篇文章](https://libertydream.github.io/2019/06/29/Google-Colab-入门/)里介绍了怎么入门Google Colab。在Colab上导入、处理Kaggle数据的方法则总结成本文。

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-07-13_run_in_rain.png)

Colab本身使用很方便，但还有一些细节你需要知道，Kaggle就是其中之一。下面是我找到的首次在Colab加载Kaggle数据时最便捷的方法：

## 准备工作

_小提示：为了加载 Kaggle 数据，你需要先注册一个 Kaggle 账号（免费），并同意你所参加比赛给出的条款和条件_

首先要获取 Kaggle 口令。前往你的账号主页（下拉菜单右上角点击头像进行访问）

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-07-13_Kaggle_account.png)

下拉到底部，找到 API 模块，点击“Create New API Token”。

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-07-13_create_new_token.png)

之后会自动下载一个**kaggle.json**文件，请妥善保存该文件。打开该文件会看到类似下面格式的内容：

```
{“username”:”YOUR-USER-NAME”,”key”:”SOME-VERY-LONG-STRING”}
```

保留这行内容便于未来复制粘贴。

接着，前往 Colab 并新建一个 notebook。如果想要立刻使用 GPU。可以在“runtime”菜单下，选择“change runtime type"并在”Hardware accelerator"菜单内选择 GPU，点击保存。

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20190715234741.png)

之后，开始安装 Kaggle。这和你在本地 Jupyter Notebook 上安装很像，记得前面加上感叹号`!`

```bash
!pip install kaggle
```

可以使用`!ls`来查看是否已经存在名为Kaggle的文件夹。如果没有使用下列命令创建一个。

```bash
!mkdir .kaggle
```

下一步，运行下列命令，有一些事情要注意：

* 这些语句不需要感叹号
* 将`username`和`key`的内容替换为前文准备好的内容

```python
import json
token = {"username":"YOUR-USER-NAME","key":"SOME-VERY-LONG-STRING"}
with open('/content/.kaggle/kaggle.json', 'w') as file:
    json.dump(token, file)
```

当我复制粘贴来运行这些指令时，有点小问题。不知道是什么原因，我必须删除并重新输入代码里的单引号以保证单元格正常运行。如果你报无法识别的错误，可以试一试。

接着运行

```bash
!cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
```

之后

```bash
!kaggle config set -n path -v{/content}
```

你也许会遇到下列警告：

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20190716000250.png)

这个问题可以通过下列语句解决：

```bash
!chmod 600 /root/.kaggle/kaggle.json
```

执行下列语句查看 Kaggle datasets。

```bash
!kaggle datasets list
```

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20190716000552.png)

如果要访问特定数据集，使用：

```bash
!kaggle datasets list -s sentiment
```

以此来筛选出标题中带有"sentiment"的数据集。

## 下载数据

前往 Kaggle，找到你想要的数据集，在数据集页面上点击 API 按钮（会自动复制代码）

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20190716001156.png)

粘贴到单元格里，记住在开头加上`!`，并使用`-p /content`来指明路径。

```bash
!kaggle datasets download -d kazanova/sentiment140 -p /content
```

使用下列命令解压文件：

```bash
!unzip \*.zip
```

现在就可以愉快的查看、处理数据了。

```python
import pandas as pd
d = pd.read_csv('training.1600000.processed.noemoticon.csv')
d.head()
```

_文件名替换为你的，这不用说了吧_

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20190716001945.png)

