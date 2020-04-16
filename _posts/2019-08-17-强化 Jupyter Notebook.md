---
layout:     post
title:      Jupyter Notebook 强化指南
date:       2019-08-17
author:     一轩明月
header-img: img/post-bg-rwd.jpg
catalog: 	 true
tags:
    - Jupyter Notebook
excerpt:    原生 Jupyter Notebooks 界面十分简陋，缺少很多开发实用、用户友好的特性。本文列举了一些 Jupyter Notebooks 附加组件/扩展，以及一些能提高效率的 jupyter 命令。一切旨在强化 Jupyter Notebooks。
---

Jupyter Notebooks 是机器学习和数据科学领域特别钟爱的 Python 编程环境。但是原生 Jupyter Notebooks 界面十分简陋，缺少很多开发实用、用户友好的特性。

本文我列举了一些 Jupyter Notebooks 附加组件/扩展，以及一些能提高效率的 jupyter 命令。一切旨在强化 Jupyter Notebooks。

跟随本文，你的 Jupyter Notebooks 可以拥有（或部分拥有）下列特性：

1. 环境热切换。在线切换不同的 Conda 环境，免于重启 Jupyter Notebook
2. 单击生成带链接的内容目录
3. 弹出式便笺，可以在不影响主笔记的情况下测试代码或尝试开发
4. 代码格内的代码折叠
5. 单击隐藏代码格。隐去代码可以突出图表和图片，便于讲故事
6. 变量检查
7. Markdown 内容格拼写检测
8. ZenMode，类似 Xmind 的 Zen 模式，屏前只保留代码保证专注
9. 代码片段菜单，在线添加列表推导式等常用的 Python 结构
10. 一个漂亮舒适的深蓝配色方案

## 开始行动

### 深蓝午夜主题

Jupyter Notebooks 原生白色背景下工作久了眼睛会很难受。使用下列命令安装黑色主题

```
# 关闭并退出 Notebook server
# 保证当前 base 环境
conda activate base

# 安装 jupyterthemes
pip install jupyterthemes

# 更新至最新版本
pip install --upgrade jupyterthemes
```

安完包并更新后，使用下列命令切换到深蓝午夜主题

```
# 启用深色主题
jt -t onedork -fs 95 -altp -tfs 11 -nfs 115 -cellw 88% -T
```

### 环境热切换

将 Anaconda 内创建的自定义环境作为内核添加到 Jupyter Notebooks。这样切换环境时只需简单的选内核即可。切换内核不用重启 notebook。

假设你在 Anaconda 创建了两个自定义环境，my_NLP 和 gym。执行下列命令将其添加到 Jupyter Notebook

```
# 首先关闭并退出 Jupyter Notebook server
# 终端内激活环境
conda activate my_NLP

# 安装 IPython 内核
pip install ipykernel

# 将环境连接到 Jupyter
python -m ipykernel install --user --name=my_NLP

# 其他环境执行相同步骤
conda activate gym
pip install ipykernel 
python -m ipykernel install --user --name=gym
```

现在打开 Jupyter Notebooks，在 Kernel 菜单下的 Change Kernel 选项卡内可以看到所有环境了，点击切换。

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-08-17_add_env_to_jupyter.png)

### 其他特性

要获得其他特性，需要首先安装 _[nbextensions for Jupyter Notebooks](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html)_，执行下列命令：

```
# 关闭并退出 Jupyter Notebook server
# 确保在 base 模式下
conda activate base

# 安装 nbextensions
pip install jupyter_contrib_nbextensions

# 安装必要的 JS 和 CSS 文件
jupyter contrib nbextension install --system
```

启动 Jupyter Notebook server，你会看到首页内出现了第四个选项 Nbextensions。点开它你会看到各种梦寐以求的特性

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-08-17_nbextensions.png)

可以看到扩展十分多，但不是所有都用得到的。下面是我使用的一些：

1. *Table of Contents(2)* ——单击即可对整个 notebook 创建目录，且带有超链接
2. *Scratchpad*——十分好用的便笺，单独开辟环境运行代码而不干扰 notebook 其余内容
3. *Codefolding* ——代码折叠，无需解释
4. *Hide Input All*——隐藏所有代码格，同时留下 output 和 Markdown 内容格。
5. *Variable Inspector*——有点像 Spyder IDE 里的变量检查器
6. *Spellchecker* —— 对 Markdown 内容格执行拼写检查
7. *Zenmode*——除了代码其他的杂乱内容都从屏幕消除，保证专注
8. *Snippets Menu* —— 很棒的常见代码段集合，从列表表达式到 pandas。最妙的在于你可以修改窗口部件并添加自定义代码段

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-08-17_snippets_menu.png)

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-08-17_scratch_pad.png)

上面只是我常用的一些扩展，你可以自行尝试探索更多。一些扩展也很有趣，比如*ScrollDown*， *table_beautifier*， 和 *Hinterland*

### 后记

1. 如果没有安装 Anaconda，请忽略所有带有 conda 的命令。当然还有环境热切换的部分
2. 如果遇到大部分扩展显示 *possibly incompatible*，去掉上方*disable configuration for nbextensions without explicit compatibility* 的对勾
3. 如果不喜欢深蓝主题，终端执行 `jt -r`。重启 Jupyter Notebook 并清理浏览器缓存