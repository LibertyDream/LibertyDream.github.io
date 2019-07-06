---
layout:     post
title:      在远程服务器上运行Jupyter NoteBook
subtitle:   
date:       2019-07-06
author:     一轩明月
header-img: img/post-bg-ios9-web.jpg
catalog: 	 true
tags:
    - trick
    - Jupyter Notebook
---

# 在远程服务器上运行 Jupyter Notebook

> 文章编译自：
>
> https://towardsdatascience.com/running-jupyter-notebooks-on-remote-servers-603fbcc256b3

一般数据科学家的常用工具里一定少不了 Jupyter Notebook。借助 Jupyter Notebook 的交互式特性能有效增强生产力，简化数据分析和模型设计实验过程，大幅精简“编程-看结果—再编程”的循环过程，反馈链条变短，“所写即所得”。

多数情况下，在笔记本或工作站上运行 Jupyter Notebook 已经足够。但如果你想处理大数据集，执行成本高昂的计算任务，或者学习一些复杂模型，笔记本能提供的算力和空间就捉襟见肘了。你可能在一个大规模图上跑图卷积网络，也可能要在大体量语料库上训练循环神经网络进行机器翻译，因此需要更多的 CPU 核心，内存和一些 GPU。幸运的是，你可以在远程服务器上获取这些资源。

如果你的远端服务器有图形用户界面（GUI），那很简单。你可以用远程桌面软件访问远端服务器，像在你的笔记本上一样使用 Jupyter Notebook 即可。

但是，很多人的服务器并没有GUI。如果你是这样的情况，你可以在你的笔记本上编写 Python 脚本设计实验，在小规模数据上确认它可行，之后复制到远程服务器上，通过命令行执行。你甚至可以通过`jupyter nbconvert --to script your_notebook.ipynb`将设计好实验的 notebook 转换成脚本执行。这样的方式当然能让你在远程服务器上跑实验，但是同时也放弃了 Jupyter Notebook 交互、可视化实验的特性。太可惜了。

这里我会向你展示如何在远程服务器上跑 Jupyter Notebook 并在你的笔记本上访问它。我还会向你展示如何通过两条`bash`命令使整个过程更加流畅。

## 启用远程 Notebook 服务器

使用SSH（Secure Shell Protocol）在远程服务器上启动 Jupyter Notebook 服务。基本语法如下：

```bash
ssh username:password@remote_server_ip command
```

具体命令形式取决于你的现实情况。我这里因为要和其他人共享服务器，所以没有在共享环境上安装 Jupyter。因此我得先到我的工程目录下激活虚拟环境并启动 notebook server。这种情形下我会在远程服务器上执行下列三条`bash`命令：

```bash
cd project_folder
.virtual_enviroment/bin/activate
jupyter notebook --no-browser --port=8889
```

如果远程服务器没有 GUI 自然不能显示浏览器了，所以我给`jupyter notebook`命令加上了`--no-browser`参数再启动 Jupyter Notebook。通过`--port=8889`参数把端口从默认端口8888改成了8889，这是只是个人习惯，本地和远程 notebook 运行在不同端口下方便我确认我的运行环境。

在远程服务器上执行命令，我们要执行组合命令：

```bash
nohup ssh -f username:password@remote_server_ip "cd project_floder; .virtual_environment/bin/activate; jupyter notebook --no-browser --port=8889"
```

注意我使用分号`;`而不是换行将三个命令写在了一行命令中。执行这条命令会在8889端口启动 Jupyter Notebook 服务，并使其在后台运行。最后，我使用了参数`-f`后台执行`ssh`，同时添加了`nohup`命令屏蔽了这期间的所有输出来保证你能正常继续使用终端(terminal)。通过[这里](https://www.computerhope.com/unix/unohup.htm)了解`nohup`命令。

## 访问你的 Notebook

通过下方给出的url形式来访问notebook：

```bash
remote_server-ip:8889
```

执行这条命令需要你记住 IP 地址或者为网页添加书签。但我们可以通过转发端口来使访问远程 notebook 的体验和本地 notebook 别无二致。

```bash
nohup ssh -N -f -L localhost:8889:localhost:8889 username:password@remote_server_ip
```

参数`-N`告诉`ssh`没有远程命令要执行。此时我们不需要执行任何远程命令。前面说过了，`-f`参数使`ssh`后台运行。至于`-L`参数指定了端口转发内容，其语法为

```bash
local_server:local_port:remote_server:remote_port
```

我们构建的命令指定所有发送到本地机器，比如你的笔记本，`8889`端口的请求，都转到位于`username:password@remote_server_ip`的远端机器上的`8889`端口处理。如前文所述，`nohup`保证了终端“清净”。

通过上述步骤，你现在可以在你的本地浏览器上的`localhost:8889`访问远程 Jupyter Notebook 服务上的内容了。

## 停止远程 Notebook 服务

原则上讲，你可以让notebook服务在远程服务器上一直运行下去（除去重启和崩溃的情况），但你可能需要停下服务来更新`jupyter`。如果你有这样的需求，有两个方法：通过浏览器或者命令行

### 通过浏览器

如果你使用的 Jupyter Notebook 版本比较新，你可以在浏览器窗口右上角找到一个`Quit`按钮。如果你点击了退出，你必须通过前文所说的启动命令重启服务。

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-07-07-quit-button.png)

### 通过命令行

如果你的 Jupyter 版本还没有退出按钮，或者你就是更喜欢通过终端工作，你可以通过命令行停止服务。Jupyter 有一个`stop`命令来停止notebook：

```bash
jupyter notebook stop 8889
```

`8889`是端口号。在远程服务器上你可以依靠下列命令：

```bash
ssh username:password@remote_server_ip "jupyter notebook stop 8889"
```

不幸的是这条命令当前是有问题的，但我把它留在这里希望将来可以执行。作为替代方案，你可以通过下面的命令结束`jupyter`进程：

```bash
ssh username:passowrd@remote_server_ip "pkill -u username jupyter"
```

`-u username`指定了只有`username`启动的`jupyter`进程才要被停掉。这条语句可以在你同时运行多个 notebook 的时候一次性中断这些进程。当然，你也可以选择到服务器上手动启动、管理 notebook 服务并保证终端一直开启，此时你可以通过`CTRL + C`键盘指令结束 notebook 服务。

## 让操作更加流畅

如果要记住所有这些命令会很麻烦，幸运的是你可以为每一个命令创建一个bash别名来简化操作。将下列语句添加到`~/.bashrc`文件下：

```bash
alias port_forward='nohup ssh -N -f -L localhost:8889:localhost:8889 username:password@remote_server_ip'
alias remote_notebook_start='nohup ssh -f username:password@remote_server_ip "cd rne; . virtual_environment/bin/activate; jupyter notebook --no-browser --port=8889"; port_forward'
alias remote_notebook_stop='ssh username:password@remote_server_ip "pkill -u username jupyter"'
```

在终端执行`source .bashrc`加载这些命令。现在你可以通过`remote_notebook_start`和`remote_notebook_stop`命令启动远程 notebook 服务(包含转发端口)或是关闭它。