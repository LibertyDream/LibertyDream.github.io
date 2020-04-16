---
layout: post
title: 用 Cython 加速你的 Python 代码  
date: 2019-07-27
author: 一轩明月
header-img: img/post-bg-future.jpg
catalog: true
tags:
    - Cython
excerpt:    Python 运行效率上做出了很大的牺牲，为了提升速度可以考虑 Cython，它会将 Python 代码转为 C 程序，能便捷地整合到 Jupyer Notebook 等环境中，本文介绍了 Cython 的安装与使用方法，并展示了它的执行效率
---

> 文章编译自：
>
> https://towardsdatascience.com/speed-up-your-python-code-with-cython-8879105f2b6f

## 引言

如果你写过 Python 代码，你一定感受过代码运行时长超出预期的感觉。有很多方法能让你的 Python 运行效率变高，但通常仍旧会比 C 要慢。这源自于 Python 是动态语言，会将很多 C 在编译时解决的内容放到运行时才解决。

尽管如此，如果你像我一样喜欢 Python 并希望提高运行速度，那么你可以考虑一下**Cython**。 虽然 Cython 是一个独立的编程语言，可是它很容易整合到你的 Jupyter Notebook 之类的工作流中。代码运行时， Cython 会将你的 Python 代码转为 C 程序执行，以此来获取效率提升。

## 安装 CPython

为了能够使用 Cython，你需要安装一个 C 编译器。安装流程会因你的操作系统不同而不同。比如对于 Linux 来说，gcc(GNU C Complier) 编译器通常是现成的。对于 Mac  系统，可以下载 Xcode 来安装 gcc。如果是 Windows，安装过程会有一些复杂，可以参见[CPython Git指南](https://github.com/cython/cython/wiki/InstallingOnWindows)。

安好 C 编译器后执行下列语句：

```bash
pip install Cython
```

## 使用Cython

我们借助 Jupyter Notebook 展示 Cython 的能力，在 notebook 里使用 Cython 需要用到 IPython 魔法命令。魔法命令以百分号开头并给你的工作流赋予额外特性。魔法命令通常有以下两种：

1. 行级魔法以单个`%`开头并只作用在一条输入上
2. 块级魔法以两个`%`开头并作用于多行输入上

首先，运行：

```bash
%load_ext Cython
```

当我们在一个代码格内运行 Cython 时，必须将魔法命令`%%cython`置于格内，这样就可以写 Cython 代码了。

## Cython 有多快

Cython 和常规 Python 代码的运行效率差异取决于代码本身。比如你要跑一个计算量大变量又多的循环，Cython 会比常规 Python 快很多。递归函数的运行效率上 Cython 也大幅优于 Python。

可以通过斐波那契数列来证实一下。斐波那契数列就是在一个数列中，某一个数的值是由它前面两个数相加得到的。 Python 代码如下：

```python
def fibonacci(n):
    if n < 0:
        print("1st fibonacci number = 0")
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-07-27_fib_python.png)

如图中所示，找到序列中第39个数需要花费13.3秒。挂起时间(Wall time)是指从函数调用开始到结束所花费的总时间。

让我们看一下 Cython 的表现

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-07-27_fib_cython_1.png)

如你所见，我们使用魔法函数准许该代码格使用 Cython。`-a`能让我们看到有多少 Python 交互并高亮显示。我们的目标就是消除黄色行使其变成白底，这样就不会有 Python 交互并且所有代码都以 C 运行。你也可以点击旁边的`+`看一看翻译成 C 后的代码。

接下来我们运行与上面相似的代码，只是我们现在实际已经可以使用静态声明并将 n 定义为整型。

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-07-27_fib_cython_2.png)

可以看到，本次实验中 Cython 比 Python 快了 6.75倍，这充分展示了 Cython 对 Python 的改善。

## 更多的尝试

如果你知道对于那些 Cython 作者还没添加即用声明的C代码，C和Cython也是可以访问的。那么，通过下列类似代码你可以给 C 函数加一个 Python 包装器并把它加到模块字典里面。

```python
%%cython
    cdef extern from "math.h":
        cpdef double sin(double x)
```

Cython的能力不止于此，还有并行化等，可以在[这里](http://docs.cython.org/en/latest/index.html)查看更多文档信息。如果你熟悉C的话，强烈建议通读一遍文档。

