---
layout:     post
title:      Python3 技巧：处理 Windows 和 Mac/Linux 文件路径差异的窍门
date:       2019-08-10
author:     一轩明月
header-img: img/post-bg-os-metro.jpg
catalog: 	 true
tags:
    - trick
    - Python
excerpt:    路径分隔符在不同系统间的差异有时对编程来讲是个不大不小的烦恼，本文探讨了传统 os.path 模块和解决方案，并给出了更好的路径编码方式——使用 pathlib 模块
---

> 文章编译自：
>
> https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f

如果你有编程经历，处理文件路径的时候可能会有些闹心，Microsoft Windows 使用反斜线`\`分隔文件，而其他系统几乎都是用正斜线`/`。

```bash
Windows filenames:
C:\some_folder\some_file.txt

Most other operating systems:
/some_folder/some_file.txt
```

这是一个历史遗留问题，初代 MS-DOS 系统使用正斜线分隔命令行选项。当微软在 MS-DOS 2.0 添加文件夹功能的时候，正斜线已经被用了，就只能用反斜线代替了。几十年过去了，我们依然被这点麻烦困扰。

如果想让你的 Python 脚本在 Windows 和 Mac/Linux 上都能运行，那你必须为这类平台差异专门做出调整。好消息是，Python3 提供了名为 **pathlib** 的新模块来专门处理文件。

让我们快速对比一下手动处理和使用 pathlib 间的差异，感受一下 pathlib 的便利。

##错误示例：手工编辑文件路径 

假设我们有如下的目录结构，且你想在 Python 程序里打开数据文件：

![]( https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-08-10__eg_dir.png)

下面这种编码方式要不得：

```python
data_folder = "source_data/text_files/"

file_to_open = data_folder + "raw_data.txt"

f = open(file_to_open)

print(f.read())
```

注意因为我用的 Mac，所以我使用 Unix 风格的正斜线硬编码文件路径。这无疑会让 Windows 用户不满。

技术上讲，这段代码在 Windows 上依然能运行，因为 Python 内部有一个[程式](https://docs.python.org/3/library/os.html#os.altsep)调用了 Windows 系统的内置方法，当你在 Windows 上调用 **open()** 方法的时候能识别任意类型的斜线，Windows 下其他环境多数也如此。正因为这一系统特性，你不该依赖于此。不是所有 Python 库都能在平台和斜线类别不统一的情况下正常工作的，特别是当它们还要与其他程序或库打交道的时候。

不同类别斜线混用只在 Windows 上行得通。如果反过来，在 Mac/Linux 上编码时用反斜线只会得到错误警告：

```python
data_folder = "source_data\\text_files\\"

file_to_open = data_folder + "raw_data.txt"

f = open(file_to_open)

print(f.read())

# Mac 上会抛出异常:
# FileNotFoundError: [Errno 2] No such file or directory: 'source_data\\text_files\\raw_data.txt'
```

有鉴于此，硬编码文件路径只会招来他人鄙夷的目光，你应该努力避免这点。

## 传统方法：使用 Python 的 os.path 模块

Python 的 os.path 模块有大量工具用于处理这类操作系统特性相关的文件系统问题。

你可以使用 **os.path.join** 构建路径字符串，它会根据系统自动适配斜线种类。

```python
import os.path

data_folder = os.path.join("source_data", "text_files")

file_to_open = os.path.join(data_folder, "raw_data.txt")

f = open(file_to_open)

print(f.read())
```

这段代码在 Windows 和 Mac 都可以运行。但这用起来有点难受。写 **os.path.join()** 语句还要把路径生生分隔成独立部分不仅啰嗦且不直观

因为大部分 **os.path** 里的方法都这么别扭，很多程序员明知道调库更好但还是会“忘记”使用。这也便带来了很多跨平台的 bug 和为此抓狂的人们。

## 更好的方案：Python3 的 pathlib

Python 3.4 起引入了名为 **pathlib** 的新标准库专门处理文件和路径。

你只需要用正斜线构造的路径或文件名创建一个 Path 对象，剩下的就不用管了

```python
from pathlib import Path

data_folder = Path("source_data/text_files/")

file_to_open = data_folder / "raw_data.txt"

f = open(file_to_open)

print(f.read())
```

注意两点：

1. 调用 pathlib 方法要用正斜线。 Path 对象会根据系统适配相应的斜线。
2. 如果你想在路径上追加内容，直接用`/`连起来就好。和 os.path.join 说再见吧

但 pathlib 能做的远不止这些。比如我们不必调用 open 和 close 直接查看内容：

```python
from pathlib import Path

data_folder = Path("source_data/text_files/")

file_to_open = data_folder / "raw_data.txt"

print(file_to_open.read_text())
```

_提示：前面的示例都是有问题的，因为都没有显式关闭文件。而本方法完全避免了这一bug_

pathlib 使得许多标准文件操作异常简单：

```python
from pathlib import Path

filename = Path("source_data/text_files/raw_data.txt")

print(filename.name)
# prints "raw_data.txt"

print(filename.suffix)
# prints "txt"

print(filename.stem)
# prints "raw_data"

if not filename.exists():
    print("Oops, file doesn't exist!")
else:
    print("Yay, the file exists!")
    
with q.open() as f: f.readline()
# ...
# '#!/bin/bash\n'
```

你甚至可以使用 pathlib 显式的将 Unix 路径转换为 Windows 路径：

```python
from pathlib import Path, PureWindowsPath

filename = Path("source_data/text_files/raw_data.txt")

# 转换成 Windows 格式
path_on_windows = PureWindowsPath(filename)

print(path_on_windows)
# prints "source_data\text_files\raw_data.txt"
```

假如你确实是想安全的在代码里使用反斜线，你可以声明一个 Windows 格式路径，pathlib 可以转换成当下系统的适配形式。

```python
from pathlib import Path, PureWindowsPath

# 已经显式指明是 Windows 格式路径，使用正斜线也可以
filename = PureWindowsPath("source_data\\text_files\\raw_data.txt")

# 转换成适配当前系统的文件路径
correct_path = Path(filename)

print(correct_path)
# prints "source_data/text_files/raw_data.txt" on Mac and Linux
# prints "source_data\text_files\raw_data.txt" on Windows
```

如果想玩的花哨些，甚至可以用 pathlib 解析文件相对地址，解析网络分享路径和生成 file:// url。举例来讲，你可以通过两行代码在浏览器上打开本地文件:

```python
from pathlib import Path
import webbrowser

filename = Path("source_data/text_files/raw_data.txt")

webbrowser.open(filename.absolute().as_uri())
```

当然，本文只是简单介绍了 pathlib，它确实很好的代替了大量原本分布于各个 Python 模块内的文件相关操作。[查看更多](https://docs.python.org/3/library/pathlib.html)