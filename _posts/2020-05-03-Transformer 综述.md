---
layout:		post
title:  	Transformer 综述
subtitle:   
date:       2020-05-03
author:     一轩明月
header-img: img/post-bg-ai.jpg
catalog:    true
tags:
    - attention
    - Transformer
excerpt:    各式强化版 Transformer 模型已是层出不穷，本文揭示了怎样改进“朴素” Transformer 以用于超长注意力跨度，减少记忆和计算开销以及强化学习的适配等多种场景
---

> 编译自：The Transformer Family，[Lilian Weng](https://lilianweng.github.io/lil-log/)
>
> 文内有大量数学公式，chrome 浏览器可以安装[这个](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)插件，以便浏览

# 符号说明

| 符号                                                         | 含义                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $$d$$                                                        | 模型大小 / 隐态维度 / 位置编码尺寸                           |
| $$h$$                                                        | 多头注意力层中头的数量                                       |
| $$L$$                                                        | 输入序列片段长度                                             |
| $$\mathbf{X} \in \mathbb{R}^{L \times d}$$                   | 输入序列，其中每个元素都已映射为 $$d$$ 维嵌入向量，和模型尺寸相当 |
| $$\mathbf{W}^k \in \mathbb{R}^{d \times d_k}$$               | 键权重矩阵                                                   |
| $$\mathbf{W}^q \in \mathbb{R}^{d \times d_k}$$               | 查询权重矩阵                                                 |
| $$\mathbf{W}^v \in \mathbb{R}^{d \times d_v}$$               | 值权重矩阵，一般有 $$d_k = d_v = d$$                         |
| $$\mathbf{W}^k_i, \mathbf{W}^q_i \in \mathbb{R}^{d \times d_k/h}; \mathbf{W}^v_i \in \mathbb{R}^{d \times d_v/h}$$ | 每个头对应的权重矩阵                                         |
| $$\mathbf{W}^o \in \mathbb{R}^{d_v \times d}$$               | 输出权重矩阵                                                 |
| $$\mathbf{Q} = \mathbf{X}\mathbf{W}^q \in \mathbb{R}^{L \times d_k}$$ | 嵌入输入的查询                                               |
| $$\mathbf{K} = \mathbf{X}\mathbf{W}^k \in \mathbb{R}^{L \times d_k}$$ | 嵌入输入的键                                                 |
| $$\mathbf{V} = \mathbf{X}\mathbf{W}^v \in \mathbb{R}^{L \times d_v}$$ | 嵌入输入的值                                                 |
| $$S_i$$                                                      | 第 $$i$$ 个查询 $$\mathbf{q}_i$$ 要关照的键位集合            |
| $$\mathbf{A} \in \mathbb{R}^{L \times L}$$                   | 输入序列长度为 $$L$$ 的自注意矩阵 $$\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})$$ |
| $$a_{ij}\in \mathbf{A}$$                                     | 查询 $$\mathbf{q}_i$$ 和键  $$\mathbf{k}_j$$ 间的注意力大小  |
| $$\mathbf{P} \in \mathbb{R}^{L\times d}$$                    | 位置编码矩阵，第 $$i$$ 行 $$\mathbf{p}_i$$ 是输入 $$\mathbf{x}_i$$ 的位置编码 |

# 注意力与自注意

_注意力_ 是神经网络的一种机制，拥有该机制的模型能学会有选择地关照给定数据集以作预测。投放注意力大小取决于习得权重，因而模型输出结果通常是加权平均的形式。

_自注意_ 是注意力机制中的一种，模型通过观察样本中的其余部分来对该样本中的目标位置进行预测，直观上很像[非局部平均](https://en.wikipedia.org/wiki/Non-local_means)。同时要知道自注意具有排列不变性，换句话说它是一种集合上的运算。

注意力/自注意的形式五花八门，Transformer ([Vaswani 等, 2017](https://arxiv.org/abs/1706.03762)) 采用的是 _比例点积注意力_ ：给定查询矩阵 $$\mathbf{Q}$$ ，键矩阵 $$\mathbf{K}$$ 和值矩阵 $$\mathbf{V}$$ ,输出结果就是值向量的加权和，而各个值分得的权重则由查询和键向量的点积结果决定：
$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\frac{\mathbf{Q} {\mathbf{K}}^\top}{\sqrt{d_k}})\mathbf{V}
$$
对某个查询和键向量 $$\mathbf{q}_i, \mathbf{k}_j \in \mathbb{R}^d$$ 来说，可以得到一个标量得分：
$$
a_{ij} = \text{softmax}(\frac{\mathbf{q}_i {\mathbf{k}_j}^\top}{\sqrt{d_k}})
= \frac{\exp(\mathbf{q}_i {\mathbf{k}_j}^\top)}{ \sqrt{d_k} \sum_{r \in S_i} \exp(\mathbf{q}_i {\mathbf{k}_r}^\top) }
$$
其中 $$S_i$$ 是第 $$i$$ 个查询要关照的键位集合。

> 更多注意力机制介绍可以看我的[另一篇文章](https://libertydream.github.io/2020/04/26/Attention-综述/)

# 多头自注意力

_多头自注意力单元_ 是 Transformer 的一个关键部件。相较于只计算一次注意力，多头机制将输入分成小块然后在各个子空间并行计算比例点积注意力。各个注意力结果简单拼接并线性转换为期望维度。
$$
\begin{aligned}
\text{MultiHeadAttention}(\mathbf{X}_q, \mathbf{X}_k, \mathbf{X}_v) &= [\text{head}_1; \dots; \text{head}_h] \mathbf{W}^o \\ 
\text{where head}_i &= \text{Attention}(\mathbf{X}_q\mathbf{W}^q_i, \mathbf{X}_k\mathbf{W}^k_i, \mathbf{X}_v\mathbf{W}^v_i)
\end{aligned}
$$
其中 $$[.;.]$$ 是拼接操作，权重矩阵 $$\mathbf{W}^q_i, \mathbf{W}^k_i \in \mathbb{R}^{d \times d_k/h}, \mathbf{W}^v_i \in \mathbb{R}^{d \times d_v/h}$$ 将大小为  $$L \times d$$ 的输入嵌入阵映射为查询，键和值矩阵。至于 $$\mathbf{W}^o \in \mathbb{R}^{d_v \times d}$$ 则是线性转换结果。所有权重在训练时一起进行学习。

<img src="https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_multi-head-attention.png" style="zoom:33%;" />

# Transformer

**Transformer**（这里指代 vanilla Transformer，以和其他强化版做区分; [Vaswani 等, 2017](https://arxiv.org/abs/1706.03762)）模型采用编码器-解码器架构，这也是多数 [NMT](https://libertydream.github.io/2020/04/26/Attention-综述#为翻译而生)  模型所采用的模式。后来只采用编码器/解码器的 Transformer 在语言模型任务中取得了亮眼表现，比如 [BERT](https://libertydream.github.io/2019/11/16/NLP-迁移学习演化之路/) 和 [GPT-2](https://libertydream.github.io/2019/11/23/图解GPT-2/)。

### 编码器-解码器架构

**编码器**可以生成基于注意力的表示，能从宽泛的上下文中定位特定信息片段。编码器由 6 个独立单元堆叠而成，每个单元又含两个子单元——一个 _多头注意力_ 层和一个 _逐点_ 全连接前馈网络。所谓“逐点”，就是说对序列中的每个元素都施加相同的线性变换（权重也相同）。这也可以看作是核大小为 1 的卷积层。每个子单元都以残差连接并经由层归一化处理。此外，所有子单元输出的都是相同的 $$d$$ 维数据。

**解码器**的功用在于从编码过的表示中检索信息，其整体架构很像编码器，只是每个独立重复单元中有两个多头注意力子单元而非一个。第一个多头注意力子单元是“盖了” _遮罩（masked）_的，防止将来要处理的后续位置信息对当下计算产生干扰。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_transformer.png)

### 位置编码

因为自注意运算是排列不变的，所以用适当的**位置编码**给模型注入 _顺序信息_ 十分重要。位置编码 $$\mathbf{P} \in \mathbb{R}^{L \times d}$$ 有着和输入嵌入相同的维数，所以可以直接加在输入上。vanilla Transformer 考虑了两种编码方式：

（1）_正弦位置编码_ 定义如下，给定标识位置 $$i=1,\dots,L$$ 和维度 $$\delta=1,\dots,d$$：
$$
\text{PE}(i,\delta) = 
\begin{cases}
\sin(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta'\\
\cos(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' + 1\\
\end{cases}
$$
这样位置编码的各个维度对应于不同维数下的正弦波长，从  $$2\pi$$ 到 $$10000 \cdot 2\pi$$。下图展示了 $$L=32$$，$$d=128$$ 的正弦位置编码，取值介于 -1（黑）和 1（白）之间，值为 0 时为灰色

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_sinoidual-positional-encoding.png)

（2）_习得位置编码_，意同其名，对各元素的 _绝对_ 位置学习一个向量出来并分配给各元素作为位置编码结果（[Gehring, 等 2017](https://arxiv.org/abs/1705.03122)）

### 后继

紧跟着 vanilla Transformer，[Al-Rfou 等 (2018)](https://arxiv.org/abs/1808.04444) 加上了一系列辅助损失，从而能在字符层面训练深度 Transformer 语言模型，其表现优于一众 LSTM 模型。该模型主要添加了几类辅助任务：

- 相较于只在序列尾端指生成一个预测，每个_即时位置_ 也被要求做出恰当预测，给定更小的上下文环境强迫模型预估（比如上下文窗口开始处的第一对标识）
- Transformer 的各个中间层也要进行预测。训练过程中层次越低参与权重越小，带来的总体损失也越小
- 序列中的每个位置都能预测多个目标，也就是对两个或以上的后续标识进行预测

下图展示了这几种辅助任务

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_transformer-aux-losses.png)

# 自适应耗时

**自适应耗时（Adaptive Computation Time, ACT; [Graves, 2016](https://arxiv.org/abs/1603.08983)）**机制能动态决定循环神经网络需要进行多少步计算。_个人曾在早先[文章](https://libertydream.github.io/2019/10/05/注意力和增强循环神经网络/#自适应计算耗时)中介绍过 ACT_

比方说，现有 RNN 模型 $$\mathcal{R}$$ ，由输入权重 $$W_x$$，参数化的状态转移函数 $$\mathcal{S}(.)$$，一系列输出权重 $$W_y$$ 以及输出偏置 $$b_y$$ 组成。如果输入序列为 $$(x_1, \dots, x_L)$$，输出序列 $$(y_1, \dots, y_L)$$ 计算方式为：


$$
s_t = \mathcal{S}(s_{t-1}, W_x x_t), \quad y_t = W_y s_t + b_y\quad\text{for }t=1, \dots, L
$$


ACT 能让上述 RNN 结构对每个输入元素进行多步计算，至于具体数量是可变、动态的。多步计算会生成一系列中间状态 $$(s_t^1, \dots, s_t^{N(t)})$$ 和中间结果 $$(y_t^1, \dots, y_t^{N(t)})$$ ——他们共享同样的状态转移函数 $$\mathcal{S}(.)$$，输出权重 $$W_y$$ 和偏置 $$b_y$$ ：


$$
\begin{aligned}
s_t^0 &= s_{t-1} \\
s_t^n &= \mathcal{S}(s_{t}^{n-1}, x_t^n) = \mathcal{S}(s_{t}^{n-1}, x_t + \delta_{n,1}) \text{ for } n=1, \dots, N(t)\\
y_t^n &= W_y s_t^n + b_y
\end{aligned}
$$


其中 $$\delta_{n,1}$$ 是二值信号量，表明是否加上了某步计算

计算步骤数 $$N(t)$$ 由外部 sigmoid 型停止单元 $$h$$ 决定。单元计算要结合权重矩阵 $$W_h$$ 和偏置 $$b_h$$，对当前 $$n$$ 步的第 $$t$$ 个输入元素给出一个停止概率 $$p_t^n$$ ：


$$
h_t^n = \sigma(W_h s_t^n + b_h)
$$


为保证可以计算一步就停止，ACT 引入小常数 $$\epsilon$$ (比如 0.01) ，只要累计概率超过 $$1-\epsilon$$，计算就会停止：


$$
\begin{aligned}
N(t) &= \min(\min\{n': \sum_{n=1}^{n'} h_t^n \geqslant 1 -\epsilon\}, M) \\
p_t^n &= \begin{cases}
h_t^n & \text{if }n < N(t) \\
R(t) = 1 - \sum_{n=1}^{N(t)-1} h_t^n & \text{if }n= N(t)\\
\end{cases}
\end{aligned}
$$


其中 $$M$$ 是所允许的中间步骤数的上界，最终状态和输出结果是中间结果的加权和：


$$
s_t = \sum_{n=1}^{N(t)} p_t^n s_t^n,\quad y_t = \sum_{n=1}^{N(t)} p_t^n y_t^n
$$


![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_ACT-computation-graph.png)

为了避免对各个输入进行不必要的过度思考，ACT 在损失函数里加上了 _思考损失（ponder cost）_$$\mathcal{P}(x) = \sum_{t=1}^L N(t) + R(t) $$ ，这样模型就会倾向于减少中间计算步骤的数量

# 提升注意力跨度

之所以要改善注意力跨度，目的是使自注意的上下文范围更广，用起来更高效，更灵活。

## 更长跨度

vanilla Transformer 的注意力跨度是固定且有限的。每次模型都是在处理相同片段中的其余元素，同时各个定长片段间的信息也不能交流。

这种 _上下文片段_ 导致以下一些问题：

- 模型学习不到间距特别长的词语间的依赖关系
- 每段里的头几个标识因为上下文内容很少甚至没有，很难进行预测
- 计算代价高昂。每当片段右移一格，新片段就要从头重新分析，尽管其中有很多重复交叠的标识

**Transformer-XL** ([Dai 等, 2019](https://arxiv.org/abs/1901.02860); “XL” 意指“超长”) 通过两点改进解决了上下文片段问题：

1. 复用片段间的隐态
2. 为方便状态复用，采用新的位置编码方法

### 隐态复用

通过接连不断的使用前一片段的隐态，模型在片段间构建了循环连接关系。下图对比了片段长度为 4 时 vanilla Transformer 和 Transformer-XL 的训练状态

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_transformer-XL-training.png)

记模型中第 $$n$$ 层的第 $$(\tau + 1)$$ 个片段为 $$\mathbf{h}_{\tau+1}^{(n)} \in \mathbb{R}^{L \times d}$$，相同片段的上一层隐态为  $$\mathbf{h}_{\tau+1}^{(n-1)}$$，以及同层的前一个片段为 $$\mathbf{h}_{\tau}^{(n)}$$。在整合了先前隐态中的信息后，模型可以将注意力投放到很“久远”的地方，也不再受到单个片段的束缚。


$$
\begin{aligned}
\color{red}{\widetilde{\mathbf{h}}_{\tau+1}^{(n-1)}} &= [\text{stop-gradient}(\mathbf{h}_{\tau}^{(n-1)}) \circ \mathbf{h}_{\tau+1}^{(n-1)}] \\
\mathbf{Q}_{\tau+1}^{(n)} &= \mathbf{h}_{\tau+1}^{(n-1)}\mathbf{W}^q \\
\mathbf{K}_{\tau+1}^{(n)} &= \color{red}{\widetilde{\mathbf{h}}_{\tau+1}^{(n-1)}} \mathbf{W}^k \\
\mathbf{V}_{\tau+1}^{(n)} &= \color{red}{\widetilde{\mathbf{h}}_{\tau+1}^{(n-1)}} \mathbf{W}^v \\
\mathbf{h}_{\tau+1}^{(n)} &= \text{transformer-layer}(\mathbf{Q}_{\tau+1}^{(n)}, \mathbf{K}_{\tau+1}^{(n)}, \mathbf{V}_{\tau+1}^{(n)})
\end{aligned}
$$


无论是键还是值都依赖于拓展的隐态，只有查询向量只依靠当前时步的隐态。联结运算 $$[. \circ .]$$ 是建立在序列长度之上的。

### 相对位置编码

要适配这种新型注意力跨度，Transformer-XL 提出了新型的位置编码方式。如果依旧使用和 vanilla Transformer 一样的方法对绝对位置编码，先前片段和当前片段的编码就没差别了，这不是我们想要的。

为了使位置编码随片段移动而变化，Transformer-XL 选择对 _相对位置_ 进行编码，毕竟要做出好预测知道相对偏移量就够了，也就是键向量 $$\mathbf{k}_{\tau, j}$$ 和其查询向量 $$\mathbf{q}_{\tau, i}$$ 间的跨度 $$i -j$$。

如果只看位置编码，忽略掉比例系数  $$\frac {1}{\sqrt{d_k}}$$  和 softmax 中的归一化项，可以将 $$i$$ 处查询和 $$j$$ 处键间的注意力得分记为：


$$
\begin{aligned}
a_{ij} 
&= \mathbf{q}_i {\mathbf{k}_j}^\top = (\mathbf{x}_i + \mathbf{p}_i)\mathbf{W}^q ((\mathbf{x}_j + \mathbf{p}_j)\mathbf{W}^k)^\top \\
&= \mathbf{x}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{x}_j^\top + \mathbf{x}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{p}_j^\top + \mathbf{p}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{x}_j^\top + \mathbf{p}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{p}_j^\top
\end{aligned}
$$


Transformer-XL 将上述式子重新参数化为：


$$
a_{ij}^\text{rel} = 
\underbrace{ \mathbf{x}_i\mathbf{W}^q \color{blue}{ {\mathbf{W}_E^k}^\top } \mathbf{x}_j^\top }_\text{内容索引} + 
\underbrace{ \mathbf{x}_i\mathbf{W}^q \color{blue}{ {\mathbf{W}_R^k}^\top } \color{green}{\mathbf{r}_{i-j}^\top} }_\text{内容位置偏置} + 
\underbrace{ \color{red}{\mathbf{u}} \color{blue}{ {\mathbf{W}_E^k}^\top } \mathbf{x}_j^\top }_\text{全局内容偏置} + 
\underbrace{ \color{red}{\mathbf{v}} \color{blue}{ {\mathbf{W}_R^k}^\top } \color{green}{\mathbf{r}_{i-j}^\top} }_\text{全局位置偏置}
$$


- 用相对位置编码  $$\mathbf{r}_{i-j} \in \mathbf{R}^{d}$$ 替代 $$\mathbf{p}_j$$ 
- 分别用两个要训练的参数 $$\mathbf{u}$$ （代表内容）和 $$\mathbf{v}$$ （代表位置）代替两项中的 $$\mathbf{p}_i\mathbf{W}^q$$ 
- 键权重矩阵 $$\mathbf{W}^k$$ 拆分为指代内容信息的 $$\mathbf{W}^k_E$$ 以及指代位置信息的 $$\mathbf{W}^k_R$$ 两部分

## 自适应跨度

Transformer 一大核心优势在于对长文依赖的学习能力。视环境而定，模型有时可能会给远处内容更多关照，或者每个注意力头都有各自的注意力模式。在需要的时候，如果能灵活调整注意力跨度且更关注远处的内容，就能减少计算量和记忆开销，进而扩大模型所能支持的最大上下文范围。

这也正是**自适应注意力跨度（Adaptive Attention Span）**的目标。[Sukhbaatar 等, 2019](https://arxiv.org/abs/1905.07799) 提出了一种自注意机制，谋求对注意力跨度进行优化。该团队假设不同注意力头会对相同上下文窗口打不同的分数（见下图），所以要优化跨度就该每个头独自训练。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_attention-per-head.png)

_A 和 B 两个注意力头给相同上下文窗口打出了不同注意力分值，A 头更关注近期标识，而 B 头会统一回看过往内容_

对于第 $$i$$ 个标识，计算其与 $$j \in S_i$$ 处的其他键间的注意力权重，这里 $$S_i$$ 是 $$i$$ 处标识的上下文窗口。


$$
\begin{aligned}
e_{ij} &= \mathbf{q}_i {\mathbf{k}_j}^\top \\ 
a_{ij} &= \text{softmax}(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{r=i-s}^{i-1} \exp(e_{ir})} \\
\mathbf{y}_i &= \sum_{r=i-s}^{i-1}a_{ir}\mathbf{v}_r = \sum_{r=i-s}^{i-1}a_{ir}\mathbf{x}_r\mathbf{W}^v
\end{aligned}
$$


_柔性遮罩函数（soft mask function）_ $$m_z$$ 将查询和键向量间的距离映射为 $$[0, 1]$$ 区间内的某个值，从而有效调节注意力跨度。$$m_z$$ 经 $$z \in [0, s]$$  参数化，$$z$$ 要通过学习获得：


$$
m_z(x) = \text{clamp}(\frac{1}{R}(R+z-x), 0, 1)
$$


其中 $$R$$ 是用于指定 $$m_z$$ 柔性的超参数

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_soft-masking-function.png)

求解注意力权重时，柔性遮罩函数会参与到对元素进行 softmax 变换的过程当中。


$$
a_{ij} = \frac{m_z(i-j)\exp(s_{ij})}{\sum_{r=i-s}^{i-1}m_z(i-r) \exp(s_{ir})}
$$


上式中的 $$z$$ 是可微的，故将其与模型其他部分一同联合训练。每个注意力头的参数 $$z^{(i)}, i=1, \dots, h$$  独立训练，同时还要在损失函数中加上对 $$\sum_{i=1}^h z^{(i)}$$ 的 $$L_1$$ 惩罚。

如果是采用[自适应耗时](#自适应耗时)，该方法能进一步增强注意力跨度的灵活性，根据当前输入动态变化。注意力头在 $$t$$ 时刻的跨度参数 $$z_t$$ 是一个 sigmoid 型函数，$$z_t = S \sigma(\mathbf{v} \cdot \mathbf{x}_t +b)$$，其中向量 $$\mathbf{v}$$ 和偏置 $$b$$ 与其他参数一起联合训练。

经过对带有自适应注意力跨度的 Transformer 的系列实验， [Sukhbaatar 等 (2019)](https://arxiv.org/abs/1905.07799) 发现了一些通行趋势。较低层次并不需要很长的注意力跨度，而对高层的一些注意力头可能需要非常长的跨度。此外，自适应跨度能大幅减少 FLOPS（浮点运算次数/秒） 数量，对那些有着许多注意力层和大范围上下文的模型尤为明显。

## 小范围跨度

 Transformer 一开始同时也是最流行的用途是做语言模型，一维文本序列按时间顺序排好，此时注意力跨度随着上下文范围增大而线性增长。

但如果想将 Transformer 用在图像上，就得先明白此时的上下文范围或次序的定义方式。 **Image Transformer** ([Parmer 等 2018](https://arxiv.org/abs/1802.05751))  在 Transformer 框架下找到了一种类似于序列模型的图像生成方式，同时将自注意范围局限于 _局部_ 近邻之上，从而使模型能并行处理更多图像并保证似然损失易于处理。

图像生成情景下编码器-解码器框架得以保留：

- 编码器会基于原图生成上下文相关的单像素通道表示
- 解码器则会 _自回归_ 生成输出图像，每个时步每个像素一个通道

定义当前要生成的像素表示为查询 $$\mathbf{q}$$ 。键向量  $$\mathbf{k}_1, \mathbf{k}_2, \dots$$ 代表其他位置的表征，用于计算 $$\mathbf{q}$$ 。它们共同组成记忆矩阵  $$\mathbf{M}$$。 $$\mathbf{M}$$ 的范围决定了像素查询 $$\mathbf{q}$$ 的上下文窗口大小。

Image Transformer 一共有两种小范围 $$\mathbf{M}$$ ，如下图所示

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_image-transformer-attention.png)

_图中展示了视觉输入的 1D 和 2D 注意力跨度，黑线标出了查询块，青绿色的轮廓则是像素 q 的实际注意力范围_

（1）_1D 局部注意力_：输入图像扁平化为[光栅扫描](https://en.wikipedia.org/wiki/Raster_scan#Scanning_pattern)序列，即从左到右，从上而下扫描。接着线性化图形被分为若干不重叠的查询块。上下文窗口由同处查询块 $$\mathbf{q}$$ 中的像素以及固定数量的之前生成的若干像素组成

（2）_2D 局部注意力_：图像分成多个不重叠的矩形查询块，待查像素能看到相同记忆块中的其他像素。为了使左上角的像素同样有合理的上下文窗口，记忆块分别向上，向左和向右进行定量拓展

# 降低时间与记忆开销

这一部分介绍的几种 Transformer 的改进追求的是降低耗时，减少记忆开销

## 稀疏注意力矩阵分解

vanilla Transformer 的计算和记忆开销与序列长度二次方成正比，所以很难用到超长序列上。

**Sparse Transformer** ([Child 等, 2019](https://arxiv.org/abs/1904.10509)) 提出 _因子分解自注意_ 模型，通过对稀疏矩阵进行因子分解使得在长达 16,384 的序列上训练百层稠密注意力网络成为可能，一般而言这对当代硬件设备来说是不可能的任务。

考虑系列连接模式 $$\mathcal{S} = \{S_1, \dots, S_n\}$$，其中 $$S_i$$ 记录了第 $$i$$ 个查询向量关照的一组键位。


$$
\begin{aligned}
\text{Attend}(\mathbf{X}, \mathcal{S}) &= \Big( a(\mathbf{x}_i, S_i) \Big)_{i \in \{1, \dots, L\}} \\
\text{ where } a(\mathbf{x}_i, S_i) &= \text{softmax}\Big(\frac{(\mathbf{x}_i \mathbf{W}^q)(\mathbf{x}_j \mathbf{W}^k)_{j \in S_i}^\top}{\sqrt{d_k}}\Big) (\mathbf{x}_j \mathbf{W}^v)_{j \in S_i}
\end{aligned}
$$


注意 $$S_i$$ 的大小不固定，$$a(\mathbf{x}_i, S_i)$$ 大小恒定为 $$d_v$$ ，进而 $$\text{Attend}(\mathbf{X}, \mathcal{S}) \in \mathbb{R}^{L \times d_v}$$。

自回归模型中，注意力跨度被定义为 $$S_i = \{j: j \leqslant i\}$$ ，即每个标识会同时关照此前所有位置的内容。

而在因子分解自注意模型中，集合 $$S_i$$ 被拆分为依赖树，从而每个元素对 $$(i, j)，j \leqslant i$$ 都存在将 $$i$$ 连回 $$j$$ 的路径，而且 $$i$$ 可以直接或间接的关照到 $$j$$ 。

确切地讲，集合 $$S_i$$ 被分为 $$\mathbf{p}$$ 个不重叠子集，第 $$m$$ 个子集记为 $$A^{(m)}_i \subset S_i, m = 1,\dots, p$$ 所以输出位置 $$i$$ 和任意 $$j$$ 间最大距离为 $$p+1$$。比如，如果 $$(j,a,b,c,\dots,i)$$ 是 $$i$$ 和$$j$$ 间的索引，有 $$j \in A_a^{(1)}, a \in A_b^{(2)}, b \in A_c^{(3)}, \dots$$ 以此类推

### 稀疏注意力因子分解

Sparse Transformer 提出了两种类型的因子分解注意力。下面以 2D 图像为例，

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_sparse-attention.png)

_第一行展示了 (a) Transformer，(b) 跨位注意力 Sparse Transformer 和 (c) 固定注意力 Sparse Transformer 三种形态下的注意力连接模式。第二行是相应自注意连接矩阵，要强调的是上下两行尺度并不相同_

（1）_跨位_ 注意力一般带有步幅 $$\ell \sim \sqrt{n}$$。当图像数据的结构以步幅为基准对齐时，这种注意力效果良好。每个像素会以光栅扫描顺序（自然覆盖了整个图像宽度）关照此前 $$\ell$$ 个像素点，接着这些像素点会关照同一列的其他像素（以另一种注意力连接方式）。


$$
\begin{aligned}
A_i^{(1)} &= \{ t, t+1, \dots, i\} \text{, where } t = \max(0, i - \ell) \\
A_i^{(2)} &= \{j: (i-j) \mod \ell = 0\}
\end{aligned}
$$


（2）_固定_ 注意力。一小批标识总结了之前位置的信息并向所有之后的位置广播相关信息。


$$
\begin{aligned}
A_i^{(1)} &= \{j: \lfloor \frac{j}{\ell} \rfloor = \lfloor \frac{i}{\ell} \rfloor \} \\
A_i^{(2)} &= \{j: j \mod \ell \in \{\ell-c, \dots, \ell-1\} \}
\end{aligned}
$$


其中 $$c$$ 是超参数，如果 $$c=1$$ 则限制表示，而许多表示依赖于少数位置。论文对  $$\ell \in \{ 128, 256 \}$$ 选择了 $$c\in \{ 8, 16, 32 \}$$

### Transformer 里的因子分解自注意力

有三种方式将稀疏因子分解注意力用到 Transformer 体系中：

1. 每个残差块一种注意力然后穿插交织。<br/> $$\text{attention}(\mathbf{X}) = \text{Attend}(\mathbf{X}, A^{(n \mod p)}) \mathbf{W}^o$$ ，其中当前残差块索引为 $$n$$
2. 设置一个所有因子分解头都要关照的头。<br/>$$\text{attention}(\mathbf{X}) = \text{Attend}(\mathbf{X}, \cup_{m=1}^p A^{(m)}) \mathbf{W}^o $$。
3. 使用多头注意力机制，但与 vanilla Transformer 不同的是，每个头可能会采用上述模式之一，1 或 2 => 这么选通常效果最好

Sparse Transformer 还提出了一系列变革从而能训练上百层 Transformer，包括梯度检查点，反向传播时重计算注意力与 FF 层，混合精度训练，高效实现块稀疏操作等等。更多相关内容请看[论文](https://arxiv.org/abs/1904.10509)

## 局部感知哈希

**Reformer** 模型（[Kitaev 等 2020](https://arxiv.org/abs/2001.04451)）针对下列 Transformer 痛点进行改进：

- $$N$$ 层模型所需记忆空间比单层模型大 $$N$$ 倍，因为要存储的反向传播激活值数量与层数正相关
- 中间的 FF 层经常很大
- 长为 $$L$$ 的序列所对应的注意力矩阵往往需要 $$O(L^2)$$ 的记忆和时间开销

Reformer 主要有两点改进：

1. 将点积注意力替换为 _局部感知哈希（locality-sensitive hashing，LSH）_注意力，将复杂度从 $$O(L^2)$$ 降为 $$O(L\log L)$$.
2. 将标准残差块替换为 _可逆残差层（reversible residual layers）_，这样不再需要存储 $$N$$ 次激活值（即与层数正比），训练时存一次即可

### 局部感知哈希注意力

[注意力公式](#注意力与自注意)里的 $$\mathbf{Q} \mathbf{K}^\top$$ 部分中，我们实际关心的只有那个最大值，因为最大元素 softmax 之后贡献也最多。对每个查询 $$\mathbf{q}_i \in \mathbf{Q}$$，要找到距离 $$\mathbf{q}_i$$ 最近的行向量 $$\mathbf{K}$$，而为了在高维空间尽快的找到最近邻，Reformer 将[局部感知哈希](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)引入了注意力机制。

如果映射会保存数据点间的距离信息，那我们就说哈希映射 $$x \mapsto h(x)$$ 是 _局部感知_ 的，这样相近的向量哈希值相似，而相距较远的向量哈希值则差异较大。Reformer 采用的就是这样一种方案，给定固定随机矩阵 $$\mathbf{R} \in \mathbb{R}^{d \times b/2}$$（$$b$$ 是超参数），哈希函数为 $$h(x) = \arg\max([xR; −xR])$$。下图展示了局部感知注意力

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_LSH-attention-matrix.png)

在 LSH 中，查询向量只需关照同一哈希桶中的位置，$$S_i = \{j: h(\mathbf{q}_i) = h(\mathbf{k}_j)\}$$。如图所示，其执行流程如下：

- (a) 全关照注意力矩阵通常是稀疏的
- (b) 借助 LSH，可以对键与查询排序，根据各自哈希分桶对齐
- (c) 令 $$\mathbf{Q} = \mathbf{K}$$（确切的说是 $$\mathbf{k}_j = \mathbf{q}_j / \|\mathbf{q}_j\|$$），这样每个桶里就有等量的键和查询向量了。有意思的地方在于，这种 “共享 QK” 的配置并没有影响到 Transformer 的表现
- (d)  进行批处理，连续的 $$m$$ 块查询被组织在了一起。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_LSH-attention.png)

### 可逆残差网络

Reformer 另一大改进就是 _可逆残差层（[Gomez 等 2017](https://arxiv.org/abs/1707.04585)）_ 的使用了。可逆残差网络的创作动机在于设计一种架构，在仅使用模型参数的情况下，能以某种方式使任意层的激活值能从后续层的激活值中恢复出来。这样就能在反向传播的时候重新计算出激活值而不用将其全部存起来，从而降低记忆开销。

对于任一层 $$x \mapsto y$$，一般残差层是 $$y = x + F(x)$$，而可逆层的做法是把输入和输出分成结对形式 $$(x_1, x_2) \mapsto (y_1, y_2)$$ ，并按下列方式计算：


$$
y_1 = x_1 + F(x_2),\; y_2 = x_2 + G(y_1)
$$


取反很简单：


$$
x_2 = y_2 - G(y_1), \; x_1 = y_1 − F(x_2)
$$


Reformer 借鉴了这一思想，在一个可逆网络块中将注意力（$$F$$）和前馈层（$$G$$）结合了起来：


$$
Y_1 = X_1 + \text{Attention}(X_2), \; Y_2 = X_2 + \text{FeedForward}(Y_1)
$$


如果将前馈计算分块进行，记忆开销可以进一步减少：


$$
Y_2 = [Y_2^{(1)}; \dots; Y_2^{(c)}] = [X_2^{(1)} + \text{FeedForward}(Y_1^{(1)}); \dots; X_2^{(c)} + \text{FeedForward}(Y_1^{(c)})]
$$


最终可逆 Transformer 不需要存任意层的激活值。

# 循环起来

**通用 Transformer（Universal Transformer，[Dehghani 等. 2019](https://arxiv.org/abs/1807.03819)）**将 Transfomer 的自注意和 RNN 的循环机制结合了起来，期望既能享受到 Transformer 的长文全局感知的好处，又能受益于 RNN 的习得性归纳偏差。

相较于固定层数，通用 Transformer 采用自适应耗时动态调整步骤数量。如果固定了步数，通用 Transformer 等价于层间参数共享的多层 Transformer。

站高一层看，通用 Transformer 可以被视为一种循环函数，对每个标识学习相应隐态表示。循环函数在标识间并行演化，同时经由自注意力来共享不同位置间的信息。下图展示了通用 Transformer 是怎样不断并行提炼各处的一系列隐态表示的。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_universal-transformer-loop.png)

假定输入序列长为 $$L$$， 第 $$t$$ 步通用 Transformer 会以可变步数迭代更新表征 $$\mathbf{H}^t \in \mathbb{R}^{L \times d}$$。开始时的第 0 步，  $$\mathbf{H}^0$$ 和输入嵌入矩阵相同。多头自注意机制下所有位置并行计算，然后经循环转换函数处理。


$$
\begin{aligned}
\mathbf{A}^t &= \text{LayerNorm}(\mathbf{H}^{t-1} + \text{MultiHeadAttention}(\mathbf{H}^{t-1} + \mathbf{P}^t) \\
\mathbf{H}^t &= \text{LayerNorm}(\mathbf{A}^{t-1} + \text{Transition}(\mathbf{A}^t))
\end{aligned}
$$


其中转换函数 $$\text{Transition}(.)$$ 可以是[可分离卷积](https://arxiv.org/abs/1610.02357)，亦或是由两个位置层面（也就是单独处理 $$\mathbf{A}^t$$ 的每一行）的仿射变换 + 一层 ReLU 构成的的全连接神经网络

位置编码 $$\mathbf{P}^t$$ 使用正弦位置信号，只是额外加上了时间维度：


$$
\text{PE}(i, t, \delta) = 
\begin{cases}
\sin(\frac{i}{10000^{2\delta'/d}}) \oplus \sin(\frac{t}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta'\\
\cos(\frac{i}{10000^{2\delta'/d}}) \oplus \cos(\frac{t}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' + 1\\
\end{cases}
$$


![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_universal-transformer.png)

上图简单展示了通用 Transformer 构造，编码器和解码器的基础循环结构相同，解码器还多了对编码器最终表征 $$\mathbf{H}^T$$的处理。

对于自适应版的通用 Transformer，循环步数 $$T$$ 由 [ACT](#自适应耗时) 动态决定。每个位置都有动态 ACT 停止机制，一旦某个标识循环块挂起，它会终止下面的循环更新，而只是简单拷贝当前值到下一步直到所有块都停止，或者模型达到最大步数限制。

# RL 的稳定化

自注意机制避免了用固定大小的隐态表示全部过往信息，而且不会像 RNNs 那样遭遇梯度消融或爆炸问题。强化学习任务无疑可以从这些特性中获益，但是即使是监督学习下训练 Transformer 都很难，更别提 RL 情景了。毕竟自己训练一个 LSTM 代理并保持稳定是很有挑战的。

**Gated Transformer-XL** (**GTrXL**; [Parisotto 等 2019](https://arxiv.org/abs/1910.06764)) 是 RL 试水 Transformer 的一个案例。GTrXL 在 [Transformer-XL](#更长跨度) 上进行了两点改进成功做到了训练稳定：

1. 层归一化只用在残差模块的输入流上，捷径流上不用。这样做主要是为了能让原始输入从头传到尾
2. 残差连接替换为 GRU 风格（门控循环单元，Gated Recurrent Unit; [Chung et al., 2014](https://arxiv.org/abs/1412.3555)）的 _门机制_ 。

$$
\begin{aligned}
r &= \sigma(W_r^{(l)} y + U_r^{(l)} x) \\
z &= \sigma(W_z^{(l)} y + U_z^{(l)} x - b_g^{(l)}) \\
\hat{h} &= \tanh(W_g^{(l)} y + U_g^{(l)} (r \odot x)) \\
g^{(l)}(x, y) &= (1-z)\odot x + z\odot \hat{h}
\end{aligned}
$$

门控函数的参数显式初始化为近似单位阵映射的形式——这也是为什么有 $$b_g$$ 项。有 $$b_g > 0$$ 的话对学习加速大有裨益

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_gated-transformer-XL.png)

上图对 Transformer-XL, 层归一 Transformer-XL 和门控 Transformer-XL 进行了对比。