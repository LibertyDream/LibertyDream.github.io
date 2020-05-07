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
excerpt:    各式强化版 Transformer 模型已是层出不穷，而本文力图展示朴实无华的 Transformer 是怎么改进用于超长注意力跨度，降低记忆和计算消耗，应用于强化学习任务等等
---

> 编译自：The Transformer Family，[Lilian Weng](https://lilianweng.github.io/lil-log/)

# 符号说明

| 符号                                                         | 含义                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $$d$$                                                        | 模型大小 / 隐态维度 / 位置编码尺寸                           |
| $$h$$                                                        | 多头注意力层中头的数量                                       |
| $$L$$                                                        | 输入序列片段长度                                             |
| $$\mathbf{X} \in \mathbb{R}^{L \times d}$$                   | 输入序列，其中每个元素都已映射为 $$d$$ 维嵌入向量，和模型尺寸相当 |
| $$\mathbf{W}^k \in \mathbb{R}^{d \times d_k}$$               | The key weight matrix.                                       |
| $$\mathbf{W}^q \in \mathbb{R}^{d \times d_k}$$               | The query weight matrix.                                     |
| $$\mathbf{W}^v \in \mathbb{R}^{d \times d_v}$$               | 值权重矩阵，一般有 $$d_k = d_v = d$$                         |
| $$\mathbf{W}^k_i, \mathbf{W}^q_i \in \mathbb{R}^{d \times d_k/h}; \mathbf{W}^v_i \in \mathbb{R}^{d \times d_v/h}$$ | 每个头对应的权重矩阵                                         |
| $$\mathbf{W}^o \in \mathbb{R}^{d_v \times d}$$               | 输出权重矩阵                                                 |
| $$\mathbf{Q} = \mathbf{X}\mathbf{W}^q \in \mathbb{R}^{L \times d_k}$$ | 嵌入输入的查询                                               |
| $$\mathbf{K} = \mathbf{X}\mathbf{W}^k \in \mathbb{R}^{L \times d_k}$$ | 嵌入输入的键                                                 |
| $$\mathbf{V} = \mathbf{X}\mathbf{W}^v \in \mathbb{R}^{L \times d_v}$$ | 嵌入输入的值                                                 |
| $$S_i$$                                                      | 第 $$i$$ 个查询 $$\mathbf{q}_i$$ 要处理的键位置集合          |
| $$\mathbf{A} \in \mathbb{R}^{L \times L}$$                   | 输入序列长度为 $$L$$ 的自注意矩阵 $$\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})$$ |
| $$a_{ij}\in \mathbf{A}$$                                     | 查询 $$\mathbf{q}_i$$ 和键  $$\mathbf{k}_j$$ 间的注意力大小  |
| $$\mathbf{P} \in \mathbb{R}^{L\times d}$$                    | 位置编码矩阵，第 $$i$$ 行 $$\mathbf{p}_i$$ 是输入 $$\mathbf{x}_i$$ 的位置编码 |

# 注意力与自注意

_注意力_ 是神经网络的一种机制，拥有该机制的模型能学会有选择地处理给定数据集以作预测。投放的注意力大小取决于习得权重，因而模型输出结果通常是加权平均的形式。

_自注意_ 是注意力机制中的一种，模型通过观察样本中的其余部分来对该样本中的目标位置进行预测，直观感觉它很像是[非局部平均](https://en.wikipedia.org/wiki/Non-local_means)。同时留意自注意具有排列不变性，换句话说它是一种集合运算。

注意力/自注意的形式五花八门，Transformer ([Vaswani 等, 2017](https://arxiv.org/abs/1706.03762)) 采用的是 _比例点击注意力_ ：给定查询矩阵 $$\mathbf{Q}$$ ，键矩阵 $$\mathbf{K}$$ 和值矩阵 $$\mathbf{V}$$ ,输出结果就是值向量的加权和，而各个值分得的权重则由查询和键向量的点积结果决定：
$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\frac{\mathbf{Q} {\mathbf{K}}^\top}{\sqrt{d_k}})\mathbf{V}
$$
对某个查询和键向量 $$\mathbf{q}_i, \mathbf{k}_j \in \mathbb{R}^d$$ 来说，可以得到一个标量得分：
$$
a_{ij} = \text{softmax}(\frac{\mathbf{q}_i {\mathbf{k}_j}^\top}{\sqrt{d_k}})
= \frac{\exp(\mathbf{q}_i {\mathbf{k}_j}^\top)}{ \sqrt{d_k} \sum_{r \in S_i} \exp(\mathbf{q}_i {\mathbf{k}_r}^\top) }
$$
其中 $$S_i$$ 是第 $$i$$ 个查询要处理的键位置集合。

> 更多注意力机制介绍可以看我的[另一篇文章](https://libertydream.github.io/2020/04/26/Attention-综述/)

# 多头自注意力

_多头自注意力单元_ 是 Transformer 的一个关键部件。相较于只计算一次注意力，多头机制将输入分成小块然后在各个子空间并行计算比例点击注意力。各个注意力结果简单拼接并线性转换为期望维度。
$$
\begin{aligned}
\text{MultiHeadAttention}(\mathbf{X}_q, \mathbf{X}_k, \mathbf{X}_v) &= [\text{head}_1; \dots; \text{head}_h] \mathbf{W}^o \\ 
\text{where head}_i &= \text{Attention}(\mathbf{X}_q\mathbf{W}^q_i, \mathbf{X}_k\mathbf{W}^k_i, \mathbf{X}_v\mathbf{W}^v_i)
\end{aligned}
$$
其中 $$[.;.]$$ 是拼接操作，权重矩阵 $$\mathbf{W}^q_i, \mathbf{W}^k_i \in \mathbb{R}^{d \times d_k/h}, \mathbf{W}^v_i \in \mathbb{R}^{d \times d_v/h}$$ 将大小为  $$L \times d$$ 的输入嵌入映射为查询，键和值矩阵。至于 $$\mathbf{W}^o \in \mathbb{R}^{d_v \times d}$$ 则是线性转换结果。所有权重在训练时一起进行学习。

<img src="https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_multi-head-attention.png" style="zoom:33%;" />

# Transformer

**Transformer**（这里指代 vanilla Transformer，以和其他强化版做区分; [Vaswani 等, 2017](https://arxiv.org/abs/1706.03762)）模型采用编码器-解码器架构，这也是多数 [NMT](https://libertydream.github.io/2020/04/26/Attention-综述#为翻译而生)  模型所采用的模式。后来只采用编码器/解码器的 Transformer 在语言模型任务中取得了亮眼表现，比如 [BERT](https://libertydream.github.io/2019/11/16/NLP-迁移学习演化之路/) 和 [GPT-2](https://libertydream.github.io/2019/11/23/图解GPT-2/)

### 编码器-解码器架构

**编码器**可以生成基于注意力的表示，能从宽泛的上下文中定位特定信息片段。编码器由 6 个独立单元堆叠而成，每个单元又含两个子单元——一个 _多头注意力_ 层和一个 _逐点_ 全连接前馈网络。所谓“逐点”，就是说对序列中的每个元素都施加相同的线性变换（权重也相同）。这也可以看作是核大小为 1 的卷积层。每个子单元都是残差连接并要作层归一化处理。此外，所有子单元输出的都是相同的 $$d$$ 维数据。

**解码器**的功用在于从编码过的表示中检索信息，其整体架构很像编码器，只是每个独立重复单元中有两个多头注意力子单元而非一个。第一个多头注意力子单元是“盖了” _遮罩（masked）_的，防止将来要处理的后续位置信息的干扰。

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

紧跟着 vanilla Transformer，[Al-Rfou 等 (2018)](https://arxiv.org/abs/1808.04444) 加上了一系列辅助损失，从而能在字符层面训练深度 Transformer 语言模型，表现优于一众 LSTM 模型。模型主要用了几类辅助任务：

- 相较于只在序列尾端指生成一个预测，每个_即时位置_ 都要求做出正确预测，也就是给定更小的上下文环境强迫模型预估（比如上下文窗口开始处的第一对标识）
- 每个 Transformer 中间层也被用于做预测。训练过程中层次越低参与权重越小，带来的总体损失也越小
- 序列中的各位置能预测多个目标，即预测两个或多个未来标识

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_transformer-aux-losses.png)

# 自适应耗时（ACT）

**自适应耗时（Adaptive Computation Time, ACT; [Graves, 2016](https://arxiv.org/abs/1603.08983)）**机制能动态决定循环神经网络需要多少计算时步。_曾在早先[文章](https://libertydream.github.io/2019/10/05/注意力和增强循环神经网络/#自适应计算耗时)中介绍过 ACT_

比方说，现有 RNN 模型 $$\mathcal{R}$$ ，由输入权重 $$W_x$$，参数化的状态转移函数 $$\mathcal{S}(.)$$，一系列输出权重 $$W_y$$ 以及输出偏置 $$b_y$$ 组成。如果输入序列为 $$(x_1, \dots, x_L)$$，输出序列 $$(y_1, \dots, y_L)$$ 计算方式为：


$$
s_t = \mathcal{S}(s_{t-1}, W_x x_t), \quad y_t = W_y s_t + b_y\quad\text{for }t=1, \dots, L
$$


ACT 能让上述 RNN 结构对每个输入元素进行多步计算，具体数量是可变的、动态的。多步计算导致有一系列中间状态 $$(s_t^1, \dots, s_t^{N(t)})$$ 和结果 $$(y_t^1, \dots, y_t^{N(t)})$$ 产生——他们共享同样的状态转移函数 $$\mathcal{S}(.)$$，输出权重 $$W_y$$ 和偏置 $$b_y$$ ：


$$
\begin{aligned}
s_t^0 &= s_{t-1} \\
s_t^n &= \mathcal{S}(s_{t}^{n-1}, x_t^n) = \mathcal{S}(s_{t}^{n-1}, x_t + \delta_{n,1}) \text{ for } n=1, \dots, N(t)\\
y_t^n &= W_y s_t^n + b_y
\end{aligned}
$$


其中 $$\delta_{n,1}$$ 是二值信号量，表明是否加上了某步计算

计算步骤数 $$N(t)$$ 由外部 s 型停止单元 $$h$$ 决定。单元计算要结合权重矩阵 $$W_h$$ 和偏置 $$b_h$$，对当前 $$n$$ 步的第 $$t$$ 个输入元素给出一个停止概率 $$p_t^n$$ ：


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


其中 $$M$$ 是所允许的中间步骤数的上界，最终状态和输出结果是中间产物的加权和：


$$
s_t = \sum_{n=1}^{N(t)} p_t^n s_t^n,\quad y_t = \sum_{n=1}^{N(t)} p_t^n y_t^n
$$


![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-30_ACT-computation-graph.png)

为了避免对各个输入进行不必要的过度思考，ACT 在损失函数里加上了 _思考损失（ponder cost）_$$\mathcal{P}(x) = \sum_{t=1}^L N(t) + R(t) $$ ，这样模型就会倾向于减少中间计算步骤的数量

# 改善注意力跨度

之所以要改变注意力跨度，目的是使自注意的上下文范围更广，更高效，更灵活。

## 更长跨度（Transformer-XL）

vanilla Transformer 的注意力跨度是固定且有限的。每次模型都是在处理相同片段中的其余元素，同时各个定长片段间的信息也不能交流。

这种 _上下文片段_ 造成了一些问题：

- 模型学习不到间距特别长的词语间的依赖关系
- 每段里的头几个标识因为上下文内容很少甚至没有，很难进行预测
- 计算代价高昂。每当片段右移一格，新片段就要从头重新分析，尽管其中有很多重复交叠的标识

**Transformer-XL** ([Dai 等, 2019](https://arxiv.org/abs/1901.02860); “XL” 意指“超长”) 通过两点改进解决了上下文片段问题：

1. 复用片段间的隐态
2. 为方便复用状态，采用新的位置编码方法

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

如果忽略掉除却位置编码外的比例系数  $$\frac {1}{\sqrt{d_k}}$$  和 softmax 中的归一化项，可以将 $$i$$ 处查询和 $$j$$ 处的键间的注意力得分记为：


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
- 键权重矩阵 $$\mathbf{W}^k$$ 拆为指代内容信息的 $$\mathbf{W}^k_E$$ 以及指代位置信息的 $$\mathbf{W}^k_R$$

## 自适应跨度

Transformer 一大核心优势在于对长文依赖的学习能力。视环境而定，模型有时可能会给远处内容更多关照，或者每个注意力头都有各自的注意力模式。如果在需要的时候，能灵活调整注意力间隔且只关注更远处的内容，就能减少计算量和记忆开销，从而扩大模型所能支持的最大上下文范围。

这也正是**自适应注意力跨度（Adaptive Attention Span，[Sukhbaatar 等, 2019](https://arxiv.org/abs/1905.07799)）**的目标。

