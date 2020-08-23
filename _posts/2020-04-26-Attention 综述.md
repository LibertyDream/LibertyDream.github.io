---
layout:		post
title:  	Attention 综述
subtitle:   一网打尽注意力
date:       2020-04-26
author:     一轩明月
header-img: img/post-bg-2015.jpg
catalog:    true
tags:
    - attention
    - Transformer
excerpt:    注意力无疑是近几年深度学习社区中十分流行的理念，同时也是一种很好用的工具。本文从注意力起源讲起，分门别类的介绍各种注意力机制与模型，比如 transformer 和 SNAIL，同时给出了相应的数学公式
---

> 编译自：Attention? Attention!  [Lilian Weng](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
>
> 关于注意力之前写过两篇短文，[NLP 中的注意力与记忆](https://libertydream.github.io/2019/10/26/深度学习和NLP中的注意力和记忆/)和[注意力与 RNN](https://libertydream.github.io/2019/10/05/注意力和增强循环神经网络/)，本文较二者而言更系统，数理逻辑也更清晰

注意力机制一定程度上是受到人类行为模式的启发，比如我们会移动视觉焦点查看图片的不同区域，也会留意一句话中的关联词。以秋田犬图像为例

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-21_shiba-example-attention.png)

人类的视觉机制保障我们能专注于某个“高分辨率”的特定区域（比如黄框里的尖耳朵）同时屏蔽掉图像周围”低分辨率“的景物（比如雪景与轮廓），然后调整焦点或是进行相关推断。给定一幅图像的局部一角，剩余像素也在提示着图像呈现内容。我们希望在黄框里看到尖耳朵因为已经看到了狗的鼻子，在右侧的另一个尖耳朵，还看到秋田犬那迷离的双眼（红框当中）。但底部的线衫和毛毯则并不会像这些狗的属性特征一样那么有帮助。

类似的，我们可以解释句中或邻近上下文内词语词间的关系。当我们看到“eating”，就希望马上能看到一个食物名词。下图中着色短语描述了食物，但没有直接“吃”。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-22_sentence-example-attention.png)

简而言之，深度学习中的注意力可以广义地理解一个衡量重要程度的向量：为了预测或推理某个元素，比如某个图像中的像素点或是一句话中的某个词，使用注意力向量来估算其与其他元素间的关联（可能在其他论文里叫“关照”）强度，并将元素值基于注意力向量加权求和作为目标取值的近似估计。

### Seq2Seq 模型的问题

序列到序列（**seq2seq**） 模型源自语言建模领域（[Sutskever, et al. 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)）。宽泛地讲，其目的在于将输入序列（源序列）转换为一个新序列（目标序列），两序列长度不定。转换任务的例子包括多语言间的机器翻译（文本/音频），问答对话的生成，甚至是将句子解析成语法树。

seq2seq 模型通常采取编码器-解码器（encoder-decoder）结构，主要包括：

- 有一个**编码器**来处理输入序列并将信息压入一个定长环境向量中（这也被称为句子嵌入或“思维”向量）。期望这种表示能对 _整个_ 源序列进行良好的概括。
- 使用环境向量初始化一个**解码器**来生成转换结果。早期研究工作只用编码器网络最后的状态作为解码器的初始状态

不论是编码器还是解码器都是循环神经网络，也就是 [LSTM 或 CRU](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 单元。下图是编码器-解码器模型示例，将“she is eating a green apple”译成中文，其中编码器和解码器部分都被铺开了

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-22_encoder-decoder-example.png)

这种定长环境向量的设计存在明显而致命的缺陷——无法记忆长序列。处理完整个输入序列之后早已忘了最开始的部分，注意力机制（[Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf)）就是针对该问题提出的。

### 为翻译而生

注意力机制开始是为了在神经机翻译（[NMT](https://arxiv.org/pdf/1409.0473.pdf)）任务中帮助记忆长源序列而设计的。相较于根据编码器最后的隐态构建一个单独的环境向量，注意力的诀窍是在环境向量和整个源输入序列间修了条秘道。这些捷径的权重对各个输出元素来讲都是可以定制的。

尽管环境向量要遍历整个输入序列，但我们并不需要担心遗忘问题。环境向量可以学习并控制源与目标间的一致性，本质上其消费三类数据：

- 编码器隐态
- 解码器隐态
- 源与目标间的一致性

下图展示了 [Bahdanau 等，2015](https://arxiv.org/pdf/1409.0473.pdf) 研究中带加权注意力机制的编码器-解码器模型结构

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-24_encoder-decoder-attention.png)

#### 定义

现在来试着科学地给 NMT 中提出的注意力机制下个定义。假定我们现在有长度为 $$n$$ 的源序列 $$\mathbf{x}$$，试着输出长度为 $$m$$ 的目标序列 $$\mathbf{y}$$：


$$
\begin{aligned}\mathbf{x} &= [x_1, x_2, \dots, x_n] \\\mathbf{y} &= [y_1, y_2, \dots, y_m]\end{aligned}
$$


_（粗体变量表示向量，下文同理）_

编码器是双向 RNN（[bidirectional RNN](https://www.coursera.org/lecture/nlp-sequence-models/bidirectional-rnn-fyXnn) ）结构，有着一个前向隐态  $$\overrightarrow{\boldsymbol{h}}_i$$ 和一个后向隐态 $$\overleftarrow{\boldsymbol{h}}_i$$ ，简单拼接二者就是编码器的状态了。这么做是为了将中心词前面和后面的注解词都囊括在内。


$$
\boldsymbol{h}_i = [\overrightarrow{\boldsymbol{h}}_i^\top; \overleftarrow{\boldsymbol{h}}_i^\top]^\top, i=1,\dots,n
$$


解码器网络由 $$t$$ 步（$$t=1,\dots,m$$）的输出词的隐态 $$\boldsymbol{s}_t=f(\boldsymbol{s}_{t-1}, y_{t-1}, \mathbf{c}_t)$$ 构成，而环境向量 $$\mathbf{c}_t$$ 则是输入序列隐态的加权和，权重为一致性得分：


$$
\begin{aligned}
\mathbf{c}_t &= \sum_{i=1}^n \alpha_{t,i} \boldsymbol{h}_i & \small{\text{; 输出 }y_t\text{ 的环境向量}}\\
\alpha_{t,i} &= \text{align}(y_t, x_i) & \small{\text{; 单词 }y_t\text{ 和 }x_i\text{ 的一致性}}\\
&= \frac{\exp(\text{score}(\boldsymbol{s}_{t-1}, \boldsymbol{h}_i))}{\sum_{i'=1}^n \exp(\text{score}(\boldsymbol{s}_{t-1}, \boldsymbol{h}_{i'}))} & \small{\text{; 对预定义的一致性得分求 softmax}}
\end{aligned}
$$


一致性模型会给 $$i$$ 处输入和 $$t$$ 处输出所组成的数据对分配一个评分 $$\alpha_{t,i}$$ ，取值大小看匹配程度如何。得分集合 $$\{\alpha_{t, i}\}$$ 体现了每个输出词对源序列中各个隐态的取舍。在 Bahdanau 的论文中，一致性得分 $$\mathbf{\alpha}$$ 是通过一个单层**前馈网络**进行参数化的，该网络和模型其他部分一起联合训练。所以得分函数就是下面这种形式，用 tanh 做非线性激活函数：


$$
\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_t; \boldsymbol{h}_i])
$$


式中的  $$\mathbf{v}_a$$ 和 $$\mathbf{W}_a$$ 都是一致性网络中要学习的权重矩阵，而一致性得分矩阵是个不错的副产品，能细腻地展示出源词与目标词间的关联程度。

![img](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-10-26_visual_translation.png)

更多实现上的指导可以看 Tensorflow 团队这篇很棒的[教程](https://www.tensorflow.org/versions/master/tutorials/seq2seq)

### 注意力家族

在注意力的帮助下，源序列和目标序列间的依赖不再受距离约束了！受到注意力大幅改善机器翻译表现的鼓舞，很快这项技术就被推广到了计算机视觉领域（[Xu 等 2015](http://proceedings.mlr.press/v37/xuc15.pdf)），同时人们开始探索各种其他样式的注意力机制（[Luong 等, 2015](https://arxiv.org/pdf/1508.04025.pdf)； [Britz 等, 2017](https://arxiv.org/abs/1703.03906); [Vaswani 等, 2017](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)）

#### 汇总

下面是一些时下流行的注意力机制及相应一致性评分函数的汇总表

| 名称             | 一致性评分函数                                               | 出处                                                         |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 基于内容的注意力 | $$\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \text{cosine}[\boldsymbol{s}_t, \boldsymbol{h}_i]$$ | [Graves2014](https://arxiv.org/abs/1410.5401)                |
| 加权(*)          | $$\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_t; \boldsymbol{h}_i])$$ | [Bahdanau2015](https://arxiv.org/pdf/1409.0473.pdf)          |
| 基于位置         | $$\alpha_{t,i} = \text{softmax}(\mathbf{W}_a \boldsymbol{s}_t)$$<br/>注解：简化了 Softmax 对齐操作，只依赖目标位置 | [Luong2015](https://arxiv.org/pdf/1508.04025.pdf)            |
| 一般式           | $$\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \boldsymbol{s}_t^\top\mathbf{W}_a\boldsymbol{h}_i$$<br/>$$\mathbf{W}_a$$ 是注意力层要训练的权重矩阵 | [Luong2015](https://arxiv.org/pdf/1508.04025.pdf)            |
| 点积             | $$\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \boldsymbol{s}_t^\top\boldsymbol{h}_i$$ | [Luong2015](https://arxiv.org/pdf/1508.4025.pdf)             |
| 比例点积(^)      | $$\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \frac{\boldsymbol{s}_t^\top\boldsymbol{h}_i}{\sqrt{n}}$$<br/>注解：和点积注意力很像，只是多了比例因子，n 是源隐态的维度 | [Vaswani2017](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) |

（\*）指代 Luong 等人的 “连接”，和 Vaswani 等人的“加权注意力”

（^）加上了比例因子 $$\frac {1}{\sqrt {n}}$$，主要是考虑到输入很多时，softmax 函数可能会取到很小的梯度，难以高效学习

下表更宽泛的对注意力机制进行了归类：

| 名称          | 定义                                                         | 出处                                                         |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 自注意力(&)   | 在相同输入序列的不同部分间形成联系。理论上自注意力可以使用上述任意形式的评分函数，只是把目标序列换成了相同的输入序列 | [Cheng2016](https://arxiv.org/pdf/1601.06733.pdf)            |
| 全局/软注意力 | 关照整个输入状态空间                                         | [Xu2015](http://proceedings.mlr.press/v37/xuc15.pdf)         |
| 局部/硬注意力 | 关照局部输入状态空间，比如输入图像的一组像素点               | [Xu2015](http://proceedings.mlr.press/v37/xuc15.pdf); [Luong2015](https://arxiv.org/pdf/1508.04025.pdf) |

（&）在 Cheng 2016 和一些其他论文中又被称为“内在注意力”

#### 自注意力

**自注意力（self-attention）**又叫做**内在注意力（intra-attention）**，是一种将单一序列的不同部分关联起来的注意力机制，一般是为了计算相同序列的表征。在机器阅读，摘要总结或是图像描述生成等任务中十分有效。

提出长短文记忆网络（[long short-term memory network](https://arxiv.org/pdf/1601.06733.pdf)，LSTM）的论文中使用自注意力做机器阅读。如下图所示，自注意力机制使我们可以学习当前词汇和先前单词间的关系，红色是当前词，蓝色阴影大小表示活跃水平。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-24_self-attention-example.png)

#### 软注意力 vs 硬注意力

在 [show, attend and tell](http://proceedings.mlr.press/v37/xuc15.pdf) 一文中，注意力机制用来给图片生成标题。图像先经过 CNN 编码抽取特征，然后使用 LSTM 解码器解析卷积特征来逐个生成描述词，权重通过注意力学到。注意力权重的可视化清晰地展示了模型当前正“看着”哪个区域来输出准确的描述词。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-24_soft-hard-attention.png)

这篇论文首次指出，“软”与“硬”注意力间的差别在于注意力是关照整个图像还只是其中一块：

- **软**注意力：在整张源图像上学习一致性权重并“柔和地”赋予各像素块，本质上和 [Bahdanau 等, 2015](https://arxiv.org/abs/1409.0473) 中的是一类注意力
  - 优点：模型平滑可微
  - 缺点：如果源图像很大则计算成本高昂
- **硬**注意力：每次只关注图像的某个像素块
  - 优点：推理时间更少
  - 缺点：模型不可微，需要像方差缩减或强化学习等更复杂的技术来进行训练

#### 全局注意力 vs 局部注意力

[Luong 等, 2015](https://arxiv.org/pdf/1508.04025.pdf) 提出“全局”和“局部”注意力。全局注意力和软注意力类似，而局部注意力则是软硬混合体，改进硬注意力使其可微：模型首先针对当前目标词预测一个对齐位置，并以源词为中心框定窗口计算环境向量。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-24_global-local-attention.png)

### 神经图灵机

图灵在 [1936](https://en.wikipedia.org/wiki/Turing_machine) 年提出了一种简洁计算模型，由一根无限长的纸带和运算头组成，纸带上有无数单元格，每个单元格以 0，1 或空格（“ ”）进行填充，运算头可以识别、编辑单元格内的标识，能向左或向右移动纸带。理论上图灵机可以模拟任意计算机算法，无论其多么复杂或计算代价多么高昂。无限的记忆容量使图灵机没有什么计算限制，但现实中的计算机显然做不到这点，所以我们也只是将图灵机当作一种数学计算模型

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-24_turing-machine.jpg)

**神经图灵机（Neural Turing Machine，NTM, [Graves, Wayne & Danihelka, 2014](https://arxiv.org/abs/1410.5401)) **是将外部记忆存储与神经网络相结合的框架，其中记忆存储类似于图灵机的纸带，神经网络则控制着运算头从哪读、往哪写。但 NTM 的记忆容量毕竟有限，所以它更像是“神经冯诺依曼机”。

NTM 由两部分构成，一个充当 _控制器（controller）_ 的神经网络和一个 _记忆（memory）_ 库。控制器负责在记忆库上进行运算，它可以是任意类型的神经网络，前馈或循环都行。记忆库则存储着处理过的信息，是一个大小为 $$N \times M$$ 的矩阵，包含 $$N$$ 个行向量，每个 $$M$$ 维。

在一次更新迭代中，控制器会处理输入并和记忆库交互来生成输出。交互通过一组并行的_读（read）写（write）_头完成。读写操作都是“模糊的”，只是简单关照下所有记忆内容。

<img src="https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_NTM.png" style="zoom:33%;" />

#### 读写

在 $$t$$ 时刻从记忆库中读取内容时，大小为 $$N$$ 的注意力向量 $$\mathbf{w}_t$$ 控制着记忆库各位置（矩阵的行）所应分配的注意力多少。读向量 $$\mathbf{r}_t$$ 是以注意力为权重的加权和：


$$
\mathbf{r}_t = \sum_{i=1}^N w_t(i)\mathbf{M}_t(i)\text{, 其中}\sum_{i=1}^N w_t(i)=1, \forall i: 0 \leqslant w_t(i) \leqslant 1
$$


这里 $$w_t(i)$$ 是 $$\mathbf{w}_t$$ 的第 $$i$$ 个元素，而 $$\mathbf{M}_t(i)$$  则是记忆库中第 $$i$$ 个行向量。

在 $$t$$ 时刻向记忆库写入内容时，在 LSTM 的输入门和遗忘门的控制下，写入头先会按照擦除向量 $$\mathbf{e}_t$$ 删掉部分旧内容，然后加上向量  $$\mathbf{a}_t$$ 补充新信息。


$$
\begin{aligned}
\tilde{\mathbf{M}}_t(i) &= \mathbf{M}_{t-1}(i) [\mathbf{1} - w_t(i)\mathbf{e}_t] &\scriptstyle{\text{; 擦除}}\\
\mathbf{M}_t(i) &= \tilde{\mathbf{M}}_t(i) + w_t(i) \mathbf{a}_t &\scriptstyle{\text{; 新增}}
\end{aligned}
$$


#### 注意力机制

神经图灵机中怎样生成注意力分布 $$\mathbf{w}_t$$ 取决于寻址机制：NTM 采用的是基于内容和基于位置的混合寻址方式

- 基于内容的寻址

基于内容的寻址就是依靠控制器从输入中抽出来的键向量 $$\mathbf{k}_t$$ 和记忆库行向量间求相似度来构建注意力向量。基于内容的注意力得分使用余弦相似度计算然后使用 softmax 归一化。此外， NTM 加上了强度乘数 $$\beta_t$$ 来放大或缩小分布重心


$$
w_t^c(i) 
= \text{softmax}(\beta_t \cdot \text{cosine}[\mathbf{k}_t, \mathbf{M}_t(i)])
= \frac{\exp(\beta_t \frac{\mathbf{k}_t \cdot \mathbf{M}_t(i)}{\|\mathbf{k}_t\| \cdot \|\mathbf{M}_t(i)\|})}{\sum_{j=1}^N \exp(\beta_t \frac{\mathbf{k}_t \cdot \mathbf{M}_t(j)}{\|\mathbf{k}_t\| \cdot \|\mathbf{M}_t(j)\|})}
$$


接着使用插值标量 $$g_t$$ 来混合上一时步的注意力权重和刚生成的基于内容的注意力向量


$$
\mathbf{w}_t^g = g_t \mathbf{w}_t^c + (1 - g_t) \mathbf{w}_{t-1}
$$


- 基于位置的寻址

基于位置的寻址将是将注意力向量中的各元素值加权求和，由容许的整数移位分布来确认权重，等价于使用 $$\mathbf{s}_t(.)$$ 核的一维卷积，$$\mathbf{s}_t$$ 是一个位置偏移函数。有很多方式来确认分布，下图展示了其中两种

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-25_shift-weighting.png)

最后注意力分布会被锐化标量   $$\gamma_t \geqslant 1$$ 所强化。


$$
\begin{aligned}
\tilde{w}_t(i) &= \sum_{j=1}^N w_t^g(j) s_t(i-j) & \scriptstyle{\text{; 环形卷积}}\\
w_t(i) &= \frac{\tilde{w}_t(i)^{\gamma_t}}{\sum_{j=1}^N \tilde{w}_t(j)^{\gamma_t}} & \scriptstyle{\text{; 锐化}}
\end{aligned}
$$


时步 t 下完整的注意力向量 $$\mathbf{w}_t$$ 生成过程如下图所示。所有参数由控制器生成，每个头的都不一样。如果有多个读写头并行，控制器会输出多个结果集。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-25_NTM-flow-addressing.png)

### 指针网络

在分拣或旅行推销的场景下，无论是输入还是输出都是序列化数据。不幸的是，这类问题不能简单的靠传统序列到序列或 NMT 模型加以解决，毕竟输出的细分类别事先并不知道，而是完全由输入规格决定。对此专门提出了**指针网络**（**Pointer Net ，Ptr-Net **,[Vinyals 等 2015](https://arxiv.org/abs/1506.03134)）：输出元素与输入序列中的_位置_ 相对应，相较于使用注意力对编码器隐层单元和环境向量进行混合，指针网络对所有输入元素分配注意力以从中挑选一个作为解码环节的输出

<img src="https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_ptr-net.png" style="zoom:33%;" />

给定向量输入序列 $$\boldsymbol{x} = (x_1, \dots, x_n)$$，指针网络输出一个整数索引序列 $$\boldsymbol{c} = (c_1, \dots, c_m)$$， $$1 \leqslant c_i \leqslant n$$，模型整体还是编码器-解码器构型，其中编码器隐态表示为  $$(\boldsymbol{h}_1, \dots, \boldsymbol{h}_n)$$ 解码器隐态则为  $$(\boldsymbol{s}_1, \dots, \boldsymbol{s}_m)$$。注意， $$\mathbf{s}_i$$ 是解码器中在单元激活后的输出门。指针网络在状态间采用加权注意力并使用 softmax 归一化来对输出的条件概率建模：


$$
\begin{aligned}
y_i &= p(c_i \vert c_1, \dots, c_{i-1}, \boldsymbol{x}) \\
    &= \text{softmax}(\text{score}(\boldsymbol{s}_t; \boldsymbol{h}_i)) = \text{softmax}(\mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_t; \boldsymbol{h}_i]))
\end{aligned}
$$


这里的注意力机制是简化过的，因为指针网络并没有基于注意力权重将编码器状态融入输出当中，这样输出就仅对应于输入位置而非输入内容。

### Transformer

["Attention is All you Need"](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) 无疑是 2017 年最具影响力的有趣论文之一，文章对软注意力进行了大幅改进使得 seq2seq 模型可以不再依赖于循环网络单元。其中提出的 **transformer** 模型完全基于自注意机制，无需使用序列对齐的循环结构。其中所有的奥秘都存在它的模型框架中。

#### 键，值和查询

transformer 主要由_多头（端）自注意力机制（ multi-head self-attention mechanism）_ 单元构成。transformer 将输入的编码表示视作一组**键值对$$(\mathbf{K}, \mathbf{V})$$**，二者都是 $$n$$ 维（输入序列长度）的。在 NMT 场景中，键与值是编码器隐态，而在解码器部分，上一个输出会被压缩为一个 $$m$$ 维的**查询向量（query，Q）**，下一个输出通过将该向量与键值对组做映射得到。

transformer 使用的是[比例点击注意力](#汇总)：输出是取值的加权和，分配给各个取值的权重由查询向量和键向量点积结果决定


$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{n}})\mathbf{V}
$$


#### 多头自注意力

<img src="https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_multi-head-attention.png" style="zoom:50%;" />

相较于单次计算注意力，多头机制可以同时进行多个比例点积注意力的计算。彼此独立的注意力结果简单的拼接后线性变换为期望的维数。这么做莫不是因为集成总是有好处的？按论文所说

> 多头机制使模型一道关照了不同位置的不同表示**子空间**，只有一个注意力头抑制了这点


$$
\begin{aligned}
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= [\text{head}_1; \dots; \text{head}_h]\mathbf{W}^O \\
\text{其中 head}_i &= \text{Attention}(\mathbf{Q}\mathbf{W}^Q_i, \mathbf{K}\mathbf{W}^K_i, \mathbf{V}\mathbf{W}^V_i)
\end{aligned}
$$


 $$\mathbf{W}^Q_i$$, $$\mathbf{W}^K_i$$, $$\mathbf{W}^V_i$$, 和 $$\mathbf{W}^O$$ 是要学习的参数矩阵。

#### 编码器

<img src="https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_transformer-encoder.png" style="zoom: 33%;" />

编码器生成基于注意力的表示，能从潜在无限大的上下文环境中定位特定信息片段

- 一组 $$N=6$$ 个独立层
- 每一层都有一个**多头注意力层**和一个简单的位置层面的**全连接前馈网络**
- 每个子层都使用[残差](https://arxiv.org/pdf/1512.03385.pdf)连接，并跟着一个**归一化**层。所有子层输出数据都是相同的维度 $$d_\text{model} = 512$$

#### 解码器

<img src="https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_transformer-decoder.png" style="zoom: 33%;" />

解码器从编码后的表示中检索信息。

- 一组 $$N=6$$ 个独立层

- 每层有两个多头注意力子层和一个全连接前馈网络子层
- 类似于编码器，每个子层使用残差连接并接着归一化
- 对第一个多头注意力子层稍作调整防止关照到后续位置，因为我们不想在预测当前位置内容的时候看到未来的目标序列

#### 完整框架

至此，transformer 框架的整体视角就有了：

- 源序列与目标序列都要先经过嵌入层来得到  $$d_\text{model} = 512$$ 维的数据表示
- 为了保留位置信息，对位置进行正弦波编码然后加上嵌入结果
- 编码器的最终输出经 softmax 和线性层处理得到框架的输出结果

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_transformer.png)

### SNAIL

transformer 没有循环或卷积结构，即便是给嵌入向量加上了位置编码，序列顺序信息的表示还是很弱。对于位置依赖敏感的问题，比如[强化学习](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)，这会造成大麻烦。

**简单神经注意力[元学习器](http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)（Simple Neural Attention Meta-Learner，SNAIL, [Mishra 等 2017](http://metalearning.ml/papers/metalearn17_mishra.pdf) ）**部分解决了这个问题。tranformer 模型定位部分使用自注意力和[时间卷积](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)的组合，这种方式无论是在监督学习还是强化学习任务中都取得了良好效果。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_snail.png)

SNAIL 源自元学习领域，这是值得单独写篇文章介绍的话题了。简而言之，元学习是希望模型对有着相似分布的新的未知任务有较强的泛化能力。感兴趣可以看看这篇[介绍](http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)

### 自注意 GAN

_自注意 GAN_(**SAGAN**; [Zhang 等, 2018](https://arxiv.org/pdf/1805.08318.pdf)) 将自注意层引入 [GAN](https://libertydream.github.io/2020/04/19/一文了解对抗生成网络/) 中使生成器和判别器能更好的对空间区域间的关系进行建模。

传统的 [DCGAN](https://arxiv.org/abs/1511.06434)(Deep Convolutional GAN) 将判别器和生成器都用多层卷积网络进行表示。但是网络表征能力受卷积核大小限制，因为像素特征被限制在了一个较小的局部区域内。为了和遥远区域建立联系，特征必须削弱层中卷积运算的影响，而且无法保证还能维持依赖关系。

而视觉场景中的（软）自注意力就是用来精确学习单像素和其他位置像素间的关系的，即使离得很远，它能轻易地捕获全局依赖。因此人们期望结合了自注意力的 GAN 能更好的处理图像细节。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_conv-vs-self-attention.png)

SAGAN 使用非局部神经网络（[non-local neural network](https://arxiv.org/pdf/1711.07971.pdf)）进行注意力计算。图像卷积特征映射 $$\mathbf{x}$$ 拷贝三份，分别对应 transformer 中的键，值和查询向量的计算：

- Key: $$f(\mathbf{x}) = \mathbf{W}_f \mathbf{x}$$
- Query: $$g(\mathbf{x}) = \mathbf{W}_g \mathbf{x}$$
- Value: $$h(\mathbf{x}) = \mathbf{W}_h \mathbf{x}$$

接着求点积注意力得到自注意特征映射：


$$
\begin{aligned}
\alpha_{i,j} &= \text{softmax}(f(\mathbf{x}_i)^\top g(\mathbf{x}_j)) \\
\mathbf{o}_j &= \sum_{i=1}^N \alpha_{i,j} h(\mathbf{x}_i)
\end{aligned}
$$


![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_self-attention-gan-network.png)

$$\alpha_{i,j}$$ 是注意力映射入口，表示在生成第 $$j$$ 处值时要对第 $$i$$ 处位置投放多少注意力。 $$\mathbf{W}_f$$, $$\mathbf{W}_g$$, 和 $$\mathbf{W}_h$$ 都是 $$1 \times 1$$ 卷积核。如果你觉得 $$1 \times 1$$ 听起来有些怪（它不就是说整个特征映射和一个数相乘吗？），可以看看吴恩达的简明[教程](https://www.coursera.org/lecture/convolutional-neural-networks/networks-in-networks-and-1x1-convolutions-ZTb8x)。输出 $$\mathbf{o}_j$$ 是最终结果 $$\mathbf{o}= (\mathbf{o}_1, \mathbf{o}_2, \dots, \mathbf{o}_j, \dots, \mathbf{o}_N)$$ 的一个列向量

此外，注意力层的结果还要和一个比例参数相乘再加回原输入特征映射：


$$
\mathbf{y} = \mathbf{x}_i + \gamma \mathbf{o}_i
$$


比例参数 $$\gamma$$ 在训练过程从零逐渐增大，网络一开始会依靠局部区域所提供的信息而后逐渐学会给更远处的区域分配更多权重

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-04-27_SAGAN-examples.png)