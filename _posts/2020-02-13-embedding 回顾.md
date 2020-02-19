---
layout:		post
title:  	embedding 回顾
subtitle:   textRank，2vec...的数理逻辑
date:       2020-02-13
author:     一轩明月
header-img: img/post-bg-computer-vision.jpg
catalog: 	 true
tags:
    - math
    - embedding
    - NLP
---

> 文内有大量数学公式，chrome 浏览器可以安装[这个](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)插件，以便浏览

自然语言需要翻译成机器语言才能为机器处理，最基础的一个问题就是怎么让计算机理解“词”与“字”，实现方式就是数字向量化

具体而言，在早期阶段就是单纯的描述一下共现关系或者结对共现关系，比如 n-grams，textRank。实现简单，但忽视了实体间的顺序和所处上下文，故效果较差。

### textRank

$$
WS\left(V_{i}\right)=(1-d)+d * \sum_{V_{j} \in I n\left(V_{i}\right)} \frac{w_{j i}}{\sum_{V_{k} \in O u t\left(V_{j}\right)} w_{j k}} W S\left(V_{j}\right)
$$

textRank 脱胎于 Google 网页搜索技术 PageRank，主要思想是将文档视作图，文本单元、单元间关系或者整个句子作为图中节点，随机初始化图中各节点权重，之后反复迭代，每次迭代中各节点将自身权重值平均分配给所有与其相连的点，各节点将从各边传来的权重加和作为自己的新权重，如此反复直到收敛至设定阈值之内，此时节点权重趋于稳定。该过程可以概括为四步：

1. 通过工具按给定粒度标明文本单元，并将其作为结点添加至图中
2. 确认文本单元间关系，并以此画边。可以是有向或无向，有权或无权
3. 按基于图的排序算法迭代直至收敛
4. 按最终权重值给结点排序，以此顺序做决策

### Word2Vec

2013 年 Google 研究团队在前人模型基础上提出了两个新模型架构 CBOW（Continuous bag-of-word） 和 skip-gram，在词相似度任务上大幅提高了准确率，大幅降低了计算成本，同时虽是针对特殊任务提出，但对其他任务也很有效，自此掀起了一股 “2Vec” 风潮。

相较早期的词袋、n-gram等模型，word2vec 优势在于不仅在词空间内相似词离得更近，而且每个词可以从多维度计算相似性，语义运算则是“意外之喜”。预训练、迁移学习思想自此开始流行。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-02-11_cbow_sngm.png)

- 原始 CBOW

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-02-12_cbow.png)

$$\mathbf{W}$$ 是输入层到隐藏层的权重矩阵，隐藏层输出结果为（效果等同于将 one-hot 编码中非零行挑选出来） 


$$
\begin{aligned}
\mathbf{h} &=\frac{1}{C} \mathbf{W}^{T}\left(\mathbf{x}_{1}+\mathbf{x}_{2}+\cdots+\mathbf{x}_{C}\right) \\
&=\frac{1}{C}\left(\mathbf{v}_{w_{1}}+\mathbf{v}_{w_{2}}+\cdots+\mathbf{v}_{w_{C}}\right)^{T}
\end{aligned}\\
\mathbf{W}^{T} \mathbf{x}_c=\mathbf{W}_{(k, \cdot)}^{T}:=\mathbf{v}_{w_{Ic}}^{T}
$$


$$\mathbf{v}_{w_{I}}$$ 是 $$w$$ 的输入向量。$$\mathbf{W^{\prime}}$$ 是隐藏层到输出层的权重矩阵，$$\mathbf{v}_{w_{j}}^{\prime}$$ 是其第 j 列的向量，可以据此对任意上下文内的词向量中的元素计算得分  $$u_{j}={\mathbf{v}_{w_{j}}^{\prime}}^T \mathbf{h}$$  。词间相关性为


$$
\begin{aligned}
p\left(w_{O} | w_{I1}, \cdots, w_{IC}\right)=y_{j}&=\frac{\exp \left(u_{j}\right)}{\sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right)}\\
&=\frac{\exp \left({\mathbf{v}_{w_{j}}^{\prime}}^T {\mathbf{h}}\right)}{\sum_{j^{\prime}=1}^{V} \exp \left({\mathbf{v}_{w_{j^{\prime}}}^{\prime}}^T {\mathbf{h}}\right)}
\end{aligned}
$$



$${\mathbf{v}_{w}}^\prime$$ 是 $$w$$ 的输出向量。目标是求解最优化问题



$$
\begin{aligned}
\max p\left(w_{O} | w_{I1}, \cdots, w_{IC}\right) &=\max y_{j^{*}} \\
&=\max \log y_{j^{*}} \\
&=u_{j^{*}}-\log \sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right):=-E
\end{aligned}
$$



$$E$$ 是损失函数


$$
\begin{aligned}
E&=-\log p\left(w_{O} | w_{I1}, \cdots, w_{IC}\right)\\
&= -{\mathbf{v}_{w_{O}}^{\prime}}^T \cdot \mathbf{h}+\log \sum_{j^{\prime}=1}^{V} \exp \left({\mathbf{v}_{w_{j}}^{\prime}}^T \cdot \mathbf{h}\right)
\end{aligned}
$$


$$j^*$$ 是实际输出词，对输出层第 j 个单词求偏导得



$$
\frac{\partial E}{\partial u_{j}}=y_{j}-t_{j}:=e_{j}
$$

 

$$t_{j}=\mathbb{1}$$ 当且仅当 $$j=j^{*}$$ ，否则为 0，对输出矩阵 $$\mathbf{W^{\prime}}$$  元素求偏导

$$
\frac{\partial E}{\partial w_{i j}^{\prime}}=\frac{\partial E}{\partial u_{j}} \cdot \frac{\partial u_{j}}{\partial w_{i j}^{\prime}}=e_{j} \cdot h_{i}
$$


得到更新公式，其中 $$\eta > 0$$ 为学习率 


$$
\begin{array}{c}
{w_{i j}^\prime}^{(\text { new })}={w_{i j}^\prime}^{(\text { old })}-\eta \cdot e_{j} \cdot h_{i} \\
{\mathbf{v}_{w_{j}}^{\prime}}^{(\text { new })}={\mathbf{v}_{w_{j}}^{\prime}}^{(\text { old })}-\eta \cdot e_{j} \cdot \mathbf{h} \quad \text { for } j=1,2, \cdots, V
\end{array}
$$

同理可推得输入层更新方法



$$
\frac{\partial E}{\partial h_{i}}=\sum_{j=1}^{V} \frac{\partial E}{\partial u_{j}} \cdot \frac{\partial u_{j}}{\partial h_{i}}=\sum_{j=1}^{V} e_{j} \cdot w_{i j}^{\prime}:=\mathrm{EH}_{i}\\
h_{i}=\frac{1}{C}\sum_{c=1}^{C}\sum_{k=1}^{V} x_{ck} \cdot w_{k i}\\
\frac{\partial E}{\partial w_{k i}}=\frac{\partial E}{\partial h_{i}} \cdot \frac{\partial h_{i}}{\partial w_{k i}}=\mathrm{EH}_{i} \cdot \frac{1}{C}\sum_{c=1}^{C}\sum_{k=1}^{V} x_{ck}
$$



$$\mathrm{EH}$$ 可以理解为预测误差的加权求和。对输入层权重求导可以写成张量积的形式$$\frac{\partial E}{\partial \mathbf{W}}=\frac{1}{C} \sum_{c=1}^C \mathbf{x}_c \otimes \mathrm{EH}=\frac{1}{C} \sum_{c=1}^C \mathbf{x}_c \mathrm{EH}^{T}$$ 。因为 $$\mathbf{x}$$ 只有一个值为1，其余为 0，所以更新公式可以写成



$$
\mathbf{v}_{w_{I, c}}^{(\text {new })}=\mathbf{v}_{w_{I, c}}^{(\text {old })}-\frac{1}{C} \cdot \eta \cdot \mathrm{EH}^{T} \quad \text { for } c=1,2, \cdots, C
$$



- 原始 Skip-gram

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-02-12_skip_gram.png)

skip-gram 的隐藏层定义不变 $$\mathbf{h}=\mathbf{W}_{(k, \cdot)}^{T}:=\mathbf{v}_{w_{I}}^{T}$$，因为只有一个输入，所以结果就是简单的从权重矩阵中拿一行出来赋给输入向量。但此时输出变成了 C 个多维分布


$$
p\left(w_{c, j}=w_{O, c} | w_{I}\right)=y_{c, j}=\frac{\exp \left(u_{c, j}\right)}{\sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right)}\\
u_{c, j}=u_{j}={\mathbf{v}_{w_{j}}^{\prime}}^T \cdot \mathbf{h}, \text { for } c=1,2, \cdots, C
$$


$$w_{c,j}$$ 是第 c 个输出词的第 j 个元素，此时损失函数变为


$$
\begin{aligned}
E &=-\log p\left(w_{O, 1}, w_{O, 2}, \cdots, w_{O, C} | w_{I}\right) \\
&=-\log \prod_{c=1}^{C} \frac{\exp \left(u_{c, j_{c}^{*}}\right)}{\sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right)} \\
&=-\sum_{c=1}^{C} u_{j_{c}^{*}}+C \cdot \log \sum_{j^{\prime}=1}^{V} \exp \left(u_{j^{\prime}}\right)
\end{aligned}
$$


同 CBOW 一样计算反向传播需要的偏导数


$$
\frac{\partial E}{\partial u_{c, j}}=y_{c, j}-t_{c, j}:=e_{c, j}\\
\mathrm{EI}_{j}:=\sum_{c=1}^{C} e_{c, j}\\
\frac{\partial E}{\partial w_{i j}^{\prime}}=\sum_{c=1}^{C} \frac{\partial E}{\partial u_{c, j}} \cdot \frac{\partial u_{c, j}}{\partial w_{ij}^{\prime}}=\mathrm{EI}_{j} \cdot h_{i}
$$


由此得到更新公式


$$
\begin{array}{c}
{w_{ij}^\prime}^{(\text { new })}={w_{ij}^\prime}^{(\text { old })}-\eta \cdot \mathrm{EI}_{j} \cdot h_{i} \\
{\mathbf{v}_{w_{j}}^{\prime}}^{(\text { new })}=\mathbf{v}_{w_{j}}^{\prime}(\text { old })-\eta \cdot \mathrm{EI}_{j} \cdot \mathbf{h} \quad \text { for } j=1,2, \cdots, V
\end{array}
$$


输入层更新方法和 CBOW 类似，$$e_j$$ 换为 $$\mathrm{EI}_j$$


$$
\frac{\partial E}{\partial h_{i}}=\sum_{j=1}^{V} \frac{\partial E}{\partial u_{j}} \cdot \frac{\partial u_{j}}{\partial h_{i}}=\sum_{j=1}^{V} \mathrm{EI}_{j} \cdot w_{i j}^{\prime}:=\mathrm{EH}_{i}\\
h_{i}=\sum_{k=1}^{V} x_{k} \cdot w_{k i}\\
\frac{\partial E}{\partial w_{k i}}=\frac{\partial E}{\partial h_{i}} \cdot \frac{\partial h_{i}}{\partial w_{k i}}=\mathrm{EH}_{i} \cdot x_{k}>>\frac{\partial E}{\partial \mathbf{W}}=\mathbf{x} \otimes \mathrm{EH}=\mathbf{x} \mathrm{EH}^{T}
$$


得到更新公式


$$
\mathbf{v}_{w_{I}}^{(\text {new })}=\mathbf{v}_{w_{I}}^{(\text {old })}-\eta \cdot \mathrm{EH}^{T}
$$


- Hierarchical Softmax

两种优化手段都是针对输出层的（每次都要遍历整个词表太费时了）。主要关注的变量有三个：1. 损失函数 $$E$$，2. 输出向量的偏导数 $$\frac{\partial E}{\partial \mathbf{v}_{w}^{\prime}}$$，3.隐藏向量的偏导数 $$\frac{\partial E}{\partial \mathbf{h}}$$。

具体的，Hierarchical Softmax 借用哈弗曼树，将问题简化为二元选择问题（左孩子还是右孩子），这样复杂度降低到 $$ O(\log V)$$，词表中的 V 个词作为叶子，需要计算的是 V-1 个内部结点。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-02-13_hierarchical_softmax.png)


$$
p\left(w=w_{O}\right)=\prod_{j=1}^{L(w)-1} \sigma\left([n(w, j+1)=\operatorname{ch}(n(w, j)) ] \cdot {\mathbf{v}_{n(w, j)}^{\prime}}^T \mathbf{h}\right)
$$


$$ch(n)$$ 指 n 结点的左孩子，$$L(w)$$ 是指从根节点到 w 的路径长度，$$n(w,j)$$ 是该路径上第 j 个结点，$$\mathbf{v}_{n(w,j)}^\prime$$ 是其输出向量。$$[x]$$ 是指示函数


$$
[x]=\left\{\begin{array}{ll}
{1} & {\text { if } x \text { is true }} \\
{-1} & {\text { otherwise }}
\end{array}\right.
$$


如果定义前往左、右孩子的概率: 


$$
p(n, \text { left })=\sigma\left(\mathbf{v}_{n}^{\prime T} \cdot \mathbf{h}\right)\\
p(n, \text { right })=1-\sigma\left(\mathbf{v}_{n}^{\prime T} \cdot \mathbf{h}\right)=\sigma\left(-\mathbf{v}_{n}^{\prime T} \cdot \mathbf{h}\right)
$$


上图所示路径可表示为


$$
\begin{aligned}
p\left(w_{2}=w_{O}\right) &=p\left(n\left(w_{2}, 1\right), \text { left }\right) \cdot p\left(n\left(w_{2}, 2\right), \text { left }\right) \cdot p\left(n\left(w_{2}, 3\right), \text { right }\right) \\
&=\sigma\left({\mathbf{v}_{n\left(w_{2}, 1\right)}^{\prime}}^T \mathbf{h}\right) \cdot \sigma\left({\mathbf{v}_{n\left(w_{2}, 2\right)}^{\prime}}^T \mathbf{h}\right) \cdot \sigma\left(-{\mathbf{v}_{n\left(w_{2}, 3\right)}^{\prime}}^T \mathbf{h}\right)
\end{aligned}
$$




易知 $$\sum_{i=1}^{V} p\left(w_{i}=w_{O}\right)=1$$，则损失函数可表达为：


$$
[\cdot]:=[n(w, j+1)=\operatorname{ch}(n(w, j))]\\
\mathbf{v}_{j}^{\prime}:=\mathbf{v}_{n_{w, j}}^{\prime}\\
E=-\log p\left(w=w_{O} | w_{I}\right)=-\sum_{j=1}^{L(w)-1} \log \sigma\left([\cdot]\mathbf{v}_{j}^{\prime T} \mathbf{h}\right)
$$


求两个偏导


$$
\begin{aligned}
\frac{\partial E}{\partial \mathbf{v}_{j}^{\prime} \mathbf{h}} &=\left(\sigma\left([\cdot]\mathbf{v}_{j}^{\prime T} \mathbf{h}\right)-1\right) [\cdot] \\
&=\left\{\begin{array}{ll}
{\sigma\left(\mathbf{v}_{j}^{\prime T} \mathbf{h}\right)-1} & {([\cdot]=1)} \\
{\sigma\left(\mathbf{v}_{j}^{\prime T} \mathbf{h}\right)} & {([\cdot]=-1)}
\end{array}\right.\\
&=\sigma\left(\mathbf{v}_{j}^{\prime T} \mathbf{h}\right)-t_{j}
\end{aligned}\\
\frac{\partial E}{\partial \mathbf{v}_{j}^{\prime}}=\frac{\partial E}{\partial \mathbf{v}_{j}^{\prime} \mathbf{h}} \cdot \frac{\partial \mathbf{v}_{j}^{\prime} \mathbf{h}}{\partial \mathbf{v}_{j}^{\prime}}=\left(\sigma\left(\mathbf{v}_{j}^{\prime T} \mathbf{h}\right)-t_{j}\right) \cdot \mathbf{h}\\
\mathbf{v}_{j}^{\prime(\mathrm{new})}=\mathbf{v}_{j}^{\prime(\mathrm{old})}-\eta\left(\sigma\left(\mathbf{v}_{j}^{\prime T} \mathbf{h}\right)-t_{j}\right) \cdot \mathbf{h}
$$

$$
\begin{aligned}
\frac{\partial E}{\partial \mathbf{h}} &=\sum_{j=1}^{L(w)-1} \frac{\partial E}{\partial \mathbf{v}_{j}^{\prime} \mathbf{h}} \cdot \frac{\partial \mathbf{v}_{j}^{\prime} \mathbf{h}}{\partial \mathbf{h}} \\
&=\sum_{j=1}^{L(w)-1}\left(\sigma\left(\mathbf{v}_{j}^{\prime} \mathbf{h}\right)-t_{j}\right) \cdot \mathbf{v}_{j}^{\prime} \\
&:=\mathbf{E} \mathbf{H}
\end{aligned}\\
\begin{aligned}
\text{CBOW:}&\quad \mathbf{v}_{w_{I, c}}^{(\text {new })}=\mathbf{v}_{w_{I, c}}^{(\text {old })}-\frac{1}{C} \cdot \eta \cdot \mathrm{EH}^{T} \quad \text { for } c=1,2, \cdots, C\\
\text{skip-gram:} &\quad \mathbf{v}_{w_{I}}^{(\text {new })}=\mathbf{v}_{w_{I}}^{(\text {old })}-\eta \cdot \mathrm{EH}^{T},\quad \mathrm{EH}_{i}=\sum_{j=1}^{V} \mathrm{EI}_{j} \cdot w_{i j}^{\prime}
\end{aligned}
$$


- Negative Sampling

负采样的思路是构造一个子集替代全集进行训练，以此降低复杂度。原语料中的句子作为正样本，负样本从噪声分布里抽取。记 $$p(D=1 \mid w, c)$$ 为样本属于原语料的概率，则 $$p(D=0 \mid w, c)=1-p(D=1 \mid w, c)$$，假定概率分布由参数 $$\theta$$ 决定，则目标是



$$
\begin{aligned}
& \arg \max _{\theta} \prod_{(w, c) \in D} p(D=1 | w, c ; \theta) \\
=& \arg \max _{\theta} \log \prod_{(w, c) \in D} p(D=1 | w, c ; \theta) \\
=& \arg \max _{\theta} \sum_{(w, c) \in D} \log p(D=1 | w, c ; \theta)
\end{aligned}
$$



定义 $$p(D=1 \mid w, c ; \theta)=\frac{1}{1+e^{-v_{c} \cdot v_{w}}}$$ ，得到目标函数


$$
\arg \max _{\theta} \sum_{(w, c) \in D} \log \frac{1}{1+e^{-v_{c} \cdot v_{w}}}
$$


光有正例概率都是 1 了，需要加上噪声带来的负样本 $$D^{\prime}$$，目标函数变成


$$
\begin{aligned}
& \arg \max _{\theta} \prod_{\theta} p(D=1 | c, w ; \theta) \prod_{(w, c) \in D^{\prime}} p(D=0 | c, w ; \theta) \\
=& \arg \max _{\theta} \prod_{(w, c) \in D} p(D=1 | c, w ; \theta) \prod_{(w, c) \in D^{\prime}}(1-p(D=1 | c, w ; \theta)) \\
=& \arg \max _{\theta} \sum_{(w, c) \in D} \log p(D=1 | c, w ; \theta)+\sum_{(w, c) \in D^{\prime}} \log (1-p(D=1 | w, c ; \theta)) \\
=& \arg \max _{\theta} \sum_{(w, c) \in D} \log \frac{1}{1+e^{-v_{c} \cdot v_{w}}}+\sum_{(w, c) \in D^{\prime}} \log \left(1-\frac{1}{1+e^{-v_{c} \cdot v_{w}}}\right) \\
=& \arg \max _{\theta} \sum_{(w, c) \in D} \log \frac{1}{1+e^{-v_{c} \cdot v_{w}}}+\sum_{(w, c) \in D^{\prime}} \log \left(\frac{1}{1+e^{v_{c} \cdot v_{w}}}\right)
\end{aligned}
$$


如果令 $$\sigma(x)=\frac{1}{1+e^{-x}}$$ 则


$$
\begin{aligned}
& \arg \max _{\theta} \sum_{(w, c) \in D} \log \frac{1}{1+e^{-v_{c} \cdot v_{w}}}+\sum_{(w, c) \in D^{\prime}} \log \left(\frac{1}{1+e^{v_{c} \cdot v_{w}}}\right) \\
=& \arg \max _{\theta} \sum_{(w, c) \in D} \log \sigma\left(v_{c} \cdot v_{w}\right)+\sum_{(w, c) \in D^{\prime}} \log \sigma\left(-v_{c} \cdot v_{w}\right)
\end{aligned}
$$


这是对全集$$D \cup D^{\prime}$$，原始论文中的表达式是一个正例 $$(w,c)\in D$$，和 $$k$$ 个从噪声分布中挑选出来的负例 $$(w,c_j) \in D^{\prime}$$


$$
\log \sigma\left(v_c \cdot v_w\right)+\sum_{i=1}^{k} \mathbb{E}_{c_j\sim P_{n}(w)}\left[\log \sigma\left(-v_{c_j}^{\prime}\cdot v_{w}\right)\right]
$$

噪声分布 $$P_n(w)$$ 是一元分布（每个词的词频分布）的 $$\frac{3}{4}$$ 次方。也就是说负样本是正样本的 $$k$$ 倍，满足分布$$(w, c) \sim p_{w o r d s}(w) \frac{p_{contexts}(c)^{3 / 4}}{Z}$$，这里 $$p_{w o r d s}(w)$$ 和 $$p_{contexts}(c)$$ 分别是单词和其上下文的一元分布，$$Z$$ 是归一化常数，因为原始论文中的上下文只有一个词，所以  $$p_{words}=p_{contexts}=\frac{count(x)}{\lvert Text \rvert}$$

有了构建样本集的理论，但具体怎么采集负例呢？从两方面入手，一来自身出现频次太低的不要，其次要平衡高频词和低频词间的比例。所以有了下列式子：


$$
P\left(w_{i}\right)=1-\sqrt{\frac{t}{f\left(w_{i}\right)}}
$$


每个词以概率 $$P$$ 被丢弃，$$t$$ 是限定阈值，原论文为 $$10^{-5}$$，$$f(w)$$ 是统计词频

### Glove

此前的两种主流方法：1.全局矩阵分解，如 LSA，2.基于局部窗口的方法，如 skip-gram 各有优劣。全局矩阵分解充分利用了语料的统计信息，但在词相似度比较上性能很差，基于局部窗口正好相反，通过多个语言维度的差异比较能很好的衡量词语间相似性，但因为局限在了某个窗口内，对全局的语料信息感知不到位。

斯坦福研究者们将二者结合，在保证词向量是线性关系的基础上，提出了对数双线性回归模型，依靠特定的加权最小二乘模型，基于全局“词-词”共现计数进行训练，即掌握了整体环境信息，也把握住了词间相似性

研究团队通过实验指出，词向量表示的起点应该共现概率而非标识独立出现的概率。此时语料内词间相关性计算方式为


$$
J=\sum_{i, j=1}^{V} f\left(X_{i j}\right)\left(w_{i}^{T} \tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{i j}\right)^{2}\\
f(x)=\left\{\begin{array}{cc}
{\left(x / x_{\max }\right)^{\alpha}} & {\text { if } x<x_{\max }} \\
{1} & {\text { otherwise }}
\end{array}\right.
$$


$$V$$ 是词表大小，$$f$$ 是代表共现词对的权重。$$X$$ 是共现计数，$$X_{ij}$$ 表示词 $$j$$  出现时，词 $$i$$ 一同出现的概率。

$$w_{i}^{T} \tilde{w}_{k}=\log \left(P_{i k}\right)=\log \left(X_{i k}\right)-\log \left(X_{i}\right),P_{i k}=\frac{X_{i k}}{X_{i}}$$。故括号内实际表示在向量空间内一个向量加上偏置后离另一个向量还有多远。有趣的是，研究人员发现这里幂指数 $$\alpha=\frac{3}{4}$$ 效果最佳，和 skip-gram 经验一致

相较于 Glove，skip-gram 相当于预先决定好了权重参数 $$X_i$$ ，选择最小二乘作为向量距离度量方式，且省去了归一化操作，如果动态考虑词的权重，则二者是一致的。过程如下


$$
Q_{i j}=\frac{\exp \left(w_{i}^{T} \tilde{w}_{j}\right)}{\sum_{k=1}^{V} \exp \left(w_{i}^{T} \tilde{w}_{k}\right)}\\
J=-\sum_{i \in \text { corpus } \atop j \in \operatorname{context}(i)} \log Q_{i j}=-\sum_{i=1}^{V} \sum_{j=1}^{V} X_{i j} \log Q_{i j}\\
\because X_{i}=\sum_{k} X_{i k},P_{i k}=\frac{X_{i k}}{X_{i}}\\
\therefore J=-\sum_{i=1}^{V} X_{i} \sum_{j=1}^{V} P_{i j} \log Q_{i j}=\sum_{i=1}^{V} X_{i} H\left(P_{i}, Q_{i}\right)
$$


将交叉熵 $$H$$ 换为最小二乘，省去归一化得


$$
\hat{J}=\sum_{i, j} X_{i}\left(\hat{P}_{i j}-\hat{Q}_{i j}\right)^{2}\\
\hat{P}_{i j}=X_{i j} ,\quad\hat{Q}_{i j}=\exp \left(w_{i}^{T} \tilde{w}_{j}\right)
$$


防止 $$X_{ij}$$ 过大，取对数运算，则


$$
\hat{J} =\sum_{i, j} X_{i}\left(\log \hat{P}_{i j}-\log \hat{Q}_{i j}\right)^{2} =\sum_{i, j} X_{i}\left(w_{i}^{T} \tilde{w}_{j}-\log X_{i j}\right)^{2}
$$


### FastText

受分布式词表示学习方法启发，脸书团队尝试针对性改进文本分类任务基准性能。过往基于神经网络的模型通常效果好，效率低，训练时间、测试时间都很长，限制了模型使用数据集规模。为此提出的 FastText 依旧是一个线性分类器，一大亮点在于，该模型在学习词表示的过程中，文本表示作为隐变量顺带被求出。

整体而言，FastText 结构很像 word2vec，只是预测的不再是中心词而是标签。使用了 Hieararchical Softmax 技巧，额外添加了 n-gram 特征。同时，输入不再是近邻圈定的固定窗口，而是标识序列

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2020-02-19_fasttext.png)

FastText 训练目标就是最小化分类误差：


$$
-\frac{1}{N} \sum_{n=1}^{N} y_{n} \log \left(f\left(B A x_{n}\right)\right)
$$


共 $$N$$ 篇文档，$$f$$ 是 softmax 函数，$$x_n$$ 是归一化处理后的第 $$n$$ 篇文档的特征向量，$$y_n$$ 是对应标签。$$A,B$$ 是权重矩阵。如果不优化，这种线性分类器的时间复杂度是 $$O(kh)$$，$$k$$ 是类别数量，$$h$$ 是文本表示维数。

通过 HS 技巧（同 word2vec），借助哈弗曼编码树，将复杂度降为 $$O(h\log_2 (k))$$。辅以二叉堆，可以在 $$O(log(T))$$ 时间内确认 T-top 标签。

- 字符级词表示

毫无疑问，文本分类任务的原子问题是怎么表示词与字，此前的 NLP 处理会将多余的词缀去掉保留词干，然后给每个词独立分配一个表示向量，不共享什么参数，这无疑忽略了词本身的内部结构，陷入了单一的英文视角，导致丢掉了很多词法信息。很多词法丰富，语料很少的语言比如法语、西班牙语在这种处理方式前很吃亏。

对此 FastText 在 skip-gram 模型的基础上进行改进。使用 n-gram 模式构建字符向量，用字符向量的组合表示词，这样极大的拓展了输出空间，字符级特征组合蕴含丰富的子词信息。

具体地，FastText 首先加入了两个特殊 token： `<` 和 `>`，用来表示词体开始和结尾，同时和其他词的前缀/后缀做区分。其次，词语本身也被加入了自身的特征向量集中，和 n-gram 子词特征一同训练。比如单词 `where`，当选定 $$n=3$$ 时，其特征向量的组成方式为 `<where>, <wh, whe, her, ere, re>` ，注意这里的 `her` 是 `where` 的3-gram 标识，和序列 `<her>` 指代词语 `her` 含义不同。实践中，脸书团队抽取了所有 $$3 \leqslant n \leqslant 6$$ 的 n-gram 标识。

回顾 skip-gram 模型：


$$
\sum_{t=1}^{T}\left[\sum_{c \in \mathcal{C}_{t}} \ell\left(s\left(w_{t}, w_{c}\right)\right)+\sum_{n \in \mathcal{N}_{t, c}} \ell\left(-s\left(w_{t}, n\right)\right)\right]
$$


语料集 $$T$$ 中每个词 $$w_t$$ 的近邻标识集为 $$ \mathcal{C}_{t}$$，$$s$$ 是衡量词对 $$(w_t,w_c)$$ 的得分函数，$$\ell: x \mapsto \log \left(1+e^{-x}\right)$$，$$\mathcal{N}_{t, c}$$ 是负样本集。假定 n-gram 字典大小为 $$G$$ ，词 $$w_t$$ 的 n-gram 集合为 $$\mathcal{G}_{w} \subset\{1, \ldots, G\}$$，集合内每个标识 $$g$$ 都对应一个向量 $$\mathbf{z}_g$$ 。词 $$w_t$$ 的 n-gram 组就由这些向量和表示。评分函数变为


$$
s(w, c)=\sum_{g \in \mathcal{G}_{w}} \mathbf{z}_{g}^{\top} \mathbf{v}_{c}
$$


鉴于模型体量有点大，脸书团队使用哈希技巧将 n-gram 标识映射成 1 到 $$K$$ 的整数。这样一个词就由词典中的索引号及其对应的 n-gram 哈希集表示。

借助字符级词表示，FastText 极大拓展了输出空间，能够预测未见过的新词。同时因为计算简单，训练、测试效率都很高且容易并行化。