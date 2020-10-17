---
layout:		post
title:  	从 Autoencoder 到 beta-VAE
subtitle:   
date:       2020-07-26
author:     一轩明月
header-img: img/post-bg-blue.jpg
catalog:    true
tags:
    - embedding
excerpt:    自动编码器是一家子神经网络模型，旨在学到高维数据的低维隐变量。本文从基本的自编码器模型开始，回顾了若干变体，包括降噪、稀疏和可收缩自编码器，然后介绍了变分自编码器（VAE）和它的改良版 beta-VAE
---

> 编译自：From Autoencoder to Beta-VAE， [Lilian Weng](https://lilianweng.github.io/lil-log/)

自动编码器用一个中间有狭小瓶颈层的神经网络重构高维数据（这可能对[变分自编码器](# VAE：变分自编码器)来讲不太准确，后面再说），这带来了一个不错的副产物——维度缩减：瓶颈层会习得压缩降维后的隐性编码。这样的低维表示可以担当多种应用中（比如搜索）的嵌入向量，帮助压缩数据，或是揭示潜在的数据生成要素。

### 符号说明

| 符号                                      | 含义                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| $$\mathcal{D}$$                           | 数据集，$$\mathcal{D} = \{ \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(n)} \}$$ 有 $$n$$ 个数据样本; $$\vert\mathcal{D}\vert =n $$. |
| $$\mathbf{x}^{(i)}$$                      | 数据项，是个 $$d$$ 维向量， $$\mathbf{x}^{(i)} = [x^{(i)}_1, x^{(i)}_2, \dots, x^{(i)}_d]$$ |
| $$\mathbf{x}$$                            | 数据集中的一个样本 $$\mathbf{x} \in \mathcal{D}$$            |
| $$\mathbf{x}’$$                           | $$\mathbf{x}$$ 的重构版                                      |
| $$\tilde{\mathbf{x}}$$                    | $$\mathbf{x}$$ 的缺损版                                      |
| $$\mathbf{z}$$                            | 瓶颈层学到的压缩码                                           |
| $$a_j^{(l)}$$                             | $$l$$ 层 $$j$$ 号神经元的激活函数                            |
| $$g_{\phi}(.)$$                           | 参数为 $$\phi$$ 的**编码**函数                               |
| $$f_{\theta}(.)$$                         | 参数为 $$\theta$$ 的**解码**函数                             |
| $$q_{\phi}(\mathbf{z}\vert\mathbf{x})$$   | 后验概率预估函数，也叫**概率编码器**                         |
| $$p_{\theta}(\mathbf{x}\vert\mathbf{z})$$ | 给定隐性编码生成正确数据的可能性，也叫**概率解码器**         |

### 自编码器

自编码器（Autoencoder）是个神经网络，以无监督的方式学习一个同化函数以重构原始输入数据，同时在过程中将数据压缩便于找到更有效、更凝练的表示。该思想可追溯至 [20 世纪 80年代](https://en.wikipedia.org/wiki/Autoencoder)，之后在一篇开创性论文[Hinton 和 Salakhutdinov, 2006](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.3788&rep=rep1&type=pdf)中被提出。

它包括两个网络：

- *编码器* 网络：将原高维输入变成低维隐性编码。输入尺寸比输出大
- *解码器* 网络：从编码中将数据复原，大概率是靠一个比一个大的输出层

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-autoencoder-architecture.png)

*图 1  自动编码器架构图*

编码网络本质上是在做[降维](https://en.wikipedia.org/wiki/Dimensionality_reduction)，和用主成分分析（PCA）或矩阵分解（MF）的目的一样。此外，自编码器对编码的数据重建做了显式优化。一个好的中间态表示不仅能学到隐变量，还能使整个[解压缩](https://ai.googleblog.com/2016/09/image-compression-with-neural-networks.html)过程收益。

模型包含一个参数为 $$\phi$$ 的编码函数 $$g(.)$$ 以及一个参数为 $$\theta$$ 的解码函数 $$f(.)$$。瓶颈层学到的有关输入 $$\mathbf{x}$$ 的低维编码为 $$\mathbf{z} = g_\phi(\mathbf{x})$$，重构后的输入为 $$\mathbf{x}' = f_\theta(g_\phi(\mathbf{x}))$$。

参数 $$(\theta, \phi)$$ 是一起学习的，目标是使重构的数据样本和原始输入一样， $$\mathbf{x} \approx f_\theta(g_\phi(\mathbf{x}))$$，换句话讲就是学个同化函数。有很多度量方法来量化两个向量间的差异，比如激活函数为 sigmoid 时的交叉熵，或是像 MSE 损失那么简单的方法：


$$
L_\text{AE}(\theta, \phi) = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\mathbf{x}^{(i)})))^2
$$



### 降噪自编码器

因为自编码器是在学同化函数，当网络参数比数据样本还多时就要面对“过拟合”的风险。为了避免过拟合改善鲁棒性，[Vincent 等，2008](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)在一般自编码器的基础上做了点改进提出了**降噪自编码器（Denoising Autoencoder）**。给输入加上噪声造成一定程度的缺损，或者用些随机方法遮住一些输入向量的值 $$\tilde{\mathbf{x}} \sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}} \vert \mathbf{x})$$。然后训练模型尝试复原输入（缺损项不算）


$$
\begin{aligned}
\tilde{\mathbf{x}}^{(i)} &\sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}}^{(i)} \vert \mathbf{x}^{(i)})\\
L_\text{DAE}(\theta, \phi) &= \frac{1}{n} \sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\tilde{\mathbf{x}}^{(i)})))^2
\end{aligned}
$$


其中 $$\mathcal{M}_\mathcal{D}$$ 是真实数据与噪声或缺损项之间的映射

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-denoising-autoencoder-architecture.png)

*图 2  降噪自编码器架构*

这种设计思想是受现实启发，即使视野中的景象有部分被遮挡或是缺损人们还是可以轻易识别出某个物体或场景。为了“修复”部分受损输入，降噪自编码器必须发掘并习得输入维度间的联系，这样才能推断丢掉了什么。

对那些高度冗余的高维输入，比如图像，模型很可能会从许多输入维度间的组合例证中学习复原数据而不是揪着一个维度过度拟合。这给学习*健壮* 的隐性表示打下了不错的基础。

噪声是由随机映射 $$\mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}} \vert \mathbf{x})$$ 控制的，同时不会受特定类型的混淆过程束缚（比如遮罩噪声，高斯噪声，盐和辣椒噪声等等），自然也可以在混淆过程里加上先验知识。原 DAE 论文的实验里，噪声是这么加的：随机抽取固定比例的输入维度强制其值为 0。听起来像 dropout 对吧？可惜降噪自编码器是 2008 年提出来的，比 dropout 论文（[Hinton 等，2012](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)）早了四年。

### 稀疏自编码器

**稀疏自编码器（Sparse Autoencoder）**是给隐层单元激活过程加上“稀疏”约束来防止过拟合并改善鲁棒性。它迫使模型同一时间只有少数隐层单元被激活，换句话说就是一个隐层神经元多数时候应该是没被激活的。

回想下常见[激活函数](http://cs231n.github.io/neural-networks-1/#actfun)——sigmoid，tanh，relu，leaky relu 等等。神经元的值接近 1 时被激活，接近 0 时则静默。

假设 $$l$$ 层有 $$s_l$$ 个神经元，该层的 $$j$$ 号神经元的激活函数表示为 $$a^{(l)}_j(.)$$, $$j=1, \dots, s_l$$。这个神经元的激活函数的分数 $$\hat{\rho}_j$$ 一般希望是个比较小的值 $$\rho$$，也称*稀疏性参数* ；常设 $$\rho = 0.05$$


$$
\hat{\rho}_j^{(l)} = \frac{1}{n} \sum_{i=1}^n [a_j^{(l)}(\mathbf{x}^{(i)})] \approx \rho
$$


给损失函数中加上一个惩罚项就能实现这一约束。KL 散度 $$D_\text{KL}$$ 会度量两个伯努力分布间的差异大小，其中一个均值为 $$\rho$$ 另一个均值为 $$\hat{\rho}_j^{(l)}$$，而参数 $$\beta$$ 决定了我们要对稀疏损失施加的惩罚力度。


$$
\begin{aligned}
L_\text{SAE}(\theta) 
&= L(\theta) + \beta \sum_{l=1}^L \sum_{j=1}^{s_l} D_\text{KL}(\rho \| \hat{\rho}_j^{(l)}) \\
&= L(\theta) + \beta \sum_{l=1}^L \sum_{j=1}^{s_l} \rho\log\frac{\rho}{\hat{\rho}_j^{(l)}} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j^{(l)}}
\end{aligned}
$$
![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-kl-metric-sparse-autoencoder.png)

 *图 4  均值 $$\rho=0.25$$ 和均值 $$0 \leqslant \hat{\rho} \leqslant 1$$ 的两个伯努力分布间的 KL 散度*

**稀疏 $$k$$自编码器**

稀疏 $$k$$ 自编码器（$$k$$-Sparse Autoencoder，[Makhzani and Frey, 2013](https://arxiv.org/abs/1312.5663)）是在线性激活的瓶颈层里只保留 k 个激活度最高的，以此来保证稀疏性。首先要前馈走过编码网络获取降维编码 $$\mathbf{z} = g(\mathbf{x})$$。对向量 $$\mathbf{z}$$ 的值排序，其中 k 个最大的保留其它的归零。这也可以在可调节阈值的 ReLU 层完成。于是现在就有了稀疏编码 $$\mathbf{z}’ = \text{Sparsify}(\mathbf{z})$$。计算稀疏编码的结果和损失， $$L = \|\mathbf{x} - f(\mathbf{z}') \|_2^2$$。同时注意，反向传播只在前 k 个激活了的隐层单元上进行。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-k-sparse-autoencoder.png)

*图 5  不同稀疏水平 k 下的稀疏 k 自编码器的滤镜（图片来源： [Makhzani 与 Frey, 2013](https://arxiv.org/abs/1312.5663)）*

### 可收缩自编码器

类似于稀疏自编码器，**可收缩自编码器**（**Contractive Autoencoder**，[Rifai, et al, 2011](http://www.icml-2011.org/papers/455_icmlpaper.pdf)）希望学到的表示处于一个可收缩的空间内，这样鲁棒性更好。

它在损失函数中加了一项来惩罚那些对输入太过敏感的表示，提高了对训练数据周围细微扰动的鲁棒度。敏感度的测定是结合输入，计算编码器激活的雅可比矩阵的 Frobenius 范数实现的。


$$
\|J_f(\mathbf{x})\|_F^2 = \sum_{ij} \Big( \frac{\partial h_j(\mathbf{x})}{\partial x_i} \Big)^2
$$



其中 $$h_j$$ 是压缩编码 $$\mathbf{z} = f(x)$$ 输出的一个单元。

这个惩罚项是对所有输入维度的习得编码求偏导，再求这些项的平方和得到的。作者称实践中该惩罚能生成针对低维非线性流行的表示，同时在多数正交于流行的方向上保持不变。

### VAE：变分自编码器

**变分自编码器**（**Variational Autoencoder**，[Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114)），简写为 **VAE**，其思想实际和上述所有自动编码器模型都很不一样，根植于图模型和变分贝叶斯方法。

相较于将输入映射成 *确定的* 向量，这里希望将它映射到一个分布之中。不妨将该分布记为 $$p_\theta$$，$$\theta$$ 为参数。输入数据 $$\mathbf{x}$$ 和隐向量 $$\mathbf{z}$$ 间的关系可以完全由下列三项表示：

-  先验概率 $$p_\theta(\mathbf{z})$$
-  似然概率 $$p_\theta(\mathbf{x}\vert\mathbf{z})$$
- 后验概率 $$p_\theta(\mathbf{z}\vert\mathbf{x})$$

假设我们知道该分布的实际参数 $$\theta^{*}$$，想生成和实际数据 $$\mathbf{x}^{(i)}$$ 近乎一样的样本要分两步：

1. 从先验分布 $$p_{\theta^*}(\mathbf{z})$$ 中抽出一个 $$\mathbf{z}^{(i)}$$ 
2.  从条件分布 $$p_{\theta^*}(\mathbf{x} \vert \mathbf{z} = \mathbf{z}^{(i)})$$ 中得到 $$\mathbf{x}^{(i)}$$

最优 $$\theta^{*}$$ 就是那个能最大化生成真实数据样本概率的那个参数：


$$
\theta^{*} = \arg\max_\theta \prod_{i=1}^n p_\theta(\mathbf{x}^{(i)})
$$



一般我们会用对数概率将右侧的积转换成求和形式：


$$
\theta^{*} = \arg\max_\theta \sum_{i=1}^n \log p_\theta(\mathbf{x}^{(i)})
$$



现在更新下方程便于展示数据生成过程，带上隐向量：


$$
p_\theta(\mathbf{x}^{(i)}) = \int p_\theta(\mathbf{x}^{(i)}\vert\mathbf{z}) p_\theta(\mathbf{z}) d\mathbf{z}
$$


很不幸，这种方式要计算 $$p_\theta(\mathbf{x}^{(i)})$$ 会很难，因为要计算所有 $$\mathbf{z}$$ 的值并求和计算开销会非常大。为了缩小取值空间加速搜索，我们试着引入一个新的近似函数，输出给定输入 $$\mathbf{x}$$ 时可能得到的编码 $$q_\phi(\mathbf{z}\vert\mathbf{x})$$，参数为 $$\phi$$。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VAE-graphical-model.png)

*图 6  含变分自编码器的图模型。实线表示生成分布 $$p_\theta(.)$$ 虚线表示分布 $$q_\phi (\mathbf{z}\vert\mathbf{x})$$，用来近似难搞的后验概率 $$p_\theta (\mathbf{z}\vert\mathbf{x})$$*

现在结构看起来很像自编码器了：

-  条件概率 $$p_\theta(\mathbf{x} \vert \mathbf{z})$$ 定义了一个生成模型，类似于上面讲的解码器 $$f_\theta(\mathbf{x} \vert \mathbf{z})$$。$$p_\theta(\mathbf{x} \vert \mathbf{z})$$ 也叫*概率译码器*
-  近似函数 $$q_\phi(\mathbf{z} \vert \mathbf{x})$$ 是*概率编码器* ，扮演着类似上面 $$g_\phi(\mathbf{z} \vert \mathbf{x})$$ 的角色

#### 损失函数：ELBO

我们希望预估后验 $$q_\phi(\mathbf{z}\vert\mathbf{x})$$ 非常接近实际后验 $$p_\theta(\mathbf{z}\vert\mathbf{x})$$。可以用 [Kullback-Leibler 散度](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)量化两分布间的差异。KL 散度 $$D_\text{KL}(X\|Y)$$ 表示如果用分布 Y 表示分布 X 会损失多少信息。

这里我们希望参数为 $$\phi$$ 的条件下最小化 $$D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) )$$。

但为什么用 $$D_\text{KL}(q_\phi \| p_\theta)$$ （逆 KL）而不是 $$D_\text{KL}(p_\theta \| q_\phi)$$ （正 KL）呢？ Eric Jang 在一篇有关贝叶斯变分法的[文章](https://blog.evjang.com/2016/08/variational-bayes.html)中有很好的解释，快速回顾一下：

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-forward_vs_reversed_KL.png)

*图 7  正逆 KL 散度对如何匹配两个分布有不同要求（图片来源：[Eric 博客](https://blog.evjang.com/2016/08/variational-bayes.html)）*

-  正 KL 散度：$$D_\text{KL}(P\|Q) = \mathbb{E}_{z\sim P(z)} \log\frac{P(z)}{Q(z)}$$；要保证只要 $$P(z) > 0$$，那 $$Q(z)>0$$。最优变分分布 $$q(z)$$ 必须能盖住整个 $$p(z)$$。
-  逆 KL 散度：$$D_\text{KL}(Q\|P) = \mathbb{E}_{z\sim Q(z)} \log\frac{Q(z)}{P(z)}$$；最小的 KL 散度会把 $$Q(z)$$ 压在 $$P(z)$$ 之下

现在来拓展一下方程：



$$
\begin{aligned}
& D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) & \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z} \vert \mathbf{x})} d\mathbf{z} & \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})p_\theta(\mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} & \scriptstyle{\text{; 因为 }p(z \vert x) = p(z, x) / p(x)} \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \big( \log p_\theta(\mathbf{x}) + \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} \big) d\mathbf{z} & \\
&=\log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x})\log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} & \scriptstyle{\text{; 因为 }\int q(z \vert x) dz = 1}\\
&=\log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x})\log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{x}\vert\mathbf{z})p_\theta(\mathbf{z})} d\mathbf{z} & \scriptstyle{\text{; 因为 }p(z, x) = p(x \vert z) p(z)} \\
&=\log p_\theta(\mathbf{x}) + \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z} \vert \mathbf{x})}[\log \frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z})} - \log p_\theta(\mathbf{x} \vert \mathbf{z})] &\\
&=\log p_\theta(\mathbf{x}) + D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z})) - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z}) &
\end{aligned}
$$



所以有：


$$
D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) =\log p_\theta(\mathbf{x}) + D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z})) - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z})
$$



重新分配一下方程的左右两边，


$$
\log p_\theta(\mathbf{x}) - D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) = \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z}) - D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}))
$$


方程左边就是我们在学习实际分布时的最大化目标：最大化生成真实数据的似然概率（即 $$\log p_\theta(\mathbf{x})$$），同时使实际与预估后验分布间的差异最小（$$D_\text{KL}$$ 像个正则化项）。注意就 $$q_\phi$$ 来讲 $$p_\theta(\mathbf{x})$$ 是固定的。

上式取负就是我们的损失函数了：



$$
\begin{aligned}
L_\text{VAE}(\theta, \phi) 
&= -\log p_\theta(\mathbf{x}) + D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) )\\
&= - \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) + D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}) ) \\
\theta^{*}, \phi^{*} &= \arg\min_{\theta, \phi} L_\text{VAE}
\end{aligned}
$$


变分贝叶斯法中，该损失也叫*变分下界* ，或*证据下界* 。名字中带个“下界”是因为 KL 散度总是非负的，所以 $$-L_\text{VAE}$$ 是 $$\log p_\theta (\mathbf{x})$$ 的下界。


$$
-L_\text{VAE} = \log p_\theta(\mathbf{x}) - D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) \leqslant \log p_\theta(\mathbf{x})
$$



所以损失最小化，就是生成真实样本的概率下界最大化

#### 再参数化技巧

损失函数中的期望项要用到 $$\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})$$ 生成的样本。采样是随机的所以不能反向传播梯度。为了使其可以训练人们引入了再参数化技巧：一般可以将随机变量 $$\mathbf{z}$$ 表示为一个确定变量 $$\mathbf{z} = \mathcal{T}_\phi(\mathbf{x}, \boldsymbol{\epsilon})$$，$$\boldsymbol{\epsilon}$$ 是一个辅助的独立随机变量，参数为 $$\phi$$ 的转换函数 $$\mathcal{T}_\phi$$ 会将 $$\boldsymbol{\epsilon}$$ 变成 $$\mathbf{z}$$。

比如说，一个常见的 $$q_\phi(\mathbf{z}\vert\mathbf{x})$$ 选择是带对角协方差结构的多元高斯分布：


$$
\begin{aligned}
\mathbf{z} &\sim q_\phi(\mathbf{z}\vert\mathbf{x}^{(i)}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}^{(i)}, \boldsymbol{\sigma}^{2(i)}\boldsymbol{I}) & \\
\mathbf{z} &= \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \text{, 其中 } \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I}) & \scriptstyle{\text{; 再参数化技巧}}
\end{aligned}
$$



其中 $$\odot$$ 代表按元素求积![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-reparameterization-trick.png)

*图 8  用再参数化技巧使 $$\mathbf{z}$$ 采样过程可以训练（图片来源：NIPS 2015 研讨会[幻灯片](http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf)）*

再参数化技巧对其他类型的分布也有效，不只限于高斯。用多元高斯时，通过学习分布的均值和方差， $$\mu$$ 与 $$\sigma$$，显式使用再参数化技巧使得模型可以训练，而随机性则保留在随机变量 $$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})$$ 身上。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-vae-gaussian.png)

*图 9  带多元高斯假设的变分自编码器*

### Beta-VAE

如果隐向量 $$\mathbf{z}$$ 中的每个变量只对单一生成要素敏感，而又对其他要素相对保持稳定，就说这个表示是解耦了的或是因子分解了的。解耦表示的一大好处是*良好的可解释性*，并易于泛化到各类任务上。

比如，基于人脸照片训练的模型可以捕捉到性别、肤色、发色、发长和情绪等属性，无论是否带了眼镜以及各维度上许多其他相对独立的要素存在与否。这种解耦表示对面部图像生成来讲大有用处。

β-VAE([Higgins et al., 2017](https://openreview.net/forum?id=Sy2fzU9gl)) 是变分自编码器的一个变种，格外强调对解耦态隐式要素的发掘。照 VAE 的思路，要在使生成真实数据的概率最大化的同时，令实际分布与预估后验分布的差距足够小（比一个够小的常数 $$\delta$$ 小）：


$$
\begin{aligned}
&\max_{\phi, \theta} \mathbb{E}_{\mathbf{x}\sim\mathcal{D}}[\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z})]\\
&\text{要求 } D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) < \delta
\end{aligned}
$$


可以按 [KKT 条件](https://www.cs.cmu.edu/~ggordon/10725-F12/slides/16-kkt.pdf)用拉格朗日乘子 $$\beta$$ 重写表达式。上述最优化问题只有一个不等式约束，其等价于最大化下列方程 $$\mathcal{F}(\theta, \phi, \beta)$$：


$$
\begin{aligned}
\mathcal{F}(\theta, \phi, \beta) &= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) - \beta(D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) - \delta) & \\
& = \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) - \beta D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) + \beta \delta & \\
& \geqslant \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) - \beta D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) & \scriptstyle{\text{; 因为 }\beta,\delta\geqslant 0}
\end{aligned}
$$


$$\beta$$-VAE 的损失函数为：


$$
L_\text{BETA}(\phi, \beta) = - \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) + \beta D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z}))
$$


r $$\beta$$ i

 $$L_\text{BETA}(\phi, \beta)$$  $$\mathcal{F}(\theta, \phi, \beta)$$. 

 $$\beta=1$$,  $$\beta > 1$$, $$\mathbf{z}$$. r $$\beta$$  r $$\beta$$ 

 $$\beta$$-VAE  $$\beta$$-VAE

### VQ-VAE 和 VQ-VAE-2

 $$K$$-

 $$\mathbf{e} \in \mathbb{R}^{K \times D}, i=1, \dots, K$$  $$K$$  $$\mathbf{e}_i \in \mathbb{R}^{D}, i=1, \dots, K$$. 

 $$E(\mathbf{x}) = \mathbf{z}_e$$  $$K$$  $$D(.)$$:
$$
\mathbf{z}_q(\mathbf{x}) = \text{Quantize}(E(\mathbf{x})) = \mathbf{e}_k \text{ where } k = \arg\min_i \|E(\mathbf{x}) - \mathbf{e}_i \|_2
$$
![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VQ-VAE.png)

 $$\nabla_z L$$  $$\mathbf{z}_q$$ t $$\mathbf{z}_e$$.
$$
L = \underbrace{\|\mathbf{x} - D(\mathbf{e}_k)\|_2^2}_{\textrm{reconstruction loss}} + 
\underbrace{\|\text{sg}[E(\mathbf{x})] - \mathbf{e}_k\|_2^2}_{\textrm{VQ loss}} + 
\underbrace{\beta \|E(\mathbf{x}) - \text{sg}[\mathbf{e}_k]\|_2^2}_{\textrm{commitment loss}}
$$
 $$\text{sq}[.]$$ 

 $$\mathbf{e}_i$$,  $$n_i$$  $$\{\mathbf{z}_{i,j}\}_{j=1}^{n_i}$$, $$\mathbf{e}_i$$:
$$
N_i^{(t)} = \gamma N_i^{(t-1)} + (1-\gamma)n_i^{(t)}\;\;\;
\mathbf{m}_i^{(t)} = \gamma \mathbf{m}_i^{(t-1)} + (1-\gamma)\sum_{j=1}^{n_i^{(t)}}\mathbf{z}_{i,j}^{(t)}\;\;\;
\mathbf{e}_i^{(t)} = \mathbf{m}_i^{(t)} / N_i^{(t)}
$$
 $$(t)$$ . $$N_i$$ and $$\mathbf{m}_i$$ 

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VQ-VAE-2.png)



![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VQ-VAE-2-algo.png)

### TD-VAE

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-TD-VAE-state-space.png)

s $$\mathbf{z} = (z_1, \dots, z_T)$$  $$\mathbf{x} = (x_1, \dots, x_T)$$. r $$p(z \vert x)$$  $$q(z \vert x)$$.

 $$b_t = belief(x_1, \dots, x_t) = belief(b_{t-1}, x_t)$$.  $$p(x_{t+1}, \dots, x_T \vert x_1, \dots, x_t) \approx p(x_{t+1}, \dots, x_T \vert b_t)$$. $$b_t = \text{RNN}(b_{t-1}, x_t)$$.
$$
\begin{aligned}
\log p(x) 
&\geq \log p(x) - D_\text{KL}(q(z|x)\|p(z|x)) \\
&= \mathbb{E}_{z\sim q} \log p(x|z) - D_\text{KL}(q(z|x)\|p(z)) \\
&= \mathbb{E}_{z \sim q} \log p(x|z) - \mathbb{E}_{z \sim q} \log \frac{q(z|x)}{p(z)} \\
&= \mathbb{E}_{z \sim q}[\log p(x|z) -\log q(z|x) + \log p(z)] \\
&= \mathbb{E}_{z \sim q}[\log p(x, z) -\log q(z|x)] \\
\log p(x) 
&\geq \mathbb{E}_{z \sim q}[\log p(x, z) -\log q(z|x)]
\end{aligned}
$$
 $$x_{<t}$$  $$z_t$$ and $$z_{t-1}$$, 
$$
\log p(x_t|x_{<t}) \geq \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t, z_{t-1}, z_{t}|x_{<t}) -\log q(z_{t-1}, z_t|x_{\leq t})]
$$

$$
\begin{aligned}
& \log p(x_t|x_{<t}) \\
&\geq \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t, z_{t-1}, z_{t}|x_{<t}) -\log q(z_{t-1}, z_t|x_{\leq t})] \\
&\geq \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t|\color{red}{z_{t-1}}, z_{t}, \color{red}{x_{<t}}) + \color{blue}{\log p(z_{t-1}, z_{t}|x_{<t})} -\log q(z_{t-1}, z_t|x_{\leq t})] \\
&\geq \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t|z_{t}) + \color{blue}{\log p(z_{t-1}|x_{<t})} + \color{blue}{\log p(z_{t}|z_{t-1})} - \color{green}{\log q(z_{t-1}, z_t|x_{\leq t})}] \\
&\geq \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t|z_{t}) + \log p(z_{t-1}|x_{<t}) + \log p(z_{t}|z_{t-1}) - \color{green}{\log q(z_t|x_{\leq t})} - \color{green}{\log q(z_{t-1}|z_t, x_{\leq t})}]
\end{aligned}
$$

1. $$p_D(.)$$

- $$p(x_t \mid z_t)$$
- $$p(x_t \mid z_t) \to p_D(x_t \mid z_t)$$;

2. $$p_T(.)$$

- $$p(z_t \mid z_{t-1})$$
- $$p(z_t \mid z_{t-1}) \to p_T(z_t \mid z_{t-1})$$;

3. $$p_B(.)$$

-  $$p(z_{t-1} \mid x_{<t})$$ and $$q(z_t \mid x_{\leq t})$$ 
- $$p(z_{t-1} \mid x_{<t}) \to p_B(z_{t-1} \mid b_{t-1})$$;
- $$q(z_{t} \mid x_{\leq t}) \to p_B(z_t \mid b_t)$$;

4. $$p_S(.)$$

- $$q(z_{t-1} \mid z_t, x_{\leq t})$$ c
- $$q(z_{t-1} \mid z_t, x_{\leq t}) \to  p_S(z_{t-1} \mid z_t, b_{t-1}, b_t)$$;

 $$t, t+1$$,  $$t_1 < t_2$$. 
$$
J_{t_1, t_2} = \mathbb{E}[
  \log p_D(x_{t_2}|z_{t_2}) 
  + \log p_B(z_{t_1}|b_{t_1}) 
  + \log p_T(z_{t_2}|z_{t_1}) 
  - \log p_B(z_{t_2}|b_{t_2}) 
  - \log p_S(z_{t_1}|z_{t_2}, b_{t_1}, b_{t_2})]
$$
![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-TD-VAE.png)