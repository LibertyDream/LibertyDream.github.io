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
excerpt:    自动编码器是个神经网络模型家族，旨在得到高维数据压缩后的低维隐变量。本文从基本的自编码器模型开始，回顾了若干变体，包括降噪、稀疏和可收缩自编码器，然后介绍了变分自编码器（VAE）和它的改良版 beta-VAE
---

> 编译自：From Autoencoder to Beta-VAE， [Lilian Weng](https://lilianweng.github.io/lil-log/)

自动编码器用一个中间是狭小瓶颈层的神经网络重构高维数据（对[变分自编码器](# VAE：变分自编码器)来讲可能不太准确，后面再说），这带来了一个不错的副产物——降维：瓶颈层会得到压缩降维后的隐编码。这样的低维表示可充当多类应用中（比如搜索）的嵌入向量，帮助压缩数据，或是揭示潜在的数据生成要素。

### 符号说明

| 符号                                      | 含义                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| $$\mathcal{D}$$                           | 数据集，$$\mathcal{D} = \{ \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(n)} \}$$ 有 $$n$$ 个数据样本; $$\vert\mathcal{D}\vert =n $$ |
| $$\mathbf{x}^{(i)}$$                      | 数据项，是个 $$d$$ 维向量， $$\mathbf{x}^{(i)} = [x^{(i)}_1, x^{(i)}_2, \dots, x^{(i)}_d]$$ |
| $$\mathbf{x}$$                            | 数据集中的一个样本 $$\mathbf{x} \in \mathcal{D}$$            |
| $$\mathbf{x}’$$                           | $$\mathbf{x}$$ 的重构版                                      |
| $$\tilde{\mathbf{x}}$$                    | $$\mathbf{x}$$ 的缺损版                                      |
| $$\mathbf{z}$$                            | 瓶颈层学到的凝练编码                                         |
| $$a_j^{(l)}$$                             | $$l$$ 层 $$j$$ 号神经元的激活函数                            |
| $$g_{\phi}(.)$$                           | 参数为 $$\phi$$ 的**编码**函数                               |
| $$f_{\theta}(.)$$                         | 参数为 $$\theta$$ 的**解码**函数                             |
| $$q_{\phi}(\mathbf{z}\vert\mathbf{x})$$   | 后验概率估计函数，也叫**概率编码器**                         |
| $$p_{\theta}(\mathbf{x}\vert\mathbf{z})$$ | 给定隐编码生成正确数据的可能性，也叫**概率解码器**           |

### 自编码器

自编码器（Autoencoder）是个神经网络，用无监督的方式学习一个同化函数来重构原始输入数据，在此期间将数据压缩好找到更高效、凝练的表示。该思想可追溯至 [20 世纪 80年代](https://en.wikipedia.org/wiki/Autoencoder)，之后在一篇开创性论文 [Hinton 和 Salakhutdinov, 2006](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.3788&rep=rep1&type=pdf)中被提出。

自编码器包括两个网络：

- *编码器* 网络：将原来的高维输入变成低维隐编码。输入的尺寸比输出大
- *解码器* 网络：从隐编码中将数据复原，大概率是靠逐级增大的输出层实现

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-autoencoder-architecture.png)

*图 1  自动编码器架构图*

编码网络本质上是在做[降维](https://en.wikipedia.org/wiki/Dimensionality_reduction)，和用主成分分析（PCA）或矩阵分解（MF）的目的一样。此外，自编码器为从隐编码重构数据做了显性优化。一个好的中间态表示不只是得到了隐变量，还能使整个[解压缩](https://ai.googleblog.com/2016/09/image-compression-with-neural-networks.html)过程收益。

模型包含一个参数为 $$\phi$$ 的编码函数 $$g(.)$$ 以及一个参数为 $$\theta$$ 的解码函数 $$f(.)$$。瓶颈层学到的有关输入 $$\mathbf{x}$$ 的低维编码为 $$\mathbf{z} = g_\phi(\mathbf{x})$$，重构后的输入为 $$\mathbf{x}' = f_\theta(g_\phi(\mathbf{x}))$$。

参数 $$(\theta, \phi)$$ 一起学习，目标是使重构的数据样本和原始输入一样， $$\mathbf{x} \approx f_\theta(g_\phi(\mathbf{x}))$$，换句话讲就是学个同化函数。有很多方法可以量化两个向量间的差异，比如激活函数为 sigmoid 时用的交叉熵，或简单粗暴如 MSE 损失：


$$
L_\text{AE}(\theta, \phi) = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\mathbf{x}^{(i)})))^2
$$



### 降噪自编码器

因为自编码器是在学同化函数，当网络参数比数据样本还多时就要面对“过拟合”的风险。为了避免过拟合改善鲁棒性，[Vincent 等，2008](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)在普通自编码器的基础上做了点改进，提出了**降噪自编码器（Denoising Autoencoder）**。给输入加上噪声造成一定程度的缺损，或者随机遮住一些输入向量的值， $$\tilde{\mathbf{x}} \sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}} \vert \mathbf{x})$$。然后训练模型尝试复原输入（缺损项不算）。


$$
\begin{aligned}
\tilde{\mathbf{x}}^{(i)} &\sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}}^{(i)} \vert \mathbf{x}^{(i)})\\
L_\text{DAE}(\theta, \phi) &= \frac{1}{n} \sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\tilde{\mathbf{x}}^{(i)})))^2
\end{aligned}
$$


其中 $$\mathcal{M}_\mathcal{D}$$ 是真实数据与噪声或缺损项之间的映射。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-denoising-autoencoder-architecture.png)

*图 2  降噪自编码器架构*

这种设计源自这样一种现实，即使视野中的景象有部分被遮挡或是缺损人们还是可以轻易识别出某个物体或场景。为了“复原”那部分受损输入，降噪自编码器必须发掘并掌握输入维度彼此之间的关联，这样才能推断缺失部分。

对那些高度冗余的高维输入，比如图像，模型很可能会从众多输入维度间的组合中学习复原数据而不是揪着一个维度过度拟合。这为获取 *鲁棒的* 隐编码打下了不错的基础。

噪声是由随机映射 $$\mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}} \vert \mathbf{x})$$ 控制的，不受特定类型的混淆过程束缚（遮罩噪声，高斯噪声，椒盐噪声等等），在混淆过程里加上先验知识也是可以的。原 DAE 论文的实验里，噪声是这么加的：随机抽取固定比例的输入维度强制其值为 0。听起来像 dropout 对吧？可惜降噪自编码器是 2008 年提出来的，比 dropout 论文（[Hinton 等，2012](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)）早了四年。

### 稀疏自编码器

**稀疏自编码器（Sparse Autoencoder）**是给隐层单元的激活过程加上“稀疏”约束来防止过拟合并改善鲁棒性。它迫使模型同一时间只有少数隐层单元被激活，换句话说就是一个隐层神经元多数时候应该保持“沉默”。

回想下常见[激活函数](http://cs231n.github.io/neural-networks-1/#actfun)——sigmoid，tanh，relu，leaky relu 等等。神经元的值接近 1 时被激活，接近 0 时则静默。

假设 $$l$$ 层有 $$s_l$$ 个神经元，该层 $$j$$ 号神经元的激活函数为 $$a^{(l)}_j(.)$$, $$j=1, \dots, s_l$$。这个神经元的激活程度 $$\hat{\rho}_j$$ 一般希望是个比较小的值 $$\rho$$，也称*稀疏性参数* ；常设 $$\rho = 0.05$$


$$
\hat{\rho}_j^{(l)} = \frac{1}{n} \sum_{i=1}^n [a_j^{(l)}(\mathbf{x}^{(i)})] \approx \rho
$$


给损失函数中加上一个惩罚项就能实现这种约束。KL 散度 $$D_\text{KL}$$ 可以度量两个伯努力分布间的差异大小，其中一个均值为 $$\rho$$ 另一个均值为 $$\hat{\rho}_j^{(l)}$$，而参数 $$\beta$$ 决定了我们要对稀疏损失施加的惩罚力度。


$$
\begin{aligned}
L_\text{SAE}(\theta) 
&= L(\theta) + \beta \sum_{l=1}^L \sum_{j=1}^{s_l} D_\text{KL}(\rho \| \hat{\rho}_j^{(l)}) \\
&= L(\theta) + \beta \sum_{l=1}^L \sum_{j=1}^{s_l} \rho\log\frac{\rho}{\hat{\rho}_j^{(l)}} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j^{(l)}}
\end{aligned}
$$
![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-kl-metric-sparse-autoencoder.png)

 *图 4  均值 $$\rho=0.25$$ 和均值 $$0 \leqslant \hat{\rho} \leqslant 1$$ 的两个伯努力分布间的 KL 散度*

**稀疏 $$k$$ 自编码器**

稀疏 $$k$$ 自编码器（$$k$$-Sparse Autoencoder，[Makhzani and Frey, 2013](https://arxiv.org/abs/1312.5663)）是在线性激活的瓶颈层里只保留 k 个激活度最高的，以此来保证稀疏性。首先前馈地走过编码器网络得到降维隐编码 $$\mathbf{z} = g(\mathbf{x})$$。对向量 $$\mathbf{z}$$ 的值排序，其中 k 个最大的保留其它的归零。这也可以在能调节阈值的 ReLU 层完成。于是现在就有了稀疏编码 $$\mathbf{z}’ = \text{Sparsify}(\mathbf{z})$$。计算稀疏编码的结果和损失， $$L = \|\mathbf{x} - f(\mathbf{z}') \|_2^2$$。反向传播也只在前 k 个激活了的隐层单元上进行。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-k-sparse-autoencoder.png)

*图 5  不同稀疏水平 k 下的稀疏 k 自编码器的滤镜（图片来源： [Makhzani 与 Frey, 2013](https://arxiv.org/abs/1312.5663)）*

### 可收缩自编码器

类似于稀疏自编码器，**可收缩自编码器**（**Contractive Autoencoder**，[Rifai, et al, 2011](http://www.icml-2011.org/papers/455_icmlpaper.pdf)）希望学到的表示处于一个可收缩的空间内，这样鲁棒性更好。

它在损失函数中加了一项来惩罚那些对输入太过敏感的表示，提高了对训练数据周围细微扰动的鲁棒度。敏感度的测定是结合输入，计算编码器激活的雅可比矩阵的 Frobenius 范数实现的。


$$
\|J_f(\mathbf{x})\|_F^2 = \sum_{ij} \Big( \frac{\partial h_j(\mathbf{x})}{\partial x_i} \Big)^2
$$



其中 $$h_j$$ 是压缩编码 $$\mathbf{z} = f(x)$$ 输出的一个单元。

这个惩罚项先是学习所有输入维度的编码，再求编码对输入偏导，再将这些项取平方求和。作者称实践中该惩罚能得到对低维非线性流形的表示，同时在多数正交于流形的方向上保持不变。

### VAE：变分自编码器

**变分自编码器**（**Variational Autoencoder**，[Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114)），简写为 **VAE**，其思想实际和上述所有自动编码器模型都很不一样，但与图模型和变分贝叶斯方法颇有渊源。

相较于将输入映射成 *固定的* 向量，这里希望将它映射到一个分布之中。不妨将该分布记为 $$p_\theta$$，$$\theta$$ 为参数。输入数据 $$\mathbf{x}$$ 和隐向量 $$\mathbf{z}$$ 间的关系可以完全由下列三项表示：

-  先验概率 $$p_\theta(\mathbf{z})$$
-  似然概率 $$p_\theta(\mathbf{x}\vert\mathbf{z})$$
- 后验概率 $$p_\theta(\mathbf{z}\vert\mathbf{x})$$

假设我们知道该分布的实际参数 $$\theta^{*}$$，想生成和实际数据 $$\mathbf{x}^{(i)}$$ 近乎一样的样本要分两步：

1. 从先验分布 $$p_{\theta^*}(\mathbf{z})$$ 中抽出一个 $$\mathbf{z}^{(i)}$$ 
2.  从条件分布 $$p_{\theta^*}(\mathbf{x} \vert \mathbf{z} = \mathbf{z}^{(i)})$$ 中得到 $$\mathbf{x}^{(i)}$$

最优 $$\theta^{*}$$ 能使生成真实数据样本的概率最大化：



$$
\theta^{*} = \arg\max_\theta \prod_{i=1}^n p_\theta(\mathbf{x}^{(i)})
$$



一般我们会用对数概率将右侧的积转换成求和形式：



$$
\theta^{*} = \arg\max_\theta \sum_{i=1}^n \log p_\theta(\mathbf{x}^{(i)})
$$



现在更新下方程以便于解释数据生成过程，好带上隐向量：



$$
p_\theta(\mathbf{x}^{(i)}) = \int p_\theta(\mathbf{x}^{(i)}\vert\mathbf{z}) p_\theta(\mathbf{z}) d\mathbf{z}
$$



很不幸，这种方式要计算 $$p_\theta(\mathbf{x}^{(i)})$$ 会很难，因为要计算所有 $$\mathbf{z}$$ 的值并求和，计算开销会非常大。为了缩小取值空间加速搜索，我们试着引入一个新的近似函数，其运算结果是给定输入 $$\mathbf{x}$$ 时可能得到的编码 $$q_\phi(\mathbf{z}\vert\mathbf{x})$$，参数为 $$\phi$$。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VAE-graphical-model.png)

*图 6  含变分自编码器的图模型。实线表示生成分布 $$p_\theta(.)$$ 虚线表示分布 $$q_\phi (\mathbf{z}\vert\mathbf{x})$$，用来近似难搞的后验概率 $$p_\theta (\mathbf{z}\vert\mathbf{x})$$*

现在的结构看起来就很像自编码器了：

-  条件概率 $$p_\theta(\mathbf{x} \vert \mathbf{z})$$ 定义了一个生成模型，类似于上面讲的解码器 $$f_\theta(\mathbf{x} \vert \mathbf{z})$$。$$p_\theta(\mathbf{x} \vert \mathbf{z})$$ 也叫*概率译码器*
-  近似函数 $$q_\phi(\mathbf{z} \vert \mathbf{x})$$ 是*概率编码器* ，扮演着类似上面 $$g_\phi(\mathbf{z} \vert \mathbf{x})$$ 的角色

#### 损失函数：ELBO

我们希望猜测的后验概率 $$q_\phi(\mathbf{z}\vert\mathbf{x})$$ 非常接近实际的后验概率 $$p_\theta(\mathbf{z}\vert\mathbf{x})$$。可以用 [Kullback-Leibler 散度](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)量化两分布间的差异。KL 散度 $$D_\text{KL}(X\|Y)$$ 表示如果用分布 Y 表示分布 X 会损失多少信息。

这里我们是想针对 $$\phi$$ 取 $$D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) )$$ 的最小化。

但为什么用 $$D_\text{KL}(q_\phi \| p_\theta)$$ （逆 KL）而不是 $$D_\text{KL}(p_\theta \| q_\phi)$$ （正 KL）呢？ Eric Jang 在一篇有关贝叶斯变分法的[文章](https://blog.evjang.com/2016/08/variational-bayes.html)中有很好的解释，快速回顾一下：

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-forward_vs_reversed_KL.png)

*图 7  正逆 KL 散度对如何匹配两个分布有不同要求（图片来源：[Eric 博客](https://blog.evjang.com/2016/08/variational-bayes.html)）*

-  正 KL 散度：$$D_\text{KL}(P\|Q) = \mathbb{E}_{z\sim P(z)} \log\frac{P(z)}{Q(z)}$$；要保证只要 $$P(z) > 0$$，就有 $$Q(z)>0$$。最优变分分布 $$q(z)$$ 必须能盖住整个 $$p(z)$$。
-  逆 KL 散度：$$D_\text{KL}(Q\|P) = \mathbb{E}_{z\sim Q(z)} \log\frac{Q(z)}{P(z)}$$；使 KL 散度最小化会把 $$Q(z)$$ 压在 $$P(z)$$ 之下

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

损失函数中的期望项要用到 $$\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})$$ 生成的样本。采样是随机的所以不能反向传播梯度。为了使采样也加入训练人们引入了再参数化技巧：一般可以将随机变量 $$\mathbf{z}$$ 表示为一个确定变量 $$\mathbf{z} = \mathcal{T}_\phi(\mathbf{x}, \boldsymbol{\epsilon})$$，$$\boldsymbol{\epsilon}$$ 是一个辅助的独立随机变量，参数为 $$\phi$$ 的转换函数 $$\mathcal{T}_\phi$$ 会将 $$\boldsymbol{\epsilon}$$ 变成 $$\mathbf{z}$$。

比如说，一个常见的 $$q_\phi(\mathbf{z}\vert\mathbf{x})$$ 选择是带对角协方差结构的多元高斯分布：



$$
\begin{aligned}
\mathbf{z} &\sim q_\phi(\mathbf{z}\vert\mathbf{x}^{(i)}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}^{(i)}, \boldsymbol{\sigma}^{2(i)}\boldsymbol{I}) & \\
\mathbf{z} &= \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \text{, 其中 } \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I}) & \scriptstyle{\text{; 再参数化技巧}}
\end{aligned}
$$



其中 $$\odot$$ 代表按元素求积![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-reparameterization-trick.png)

*图 8  再参数化技巧使 $$\mathbf{z}$$ 采样过程加入训练（图片来源：NIPS 2015 研讨会[幻灯片](http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf)）*

再参数化技巧对其他类型的分布也有效，不只限于高斯。用多元高斯时，显式使用再参数化技巧，通过学习分布的均值和方差，对应 $$\mu$$ 与 $$\sigma$$，使得模型可以训练，而随机性通过随机变量 $$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})$$ 得以保留。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-vae-gaussian.png)

*图 9  带多元高斯假设的变分自编码器*

### Beta-VAE

如果隐向量 $$\mathbf{z}$$ 中的每一项只对单一生成要素敏感，而又对其他要素相对保持稳定，就说这个表示已解耦或被因子分解了。解耦表示的一大好处是具有*良好的可解释性*，并易于泛化到各类任务上。

比如，基于人脸照片训练的模型可以捕捉到性别、肤色、发色、发长和情绪等属性，无论是否带了眼镜，无论在各维度上是否存在许多其他相对独立的要素。这种解耦表示对面部图像生成来讲大有用处。

β-VAE([Higgins et al., 2017](https://openreview.net/forum?id=Sy2fzU9gl)) 是变分自编码器的一个变种，专门针对解耦态隐因子的发掘。照 VAE 的思路，要在使生成真实数据的概率最大化的同时，令实际分布与猜测的后验分布间差距足够小（比一个够小的常数 $$\delta$$ 小）：



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



其中拉格朗日乘子 $$\beta$$ 是超参数。

因为对 $$L_\text{BETA}(\phi, \beta)$$ 取反就是 $$\mathcal{F}(\theta, \phi, \beta)$$ 的下界，损失最小化等价于使拉格朗日表达式最大化，也就解决了原始优化问题。

如果 $$\beta=1$$，那就是 VAE。如果 $$\beta > 1$$，意味着加强对隐层瓶颈和 $$\mathbf{z}$$ 的表示能力的限制。对于条件独立的生成因子而言，保证他们处于解耦状态就是最高效的表示了。所以 $$\beta$$ 愈大隐向量编码就越高效，解耦性就越好。与此同时，更大的 $$\beta$$ 可能会在重构质量和解耦程度之间的形成一种均衡。

[Burgess 等人(2017)](https://arxiv.org/pdf/1804.03599.pdf)结合[信息瓶颈理论](https://lilianweng.github.io/lil-log/2017/09/28/anatomize-deep-learning-with-information-theory.html)深入探讨了 $$\beta$$-VAE 的解耦问题，并进一步对 $$\beta$$-VAE 进行了改良，能更好的控制编码表达能力。

### VQ-VAE 和 VQ-VAE-2

**VQ-VAE**（“向量量化变分自编码器”；[van den Oord 等， 2017](http://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf)）模型用编码器学习离散隐变量，因为对像语言、口语、推理这些问题离散表示可能更自然些。

矢量量化是种将 $$K$$ 维向量映射为一组有限的“编码”向量的技术，处理过程和 KNN 算法十分相似。映射后欧氏距离最短的就是最优质心编码向量。

记 VQ-VAE 的潜在编码空间（也叫“编码簿”）为 $$\mathbf{e} \in \mathbb{R}^{K \times D}, i=1, \dots, K$$ ，其中 $$K$$ 是隐变量类别数，$$D$$ 是 embedding 大小。单个 embedding 向量记为 $$\mathbf{e}_i \in \mathbb{R}^{D}, i=1, \dots, K$$。

编码器的输出 $$E(\mathbf{x}) = \mathbf{z}_e$$ 会遍历最近的邻居尝试与 $$K$$ 个 embedding 向量中的某个进行匹配，匹配上的向量就是解码器 $$D(.)$$ 的输入：



$$
\mathbf{z}_q(\mathbf{x}) = \text{Quantize}(E(\mathbf{x})) = \mathbf{e}_k \text{ 其中 } k = \arg\min_i \|E(\mathbf{x}) - \mathbf{e}_i \|_2
$$



注意，应用不同离散隐变量的规格也不同；比如对文字是 1 维，图像是 2 维，视频是 3 维。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VQ-VAE.png)

*图 10  VQ-VAE 架构（图片来源：[van den Oord 等人，2017](http://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf)）*

因为 argmin() 在离散空间里无法求导，解码器输入 $$\mathbf{z}_q$$ 的梯度 $$\nabla_z L$$ 就直接复制给编码器输出 $$\mathbf{z}_e$$。除了重构损失，VQ-VAE 还在尝试优化：

- VQ 损失：embedding 空间和编码器输出间的 L2 误差
- 交付损失：一种鼓励编码器输出向 embedding 空间靠近的措施，也是为了防止过于频繁地从一个编码向量跳向另一个。


$$
L = \underbrace{\|\mathbf{x} - D(\mathbf{e}_k)\|_2^2}_{\textrm{重构损失}} + 
\underbrace{\|\text{sg}[E(\mathbf{x})] - \mathbf{e}_k\|_2^2}_{\textrm{VQ 损失}} + 
\underbrace{\beta \|E(\mathbf{x}) - \text{sg}[\mathbf{e}_k]\|_2^2}_{\textrm{交付损失}}
$$


其中 $$\text{sg}[.]$$ 是 `stop_gradient` (停止梯度) 算符。

编码簿中的 embedding 向量是靠 EMA（指数移动均值）进行更新。给定一个编码向量 $$\mathbf{e}_i$$，假设编码器输出 $$n_i$$ 个向量，对 $$\mathbf{e}_i$$ 的量化结果就是 $$\{\mathbf{z}_{i,j}\}_{j=1}^{n_i}$$。



$$
N_i^{(t)} = \gamma N_i^{(t-1)} + (1-\gamma)n_i^{(t)}\\
\mathbf{m}_i^{(t)} = \gamma \mathbf{m}_i^{(t-1)} + (1-\gamma)\sum_{j=1}^{n_i^{(t)}}\mathbf{z}_{i,j}^{(t)}\\
\mathbf{e}_i^{(t)} =\frac{\mathbf{m}_i^{(t)}}{N_i^{(t)}}
$$



其中 $$(t)$$ 是指按时间顺序进行批处理。$$N_i$$ 和 $$\mathbf{m}_i$$ 分别计算向量个数和容积。

VQ-VAE-2 ([Ali Razavi 等， 2019](https://arxiv.org/abs/1906.00446))  是结合自注意自回归模型的二级 VQ-VAE。

1. 1 阶段是**训练分级 VQ-VAE**：设计分级隐变量的初衷是从整体信息（即物体形状）中分离出局部模式（即基本结构）。较大的底层编码簿的训练以较小的顶层编码为前提，所以不必什么都从头学。
2. 2 阶段**基于隐式离散编码簿学习先验分布**，好从中采样并生成图像。这样解码器可以从与正在训练的分布相似的分布中得到输入向量。为了学习先验分布，还用上了强大的带多头自注意力层的自回归模型（像 [PixelSNAIL; Chen et al 2017](https://arxiv.org/abs/1712.09763)）

考虑到 VQ-VAE-2 靠简单层级设置下的离散隐变量，取得了那样的图像生成效果，还是蛮令人惊奇的。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VQ-VAE-2.png)

*图 11  分级 VQ-VAE 架构与多阶图像生成（图片来源： [Ali Razavi 等，2019](https://arxiv.org/abs/1906.00446)）*

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VQ-VAE-2-algo.png)

*图 12  VQ-VAE-2 算法（图片来源：[Ali Razavi 等，2019](https://arxiv.org/abs/1906.00446)）*

### TD-VAE

**TD-VAE**（“时差 VAE”；[Gregor 等, 2019](https://arxiv.org/abs/1806.03107)）面向序列化数据，主要靠三点，下面细讲。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-TD-VAE-state-space.png)

*图 13  将状态空间模型作为马尔可夫链模型*

**（1）状态空间模型**

（隐式）状态空间模型中，一系列未观测到的隐状态 $$\mathbf{z} = (z_1, \dots, z_T)$$ 决定了观测状态 $$\mathbf{x} = (x_1, \dots, x_T)$$。图 13 中马尔可夫链模型的每一时刻都能以类似图 6 中的方式训练，而难搞的后验概率 $$p(z \vert x)$$ 用函数 $$q(z \vert x)$$ 来近似。

**（2）信仰状态**

代理人要学会对过往所有状态进行编码以推断未来，这被称为*信仰状态*，$$b_t = belief(x_1, \dots, x_t) = belief(b_{t-1}, x_t)$$。由此，以过往状态为条件未来的状态分布可以写为 $$p(x_{t+1}, \dots, x_T \vert x_1, \dots, x_t) \approx p(x_{t+1}, \dots, x_T \vert b_t)$$。循环策略里的隐态用作 TD-VAE 里代理的信仰状态。所以有 $$b_t = \text{RNN}(b_{t-1}, x_t)$$

**（3）跳跃预测**

进一步地，代理人应该能基于目前已有信息去想象遥远的未来，也就是说能进行跳跃式的预测，提前几步预测到未来之事。

想一下[上面](# 损失函数：ELBO)讲到的方差下界：



$$
\begin{aligned}
\log p(x) 
&\geqslant \log p(x) - D_\text{KL}(q(z|x)\|p(z|x)) \\
&= \mathbb{E}_{z\sim q} \log p(x|z) - D_\text{KL}(q(z|x)\|p(z)) \\
&= \mathbb{E}_{z \sim q} \log p(x|z) - \mathbb{E}_{z \sim q} \log \frac{q(z|x)}{p(z)} \\
&= \mathbb{E}_{z \sim q}[\log p(x|z) -\log q(z|x) + \log p(z)] \\
&= \mathbb{E}_{z \sim q}[\log p(x, z) -\log q(z|x)] \\
\log p(x) 
&\geqslant \mathbb{E}_{z \sim q}[\log p(x, z) -\log q(z|x)]
\end{aligned}
$$



基于过往全部状态 $$x_{<t}$$ 和两个隐变量 $$z_t$$、$$z_{t-1}$$， 对状态 $$x_t$$ 的分布建模，考虑当前和上一步的情况；


$$
\log p(x_t|x_{<t}) \geqslant \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t, z_{t-1}, z_{t}|x_{<t}) -\log q(z_{t-1}, z_t|x_{\leqslant t})]
$$



将方程展开：


$$
\begin{aligned}
& \log p(x_t|x_{<t}) \\
&\geqslant \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t, z_{t-1}, z_{t}|x_{<t}) -\log q(z_{t-1}, z_t|x_{\leqslant t})] \\
&\geqslant \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t|\color{red}{z_{t-1}},\color{black}{z_{t}}, \color{red}{x_{<t}}) + \color{blue}{\log p(z_{t-1}, z_{t}|x_{<t})} \color{black}{-\log q(z_{t-1}, z_t|x_{\leqslant t})]} \\
&\geqslant \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t|z_{t}) + \color{blue}{\log p(z_{t-1}|x_{<t})} + \color{blue}{\log p(z_{t}|z_{t-1})} - \color{green}{\log q(z_{t-1}, z_t|x_{\leqslant t})}] \\
&\geqslant \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t|z_{t}) + \log p(z_{t-1}|x_{<t}) + \log p(z_{t}|z_{t-1}) - \color{green}{\log q(z_t|x_{\leqslant t})} - \color{green}{\log q(z_{t-1}|z_t, x_{\leqslant t})}]
\end{aligned}
$$



注意两点：

- <span style='color: red;'>红色</span>项按马尔可夫假设可以忽略
- <span style='color: blue;'>蓝色</span>项可以按马尔克夫假设展开
- 作为平滑分布，<span style='color: green;'>绿色</span>项拓展为包括对上一步预测回溯的形式

更确切地讲，有四类分布要学：

1. **解码器**分布 $$p_D(.)$$

- $$p(x_t \mid z_t)$$ 是一般意义下的编码器
- $$p(x_t \mid z_t) \to p_D(x_t \mid z_t)$$

2. **转换**分布 $$p_T(.)$$

- $$p(z_t \mid z_{t-1})$$ 表示隐变量间的顺序依赖关系
- $$p(z_t \mid z_{t-1}) \to p_T(z_t \mid z_{t-1})$$;

3. **信念**分布 $$p_B(.)$$

-  $$p(z_{t-1} \mid x_{<t})$$ 和 $$q(z_t \mid x_{\leqslant t})$$ 都可以用信念状态预测隐变量
- $$p(z_{t-1} \mid x_{<t}) \to p_B(z_{t-1} \mid b_{t-1})$$;
- $$q(z_{t} \mid x_{\leqslant t}) \to p_B(z_t \mid b_t)$$;

4. **平滑**分布 $$p_S(.)$$

- 回溯平滑项 $$q(z_{t-1} \mid z_t, x_{\leqslant t})$$ 也可以重写成依赖信念状态的形式
- $$q(z_{t-1} \mid z_t, x_{\leqslant t}) \to  p_S(z_{t-1} \mid z_t, b_{t-1}, b_t)$$;

基于跳跃预测的思想，连续的 ELBO 不仅要处理 $$t, t+1$$ 的情况，还要处理有点距离的时间戳  $$t_1 < t_2$$ 下的情景。 最终要最大化的 TD-VAE 目标函数为：


$$
J_{t_1, t_2} = \mathbb{E}[
  \log p_D(x_{t_2}|z_{t_2}) 
  + \log p_B(z_{t_1}|b_{t_1}) 
  + \log p_T(z_{t_2}|z_{t_1}) 
  - \log p_B(z_{t_2}|b_{t_2}) 
  - \log p_S(z_{t_1}|z_{t_2}, b_{t_1}, b_{t_2})]
$$


![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-TD-VAE.png)

*图 14  TD-VAE 架构概览（图片来源：[TD-VAE 论文](https://arxiv.org/abs/1806.03107)）*