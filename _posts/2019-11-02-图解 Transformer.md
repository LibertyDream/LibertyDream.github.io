---
layout:		post
title:  	图解 Transformer
subtitle:   编码解码的故事
date:       2019-11-02
author:     一轩明月
header-img: img/post-bg-hacker.jpg
catalog: 	 true
tags:
    - NLP
    - Transformer
excerpt:    图形化展示编码-解码（encoder-decoder）架构，逐步拆解编码器内部构造，剖析解码器内部原理。
---

> 文章编译自：
>
>  http://jalammar.github.io/illustrated-transformer/ 
>
> 博客内含有数学公式，如果使用 Chrome 浏览器观看的话，可以添加这个[插件](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)

Transformer 模型出自论文[Attention is All You Need](https://arxiv.org/abs/1706.03762) ，**借助注意力机制实现并行训练加速**。特定任务上的表现优于谷歌的神经机翻译模型（the Google Neural Machine Translation model）。

### 顶层视角

不妨先将模型视作黑箱，机器翻译场景下，其作用是将一种语言的文本翻译成另一种语言

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_the_transformer.png)

掀起盖头一角，会看到其内部有编码、解码和两者间的传递共三个部分

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_encoders_decoders.png)

编码部分由一组编码器构成（论文中是 6 个一组叠放在一起——选择 6 不是玄学可以试着来）。解码部分是同样数量样式的一组解码器。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_encoder_decoder_stack.png)

编码器结构一致但不共享参数。每一个可以被继续细分为两个子结构。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_Transformer_encoder.png)

输入的内容首先通过自关注层（Self-Attention），其使得编码器在编码某个单词时可以参照句子中的其他词。后面会对此进一步分析。

自关注层的输出会传给一个前馈神经网络，所有位置用的都是同一张前馈网络。

解码器同样有这两层，只是中间多加了一层注意力好让解码器参考输入序列中的相关内容。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_Transformer_decoder.png)

### 带入张量再来看

我们已经看到了模型的大体情况，接下来看一下不同的向量（张量）是怎样在二者间流动从输入变输出的。

像 NLP 通用套路一样，先用嵌入算法（word2vec，cbow）把每个输入的单词转换成向量，这里向量维数是 512，用方块表示。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_embeddings.png)

嵌入（embedding）只是在编码器的最底层。经抽象所有编码器接收到的都是由 512 维向量组成的列表，最底层的收到的是单词嵌入表示，其他编码器收到的则是下层编码器的输出。列表大小是手动设置的超参数，一般取训练语料中的最大句子长度。

经嵌入转换后，每个词都要经过编码器中的两个组件。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_encoder_with_tensors.png)

这里开始就接触到了 Transformer 的关键，各个位置的单词通过编码器的流程是独立的。在自关注层中这些路径中还存有依赖，但在前馈神经网络中就没有了，所以在前馈网络处就能够并行加速了。

接下来我们用一个短句作为例子，看看编码器中各个部分到底发生了什么

### 编码解密

如上文所述，编码器接收向量列表作为输入，通过自关注层的处理传入前馈神经网络，再传递给上级编码器。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_encoder_with_tensors_2.png)

不要被自关注、self-attention 这样的词唬住，好像谁都很熟一样。看看它做了什么吧。

#### 顶层视角下的自关注

假定我们要翻译的语句是 ”The animal didn't cross the street because it was too tired"。那么这里的 “ it”是指代什么呢？街道还是动物？对人来说这太简单了，但对机器来讲不是。

模型处理 “it” 时，自关注层会将其关联到“animal”上。具体地，模型处理每一个词（输入序列的每一个位置）时，自关注层使其可以参看输入序列的其他位置以获得对当前位置更好的编码效果。

如果对 RNN 比较了解，想一下是怎样通过隐态将已处理的单词/向量和当前处理内容联系在一起的。自关注就是 Transformer 模型所使用的类似技术，将其他相关单词的“理解”带入当前处理的单词上来。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_self-attention_visualization.png)

图中所示的是第五个编码器（顶层编码器）处理“it”时的场景，注意力聚焦于“The Animal”上并将其部分表示带入了“it”编码内容当中。你可以看看  [Tensor2Tensor notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) ，交互式运行体验一下。

#### 自关注的细节

先来看一下如何使用向量计算自关注量，然后再分析实际中如何通过矩阵运算实现该效果。

**第一步**，根据每个编码器的输入向量（这里是每个词的嵌入表示）构建三个向量，分别是查询向量（Query）、键向量（Key）和值向量（Value）。构建方式是将嵌入向量和三个训练过程中训练好的矩阵相乘。

注意这些向量维度比嵌入向量更低，前者 64 维，后者 512 维。当然不是必须要更小，这里主要是为了多端注意力计算稳定而做出的架构选择。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_self_attention_vectors.png)

$$x_1$$ 和 $$W^Q$$ 相乘得到 $$q_1$$，查询向量和单词相关，如此反复最终得到输入序列各个位置的“query”，“key”和“value”三个映射。那这三个向量又是什么呢？

它们是对输入向量的进一步抽象，有助于计算和考虑注意力。看了下文注意力的计算方法后，你会对这三个向量的角色有清晰的认识。

**第二步**，计算自关注得分。以第一个词“Thinking"为例，输入序列的每个词分别和其求得分。分数决定了在编码某一位置时要多大程度上参考输入序列其他部分内容。

计算方式是将查询向量和各个单词向量的键向量做点积。所以如果要计算 $$\#1$$ 位置处的自关注值，就要分别求 $$q_1 \cdot k_1$$，$$q_1 \cdot k_2$$，$$q_1 \cdot k_3 \ldots$$ 

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_self_attention_score.png)

**第三、第四步**是将计算结果除以 8（论文中键向量维数——64的平方根。这会使梯度更稳定，可能是其他值，这里只是默认值），对结果做 softmax 运算。softmax 会使分数限制在 (0,1) 之间且总和为1，大的更大，小的更小，彼此更易于分隔。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_self-attention_softmax.png)

softmax 得分决定了每个词对该位置的参照度。显然该位置的单词自身的参照度最高，但在参与和该词相关内容的计算的时候会很有用。

**第五步**，将 softmax 得分与值向量相乘，准备求和。直观理解就是关注那些我们要关注的，忽视那些不相关的（通过乘一个很小的数，比如 0.001）

**第六步**，对加权后的向量求和。这样就得到了该位置的自关注层运算结果（对第一个词）。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_self-attention-output.png)

这样就完成了自关注运算，可以将运算结果送入前馈神经网络中去。但实际操作上会使用矩阵计算进行加速。

#### 自关注的矩阵运算

**第一步**要计算查询、键和值三个矩阵。将嵌入向量组成一个矩阵 $$X$$，然后与我们训练好的权重矩阵相乘（$$W^Q,W^K,W^V$$）

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_self-attention-matrix-calculation.png)

$$X$$ 矩阵中的每一行对应输入序列中的一个词。这里再一次看到嵌入向量和 q/k/v 向量的大小差异。

**最后**，因为是矩阵运算，可以将第二步到第六步浓缩成一个式子，以此求取自关注层的计算结果。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_self-attention-matrix-calculation-2.png)

#### 多端并发

论文中通过引入多端注意力（multi-headed attention）机制进一步优化自关注层。其有两种方式提高注意力层的表现：

1. 让模型能关注不同位置。诚然上述例子中 $$z_1$$ 中包含部分其他内容的编码信息，但实际上还是被自身支配。在翻译这句 “The animal didn’t cross the street because it was too tired” 的时候，如果我们知道“it”指代什么会很有用。
2. 使关注层具有多重“表示子空间”。正如将要看到的，多端注意力下我们不只有一个，而是一组 Query/Key/Value 权重矩阵的集合（Transformer 模型有 8 端，所以每个编码器/解码器会有 8 套候选集）。其中每一套都被随机初始化，经训练后每个集合都可以将输入的嵌入向量（或下层传来的向量）映射到不同表示子空间当中。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_attention_heads_qkv.png)

跳出来看自关注运算，就是 8 套矩阵 8 组运算，得到 8 个 $$Z$$ 矩阵

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_attention_heads_z.png)

这带来了点小麻烦。前馈网络层无法接收 8 个矩阵——应该是一个矩阵，一行代表一个词。所以我们需要一种方法将 8 个浓缩成一个

对此，我们将 8 个数组合并然后与另一个权重矩阵 $$W^O$$ 相乘。

 ![img](http://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png) 

这就是多端自关注的全部了。真的就是一堆矩阵。全部放入一张图里结果如下

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_multi-headed_self-attention-recap.png)

既然已经触及注意力端的核心，不妨再来看一下之前的例子，看看编码“it”的时候不同端都在关注什么

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_self-attention_visualization_2.png)

上方 8 个色块代表 8 端注意力，上图显示了 2 端。其中一端聚焦在 “the animal”上，另一个则比较重视“tired”，某种意义上说，模型对“it”的表示夹杂了部分“animal”和“tired”的内容。

如果将各端都加进去，事情就变得难以解释了

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_self-attention_visualization_3.png)

#### 使用位置编码表示序列顺序

目前为止我们讲到的模型只关注了各位置上的内容，忽视了位置间的顺序。为了补上这一点，Transformer 给每个输入的嵌入向量加上了一个额外向量。

这些向量遵循模型学到的特定模式，有助于确认每个词的位置或者序列上两个不同词间的距离。浅在想法是这些向量间距离信息可能有助于嵌入到 Q/K/V 向量的映射和求点积的过程，所以要给嵌入向量加上。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_positional_encoding_vectors.png)

假设嵌入向量是四维的，实际位置编码会像这样：

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_positional_encoding_example.png)

这个模式会是什么样的呢？下图中每一行对应一个向量的位置编码，所以第一行代表我们向输入序列中第一个词添加的位置编码。每一行有 512 个值，每个值介于 -1 到 1 之间。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_positional_encoding_large_example.png)

可以看到似乎图像从中间分隔成了两半，这是因为左边的值是通过 sin 函数生成的，右边的是通过 cos 函数生成的。合在一起构成了位置编码向量。

具体计算公式参加论文 3.5 节，生成位置编码的代码参见 [`get_timing_signal_1d()`](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py) 。这当然不是唯一位置编码方法，但其一大优点是可以缩放到任意序列长度（比如要我们的模型翻译一个比任何现存语料中的句子都长的序列）。

#### 残差

进一步之前还有一点小细节要说明，编码器中各子层（自关注，前馈网络）都会有一个残差连接过程，并接一步层归一化操作（ [layer-normalization](https://arxiv.org/abs/1607.06450) ，有助于缩短训练耗时）。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_resideual_layer_norm.png)

如果将向量、自关注运算及层标准化图像化展示出来，效果如下：

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_resideual_layer_norm_2.png)

解码器中的子层同样如此，如果考虑一个双叠加态编码器和解码器的 Transformer，会是下面的样子：

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_resideual_layer_norm_3.png)

### 解码器解密

既然编码器端已经涵盖了绝大部分概念，基本上解码器怎么工作也就清楚了。但一起来看一下他们是怎么一同工作的。

首先由编码器开始处理输入序列，顶层编码器的输出转换成一组 K，V注意力向量，为解码器中“encoder-decoder attention”层使用。该层使解码器重点关注输入序列中的恰当位置。每一次输出一个元素到输出序列中（这里是英文译文）。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_decoding_1.gif)

之后不断重复这一过程直到接收到特定信号指示解码器已经完成输出。每一步的输出都会作为输入传给下一步翻译过程中的底层解码器，解码器的输出结果也和编码器一样向上传递。和处理编码器输入一样的套路，都要嵌入并添加位置编码。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_decoding_2.gif)

解码器中的自关注层相较于编码器中的又有些不同：

解码器中的自关注层只允许参考已经翻译的部分，在 softmax 之前罩住将要被翻译的位置（将其值设为负无穷）。

“Encoder-Decoder Attention” 层工作机理很像多端自注意力，只是它的查询矩阵是根据下层传来内容计算所得，键值矩阵是编码器组传来的。

### 最后的线性层与 softmax 层

解码器组会输出浮点值构成的向量，那又该怎么将其转换成单词呢？这就是最后的线性层及其后面的 Softmax 层的工作了。

线性层就是简单的全连接层，将解码器组得到的结果映射为很大很大的向量，称为 logits 向量。

假设我们的模型从语料中学到了 10,000 个不同的英文单词（输出表），对应的 logits 向量就有 10,000 个单元宽，每个单元对应一个单词得分。这是解释模型输出的依据。

softmax 层将得分转换为概率（全为正，总和为1）。选择最高概率的单元，与之相关的单词就是这一时步的输出结果。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_decoder_output_softmax.png)

### 回顾

目前我们已经走完了训练 Transformer 的整个前向传播过程，迅速瞥一眼模型训练的直观想法。

训练时，未训练模型走过相同的前向传播过程，因为是在打标训练集上训练，所以可以和标准答案作对照。为了展示这一点，假设我们的输出词表只有 6 个词（"a","i","thanks","student",和"\<eos>"(序列结束标记的简写)），输出词表是在模型训练开始前的预处理阶段构建好的。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_vocabulary.png)

定义好输出词表后，就能用同样大小的向量来表示词表中的每一个词，这也被称为独热编码。比如，”am“就能用下面的向量表示：

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_one-hot-vocabulary-example.png)

到这我们来讨论一下模型损失函数。这是训练阶段优化模型的依据，以其获得良好泛化能力的模型

### 损失函数

假设我们正在训练模型，当前是第一步，将"merci"翻译成"thanks"。也就是说希望结果的概率分布指向“thanks”，可模型还没训练所以当下没戏。

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_transformer_logits_output_and_label.png)

模型参数是随机初始化的，未训练的模型对每个单元/单词计算得到任意值。将其与实际输出对比，调整权重通过反向传播使预测结果尽量接近期望结果。

概率分布间怎么比较？这里只是简单作差。更多内容可以参看[交叉熵](https://colah.github.io/posts/2015-09-Visual-Information/)和 [KL 散度](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)。

注意这里是被过分简化了的样例。实际中的句子都会长些，比如输入 “je suis étudiant” ，期望结果是”i am a student“。这意味着我们希望模型输出满足这样的概率分布：

- 每个概率分布由词表大小的向量表示（例子中是 6，实际上可能是 3000 或 10000）
- 第一个概率分布的最高概率指向”i“
- 第二个概率分布的最高概率指向”am“
- 以此类推，直到第五个输出的概率分布指向”`<end of sentence>`“标识，该标识在 10,000 个元素的词表中也有相应表示

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_output_target_probability_distributions.png)

在足够大的数据集上训练足够的时长后，我们希望得到下面这样的概率分布

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/2019-11-02_output_trained_model_probability_distributions.png)

因为每次得到一个输出，假定模型选择了概率最高的词后扔掉了其它内容。有种这样的方法叫贪婪解码。另一种方式是保留，比方说最大概率的 2 个词，之后就要跑两遍模型：一次假定首先输出的是 ’I‘ 所在的位置，另一次假定首先输出的是’me‘所在的位置，无论哪种因为考虑了两个位置 $$\#1,\#2$$，误差都更小。对 $$\#2$$和$$\#3\ldots$$重复这一过程。这种方法叫”光束搜索（beam search）“，例子中光束大小为 2（因为我们在计算  $$\#1,\#2$$处的光束后做了比较），最大光束也是2（因为我们要保留2个词）。这些都是实验中可以调的超参数