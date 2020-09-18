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
excerpt:    
---

> 编译自：From Autoencoder to Beta-VAE， [Lilian Weng](https://lilianweng.github.io/lil-log/)

### 符号说明

| Symbol                                    | Mean                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| $$\mathcal{D}$$                           | The dataset, $$\mathcal{D} = \{ \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(n)} \}$$, contains $$n$$ data samples; $$\vert\mathcal{D}\vert =n $$. |
| $$\mathbf{x}^{(i)}$$                      | Each data point is a vector of $$d$$ dimensions, $$\mathbf{x}^{(i)} = [x^{(i)}_1, x^{(i)}_2, \dots, x^{(i)}_d]$$. |
| $$\mathbf{x}$$                            | One data sample from the dataset, $$\mathbf{x} \in \mathcal{D}$$. |
| $$\mathbf{x}’$$                           | The reconstructed version of $$\mathbf{x}$$.                 |
| $$\tilde{\mathbf{x}}$$                    | The corrupted version of $$\mathbf{x}$$.                     |
| $$\mathbf{z}$$                            | The compressed code learned in the bottleneck layer.         |
| $$a_j^{(l)}$$                             | The activation function for the $$j$$-th neuron in the $$l$$-th hidden layer. |
| $$g_{\phi}(.)$$                           | The **encoding** function parameterized by $$\phi$$.         |
| $$f_{\theta}(.)$$                         | The **decoding** function parameterized by $$\theta$$.       |
| $$q_{\phi}(\mathbf{z}\vert\mathbf{x})$$   | Estimated posterior probability function, also known as **probabilistic encoder**. |
| $$p_{\theta}(\mathbf{x}\vert\mathbf{z})$$ | Likelihood of generating true data sample given the latent code, also known as **probabilistic decoder**. |
| ----------                                | ----------                                                   |

### 自编码器

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-autoencoder-architecture.png)

### 降噪自编码器

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-denoising-autoencoder-architecture.png)

### 稀疏自编码器

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-kl-metric-sparse-autoencoder.png)



![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-k-sparse-autoencoder.png)

### 收缩自编码器

### VAE：变分自编码器

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VAE-graphical-model.png)

#### 损失函数：ELBO

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-forward_vs_reversed_KL.png)

#### 再参数化技巧

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-reparameterization-trick.png)



![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-vae-gaussian.png)

### Beta-VAE

### VQ-VAE 和 VQ-VAE-2

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VQ-VAE.png)



![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VQ-VAE-2.png)



![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VQ-VAE-2-algo.png)

### TD-VAE

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-TD-VAE-state-space.png)



![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-TD-VAE.png)