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