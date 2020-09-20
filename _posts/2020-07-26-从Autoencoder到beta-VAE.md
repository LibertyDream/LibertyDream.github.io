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

 $$g(.)$$  $$f(.)$$  $$\theta$$. $$\mathbf{x}$$  $$\mathbf{z} = $$  $$\mathbf{x}' = f_\theta(g_\phi(\mathbf{x}))$$.

 $$(\theta, \phi)$$  $$\mathbf{x} \approx f_\theta(g_\phi(\mathbf{x}))$$, 
$$
L_\text{AE}(\theta, \phi) = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\mathbf{x}^{(i)})))^2
$$

### 降噪自编码器

 $$\tilde{\mathbf{x}} \sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}} \vert \mathbf{x})$$. 
$$
\begin{aligned}
\tilde{\mathbf{x}}^{(i)} &\sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}}^{(i)} \vert \mathbf{x}^{(i)})\\
L_\text{DAE}(\theta, \phi) &= \frac{1}{n} \sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\tilde{\mathbf{x}}^{(i)})))^2
\end{aligned}
$$
 $$\mathcal{M}_\mathcal{D}$$ 

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-denoising-autoencoder-architecture.png)

 $$\mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}} \vert \mathbf{x})$$,

### 稀疏自编码器

 $$a^{(l)}_j(.)$$, $$j=1, \dots, s_l$$.  $$\hat{\rho}_j$$ r $$\rho$$, $$\rho = 0.05$$.
$$
\hat{\rho}_j^{(l)} = \frac{1}{n} \sum_{i=1}^n [a_j^{(l)}(\mathbf{x}^{(i)})] \approx \rho
$$
 $$D_\text{KL}$$  $$\rho$$  $$\hat{\rho}_j^{(l)}$$. $$\beta$$ 
$$
\begin{aligned}
L_\text{SAE}(\theta) 
&= L(\theta) + \beta \sum_{l=1}^L \sum_{j=1}^{s_l} D_\text{KL}(\rho \| \hat{\rho}_j^{(l)}) \\
&= L(\theta) + \beta \sum_{l=1}^L \sum_{j=1}^{s_l} \rho\log\frac{\rho}{\hat{\rho}_j^{(l)}} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j^{(l)}}
\end{aligned}
$$
![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-kl-metric-sparse-autoencoder.png)

 $$\rho=0.25$$ $$0 \leq \hat{\rho} \leq 1$$

**$$k$$-Sparse Autoencoder**

 $$k$$-Sparse

 $$\mathbf{z} = g(\mathbf{x})$$.

 $$\mathbf{z}$$.  $$\mathbf{z}’ = \text{Sparsify}(\mathbf{z})$$.

 $$L = \|\mathbf{x} - f(\mathbf{z}') \|_2^2$$.

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-k-sparse-autoencoder.png)

### 收缩自编码器

$$
\|J_f(\mathbf{x})\|_F^2 = \sum_{ij} \Big( \frac{\partial h_j(\mathbf{x})}{\partial x_i} \Big)^2
$$

 $$h_j$$  $$\mathbf{z} = f(x)$$. 

### VAE：变分自编码器

 $$p_\theta$$  $$\theta$$.   $$\mathbf{x}$$  $$\mathbf{z}$$ 

-  $$p_\theta(\mathbf{z})$$
-  $$p_\theta(\mathbf{x}\vert\mathbf{z})$$
- r $$p_\theta(\mathbf{z}\vert\mathbf{x})$$

 $$\theta^{*}$$  $$\mathbf{x}^{(i)}$$

1. $$\mathbf{z}^{(i)}$$  $$p_{\theta^*}(\mathbf{z})$$. 
2.  $$\mathbf{x}^{(i)}$$  $$p_{\theta^*}(\mathbf{x} \vert \mathbf{z} = \mathbf{z}^{(i)})$$.

 $$\theta^{*}$$ 
$$
\theta^{*} = \arg\max_\theta \prod_{i=1}^n p_\theta(\mathbf{x}^{(i)})
$$

$$
\theta^{*} = \arg\max_\theta \sum_{i=1}^n \log p_\theta(\mathbf{x}^{(i)})
$$

$$
p_\theta(\mathbf{x}^{(i)}) = \int p_\theta(\mathbf{x}^{(i)}\vert\mathbf{z}) p_\theta(\mathbf{z}) d\mathbf{z}
$$
 $$p_\theta(\mathbf{x}^{(i)})$$  $$\mathbf{z}$$  $$\mathbf{x}$$, $$q_\phi(\mathbf{z}\vert\mathbf{x})$$,  $$\phi$$.

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-VAE-graphical-model.png)

 $$p_\theta(.)$$ $$q_\phi (\mathbf{z}\vert\mathbf{x})$$ $$p_\theta (\mathbf{z}\vert\mathbf{x})$$.*

-  $$p_\theta(\mathbf{x} \vert \mathbf{z})$$  $$f_\theta(\mathbf{x} \vert \mathbf{z})$$   $$p_\theta(\mathbf{x} \vert \mathbf{z})$$ 
-  $$q_\phi(\mathbf{z} \vert \mathbf{x})$$   $$g_\phi(\mathbf{z} \vert \mathbf{x})$$ 

#### 损失函数：ELBO

 $$q_\phi(\mathbf{z}\vert\mathbf{x})$$   $$p_\theta(\mathbf{z}\vert\mathbf{x})$$.   $$D_\text{KL}(X\|Y)$$ 

 $$D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) )$$   $$\phi$$.

 $$D_\text{KL}(q_\phi \| p_\theta)$$  $$D_\text{KL}(p_\theta \| q_\phi)$$ 

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-forward_vs_reversed_KL.png)

-  $$D_\text{KL}(P\|Q) = \mathbb{E}_{z\sim P(z)} \log\frac{P(z)}{Q(z)}$$;  $$q(z)$$  $$p(z)$$.
-  $$D_\text{KL}(Q\|P) = \mathbb{E}_{z\sim Q(z)} \log\frac{Q(z)}{P(z)}$$;  $$Q(z)$$ under $$P(z)$$.


$$
\begin{aligned}
& D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) & \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z} \vert \mathbf{x})} d\mathbf{z} & \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})p_\theta(\mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} & \scriptstyle{\text{; Because }p(z \vert x) = p(z, x) / p(x)} \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \big( \log p_\theta(\mathbf{x}) + \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} \big) d\mathbf{z} & \\
&=\log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x})\log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} & \scriptstyle{\text{; Because }\int q(z \vert x) dz = 1}\\
&=\log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x})\log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{x}\vert\mathbf{z})p_\theta(\mathbf{z})} d\mathbf{z} & \scriptstyle{\text{; Because }p(z, x) = p(x \vert z) p(z)} \\
&=\log p_\theta(\mathbf{x}) + \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z} \vert \mathbf{x})}[\log \frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z})} - \log p_\theta(\mathbf{x} \vert \mathbf{z})] &\\
&=\log p_\theta(\mathbf{x}) + D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z})) - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z}) &
\end{aligned}
$$

$$
D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) =\log p_\theta(\mathbf{x}) + D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z})) - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z})
$$

$$
\log p_\theta(\mathbf{x}) - D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) = \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z}) - D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}))
$$
 $$\log p_\theta(\mathbf{x})$$)   $$D_\text{KL}$$  $$p_\theta(\mathbf{x})$$  $$q_\phi$$.


$$
\begin{aligned}
L_\text{VAE}(\theta, \phi) 
&= -\log p_\theta(\mathbf{x}) + D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) )\\
&= - \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) + D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}) ) \\
\theta^{*}, \phi^{*} &= \arg\min_{\theta, \phi} L_\text{VAE}
\end{aligned}
$$
$$-L_\text{VAE}$$   $$\log p_\theta (\mathbf{x})$$. 
$$
-L_\text{VAE} = \log p_\theta(\mathbf{x}) - D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) \leq \log p_\theta(\mathbf{x})
$$

#### 再参数化技巧

 $$\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})$$.  $$\mathbf{z}$$  $$\mathbf{z} = \mathcal{T}_\phi(\mathbf{x}, \boldsymbol{\epsilon})$$, $$\boldsymbol{\epsilon}$$ i $$\mathcal{T}_\phi$$  $$\phi$$  $$\boldsymbol{\epsilon}$$ to $$\mathbf{z}$$.

 $$q_\phi(\mathbf{z}\vert\mathbf{x})$$ 
$$
\begin{aligned}
\mathbf{z} &\sim q_\phi(\mathbf{z}\vert\mathbf{x}^{(i)}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}^{(i)}, \boldsymbol{\sigma}^{2(i)}\boldsymbol{I}) & \\
\mathbf{z} &= \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \text{, where } \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I}) & \scriptstyle{\text{; Reparameterization trick.}}
\end{aligned}
$$


 $$\odot$$ ![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-reparameterization-trick.png)

e $$\mathbf{z}$$ 

 $$\mu$$ and $$\sigma$$,  $$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})$$.

![](https://raw.githubusercontent.com/LibertyDream/diy_img_host/master/img/20200912-vae-gaussian.png)

### Beta-VAE

 $$\mathbf{z}$$ 

β-VAE $$\delta$$):
$$
\begin{aligned}
&\max_{\phi, \theta} \mathbb{E}_{\mathbf{x}\sim\mathcal{D}}[\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z})]\\
&\text{subject to } D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) < \delta
\end{aligned}
$$
r $$\beta$$  $$\mathcal{F}(\theta, \phi, \beta)$$:
$$
\begin{aligned}
\mathcal{F}(\theta, \phi, \beta) &= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) - \beta(D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) - \delta) & \\
& = \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) - \beta D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) + \beta \delta & \\
& \geq \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) - \beta D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) & \scriptstyle{\text{; Because }\beta,\delta\geq 0}
\end{aligned}
$$
 $$\beta$$-VAE
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