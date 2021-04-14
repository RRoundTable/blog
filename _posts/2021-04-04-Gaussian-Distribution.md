---
title: "Gaussian distribution"
toc: true
branch: master
badges: true
comments: true
categories: ['math']
layout: post
---

# Gaussian Distribution

Gaussian Distribution은 Normal Distribution이라고도 불리며 계산상 이점이 있어 많이 활용되는 분포이다.
Linear Regression에서 사전분포로 Gaussian Distribution이 활용되기도 하며, Gaussian Mixture에서도 활용된다.
Machine Learning에서 많은 영역에서 활용되는데 Gaussian Process, Variational Inference, Reinforcement Learning에서도 활용된다.
또한 칼만필터, Control, Hypothesis Testing에서도 활용되는 분포이다.

Univariate Random Variable의 경우 Gaussian Distribution은 다음과 같은 Density를 가진다.

$$
p(x \mid u, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp (-\frac{(x-u)^2}{2 \sigma^2})
$$

Multivariate Gaussian Distribution은 다음과 같은 분포를 가진다.

- $u$: Mean Vector
- $\Sigma$: Covariance Matrix
- $x \in R^D$

$$
p(x \mid u, \Sigma) =  (2 \pi) ^{-\frac{D}{2}} \rvert \Sigma \rvert^{-\frac{1}{2}} \exp (-\frac{1}{2}(x -u)^{T} \Sigma^{-1} (x-u))
$$


Standard Normal Distribution은 $u=0, \Sigma=I$인 경우를 의미한다.

## Marginals and Conditionals of Gaussians are Gaussians

Sum Rule을 설명하기전에 먼저 Joint Distribution에 대해서 알아보자.

- $\Sigma_{xx} = \mathbb{Cov}[x, x]$

$$
p(x,y) = \mathcal{N} (
    \begin{bmatrix} u_x \\ u_y \end{bmatrix},
    \begin{bmatrix} \Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy}\end{bmatrix}
) 
$$

Conditional Distribution은 다음과 같다.

- $u_{x \mid y} = u_x + \Sigma_{xy} \Sigma_{yy}^{-1} (y - u_x) $
- $\Sigma_{x \mid y} = \Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}$

$$
p(x \mid y) = \mathcal{N} (u_{x \mid y}, \Sigma_{x \mid y})
$$

Conditional Distribution은 다음 사례에서 활용된다.

- 칼만필터
- Gaussian Process
- Latent Linear Gaussian Model

Marginal Distribution은 다음과 같다.

$$
p(x) = \int p(x, y) dy = \mathcal{N} (x \mid u_x, \Sigma_{xx})
$$

아래의 이미지를 보자.
Joint Distribution에서 Conditional Distribution과 Marginal Distribution의 모습을 볼 수 있다.

![]({{ site.baseurl }}/images/2021-04-04-Gaussian-Distribution/distribution.png)



## Product of Gaussian Densities

Posterior Density를 구하기 위해서 Likelihood와 Prior를 곱해야한다.
Gaussian Distribution의 Product는 다음과 같이 이루어진다.

- $c \in R$: scaling
  

$\mathcal{N}(x \mid a, A)$와 $\mathcal{N}(x \mid b, B)$의 Product의 결과가 $c\mathcal{N}(x \mid \mathbf{c}, C)$인 상황을 고려해보자.


$$
C = (A^{-1} + B^{-1})^{-1} \\
\mathbf{c} = C(A^{-1}a + B^{-1}b) \\ 
c = (2 \pi)^{-\frac{D}{2}} \rvert  A + B\rvert^{-\frac{1}{2}}\exp(-\frac{1}{2} (a-b)^T (A + B)^{-1}(a + b))
$$

Scaling Constanct $c$는 Gaussian Density로 표현할 수 있다.

$$
c = \mathcal{N}(a \mid b, A + B) = \mathcal{N}(b \mid a, A + B)
$$


## Sum and Linear Transformations

$X, Y$가 독립인 Gaussian Random Variable이라면 다음과 같은 관계가 성립한다.

- $p(x, y) = p(x) p(y)$
- $p(x) = \mathcal{N}(x; u_x, \Sigma_x)$
- $p(y) = \mathcal{N}(y; u_y, \Sigma_y)$

$$
p(x + y) = \mathcal{N}(u_x + u_y, \Sigma_x + \Sigma_y)
$$

위의 관계는 Linear Regression에서 Random Variable에 대해서 Gaussian Noise의 영향을 고려할 때 사용될 수 있다.


### Theorem 6.12

두 Random Variable의 Mixture Density를 고려해보자.

- $0 \lt \alpha \lt 1$
- $p_1(x), p_2(x)$: Univariate Gaussian Densities, $(u_1, \sigma_1)$, $(u_2, \sigma_2)$

$$
p(x) = \alpha p_1(x) + (1 - \alpha) p_2(x)
$$


$$
\mathbf{E}[x] = \alpha u_1 + (1 - \alpha) u_2
$$

$$
\mathbf{V}[x] = [\alpha \sigma_1^2 + (1 - \alpha )\sigma_2^2] - ([\alpha u_1^2 + (1 - \alpha) u_2^2] - [\alpha u_1^2 + (1 - \alpha) u_2])
$$

#### Proof

$$
\mathbf{E}[x] = \int_{\infty}^{\infty} x p(x) dx = \int_{\infty}^{\infty} \alpha xp_1(x) + (1 - \alpha) xp_2(x) dx \\
=  \alpha \int_{\infty}^{\infty} xp_1(x) dx + (1 -\alpha)  \int_{\infty}^{\infty} \alpha xp_2(x) dx
= \alpha u_1 + (1 - \alpha) u_2
$$

- $\sigma^2 = \mathbf{E}[x^2] - u^2$

$$
\mathbf{E}[x^2] = \int_{\infty}^{\infty} x^2 p(x) dx = \int_{\infty}^{\infty} \alpha x^2p_1(x) + (1 - \alpha) x^2p_2(x) dx \\
= \alpha \int_{\infty}^{\infty} x^2p_1(x) dx + (1 -\alpha)  \int_{\infty}^{\infty} \alpha x^2p_2(x) dx \\
= \alpha (u_1^2 + \sigma_1^2) + (1 - \alpha)(u_2^2 + \sigma_2^2)
$$


$$
\mathbf{V}[x^2] = \mathbf{E}[x^2] -  (\mathbf{E}[x])^2 \\
= \alpha(u_1^2 + \sigma_1^2) + (1 - \alpha)(u_2^2 + \sigma_2^2) - (\alpha u_1 + (1 - \alpha) u_2)^2
= [\alpha \sigma_1^2 + (1 - \alpha )\sigma_2^2] - ([\alpha u_1^2 + (1 - \alpha) u_2^2] - [\alpha u_1^2 + (1 - \alpha) u_2])
$$

### Transformation

$X \sim \mathcal{N} (u, \Sigma)$를 고려해보자.
Matrix $A$가 주어졌을 때, $y = Ax$라고 하자.

y의 Mean은 다음과 같이 구할 수 있다.

$$
\mathbf{E}[y] = \mathbf{E}[Ax] = A \mathbf{E}[x] = Au
$$

비슷하게 Variance는 다음과 같이 구할 수 있다.

$$
\mathbf{V}[y] = \mathbf{V}[Ax] = A \mathbf{V}[x] A^T = A \Sigma A^T
$$

$$
p(y) = \mathcal{N}(y \mid Au, A \Sigma A^T)
$$

이제 y로 x를 표현해보자.

$$
y = Ax \iff x = (A^T A)^{-1} A^T y
$$

$$
p(y) = \mathcal{N} (y \mid Ax, \Sigma)
$$

$$
p(x) = \mathcal{N}(x \mid (A^T A)^{-1} A^T y, (A^T A)^{-1} A^T \Sigma A (A^T A)^{-1} )
$$


## Sampling from Multivariate Gaussian Distributions

$\mathcal{N}(u, \Sigma)$를 얻기 위해서 다음과 같은 과정을 한다.

1. $x \sim \mathcal{N}(0, I)$
2. $y = Ax + u$, $AA^T = \Sigma$
   


   http://math.mit.edu/~sheffield/600/Lecture26.pdf