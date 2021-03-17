---
title: "Summary Statistics and Independence"
toc: true
branch: master
badges: true
comments: true
categories: ['math']
layout: post
---



우리는 Random Variable을 요약하고 싶어하며, 요약된 결과는 Deterministic하다. 이런 정보는 Random Variable의 특성을 파악하는데 도움을 준다.  이번 장에서는 Mean과 Variance에 대해서 다룰 것이다. 그리고 Random Variable을 비교하는 방법에 대해서 다룰 것이다.

1. 두 Random Variable의 독립성
2. 두 Random Variabler간의 Inner Product 



## Means and Covariance



### Definition: Expected Value

- $\mathcal{X}$: Random Variable의 Target Space, Possible Outcome

Univariate Continuous Random Variable $X ~ p(x)$아래에서 함수 g $g: R \rightarrow R$의 Expected value는 아래와 같이 정의된다.

$$
\mathbb{E}_x[g(x)] = \int_{\mathcal{X}}g(x) p(x) dx
$$

Discrete Random Variable일 경우에는 아래와 같이 구할 수 있다.

$$
\mathbb{E}_x[g(x)] = \sum_{x\in \mathcal{X}} g(x)p(x)
$$

**Remark: Multivariate Case**

$$
\mathbb{E}_x[g(x)] = \begin{bmatrix} \mathbb{E}_{X_1}[g(x_1)] \\ \mathbb{E}_{X_2}[g(x_2)] \\  \vdots \\ \mathbb{E}_{X_n}[g(x_n)]\vdots \end{bmatrix} \in R^{n}
$$


**Example**

![](https://media.cheggcdn.com/media/985/98588c05-3671-473d-b27e-531bacca2d9a/phpV2YbqL.png)

위 이미지의 분포를 수식으로 나타내면 아래와 같다.

$$
p(x) = 0.4 \cdot \mathcal{N}(x \mid \begin{bmatrix} 10  \\ 2\end{bmatrix}, \begin{bmatrix} 1 & 0 \\   0 & 1 \end{bmatrix}) +  0.6 \cdot \mathcal{N}(x \mid \begin{bmatrix} 0  \\ 0\end{bmatrix}, \begin{bmatrix} 8.4 & 2.0 \\   2.0 & 1.7 \end{bmatrix})
$$


위의 분포의 Expected Value는 다음과 같이 구할 수 있다.

$$
\mathbb{E}_X[f(x)] = \int{f(x) p(x) dx}  = \int {[ag(x) + bh(x)] p(x) dx} \\ = a \int{g(x)p(x) dx} + b \int{h(x)p(x)dx} = a \mathbb{E}_X[g(x)] + b \mathbb{E}_X[h(x)]
$$




### Definition: Mean

$X$ state $x \in R^D$Random Variable의 Mean은 아래와 같이 정의된다.

$$
\mathbb{E}_x[x] = \begin{bmatrix} \mathbb{E}_{X_1}[x_1] \\ \mathbb{E}_{X_2}[x_2] \\  \vdots \\ \mathbb{E}_{X_n}[x_n]\vdots \end{bmatrix} \in R^{n}
$$


Continuous Random Variable

$$
\mathbb{E}_{x_d}[x_d] = \int_{\mathcal{X}}x_dp(x_d) dx_d
$$


Discrete Random Variable

$$
\mathbb{E}_x[x] = \sum_{x_d\in \mathcal{X}}^{\mathcal{X}} x_ip(x_d=x_i)
$$

### Definition: (Univariate) Covariance

두 Random Variable $X, Y \in R$의 Covariance는 아래와 같이 정의된다.

$$
\mathbf{Cov}_{X, Y}[x, y] = \mathbb{E}_{X,Y}[(x - \mathbf{E}_X[x])(y - \mathbf{E}_Y[y])]
$$

위의 수식은 아래와 같이 전개할 수 있다.

$$
\mathbf{Cov}_{X, Y}[x, y] = \mathbb{E}[xy] - \mathbb{E}(x) \mathbb{E}(y)
$$

- $\mathbf{Cov}[x, x] = \mathbf{Var}(x)$
- $\sqrt{\mathbf{Var}(x)} = \sigma{x}$



### Definition: (Multivariate) Covariance

두 Multivariate Random Variable $X \in R^D, Y \in R^E$는 아래와 같이 정의된다.

$$
\mathbf{Cov}[x, y] =  \mathbb{E}[xy^T] - \mathbb{E}(x) \mathbb{E}(y) = \mathbf{Cov}[y, x] ^T \in R^{D \times E}
$$

직관적으로는 Random Variable의 퍼짐정도를 나타낸다. Multivariate Random Variable의 경우에는 각 Element가 Random Variable의 Dimemsion간의 관계를 의미한다.



### Definition: Variance

Random Variable $X$ with states $x \in R^D$의 Variance는 아래와 같이 정의된다.

$$
\mathbb{V}_X(x) =  \mathbf{Cov}_X[x,x] = \mathbf{E}_X[(x-u)(x-u)^T] = \mathbf{E}_X[xx^T] - \mathbb{E}[x]\mathbf{E}[x]^T \\ = \begin{bmatrix} \mathbf{Cov}[x_1, x_1] & \mathbf{Cov}[x_1, x_2] & \cdots &  \mathbf{Cov}[x_1, x_D] \\ \vdots  & \vdots & \ddots & \vdots  \\ 
\mathbf{Cov}[x_D, x_1] & \mathbf{Cov}[x_D, x_2] & \cdots &  \mathbf{Cov}[x_D, x_D]
\end{bmatrix}
$$




### Definition: Correlation

두 Random Variable의 Correlation은 아래와 같이 정의된다.

$$
\mathbf{corr}[x, y] =  \frac{\mathbf{Cov}[x, y]}{ \sqrt{\mathbb{V}[x]\mathbb{V}[y]} }\in [-1, 1]
$$





