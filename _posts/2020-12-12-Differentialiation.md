---

title: "Differentiation 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['differentiation', 'vector calculus', 'math']
layout: post
---



## Vector Calculs

![](https://miro.medium.com/max/1108/1*WBznQc-8wfBbHd3sZT1WIQ.png)

vector calculs는 추후에 다룰 probability, optimization, regression, dimensionality reduction, density estimation, classification과 같은 주제와 직결되는 주제이다. 이번 글에서는 partial differentiation에 대해서 다룰 계획이다.



## Differentiation of Univariate Functions

![]({{ site.baseurl }}/images/2020-12-12-Differentialiation/quotient.png)

### Definition 5.1 (Difference Quotient)

- $f(x) = y, x, y \in R$

$$
\frac{\delta y}{\delta x} := \frac{f(x+\delta x) - f(x)}{x}
$$

diffrence quotient는 위의 그림과 같이 구성된다.



### Definition 5.2 (Derivertive)

$$
\frac{dx}{dy} := \lim_{h \rightarrow 0}\frac{f(x + h) - f(x)}{x}
$$

여기서 극한의 개념을 적용하게 되면, derivertive는 h가 0에 가까워지고 tangent(접선)의 기울기가 된다.



## Taylor Series

![](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Sintay_SVG.svg/1920px-Sintay_SVG.svg.png)

taylor series는 어떤 함수를 무한개의 항으로 나타내며, 이런 항들은 $f$의 $x_0$에 대한 derivative를 통해서 나타낸다.



### Definition 5.3 (Taylor Polynomial)

$$
T_n(x) := \sum_{k=0}^n\frac{f^{(k)}(x_0)}{k!}(x - x_0)^k
$$



- $f: R \rightarrow R$
- $f^{(k)}(x_0)$: k-th derivative of $f$ at $x_0$

### Definition 5.4 (Taylor Series)

- $f: R \rightarrow R, f \in C^{\infty}$: smooth function, continuously differentiable infinitely many times

$$
T_{\infty}(x) := \sum_{k=0}^{\infty}\frac{f^{(k)}(x_0)}{k!}(x - x_0)^k
$$

만약에 $f(x) = T_{\infty}(x)$라면, $f$가 analytic하다고 한다.



