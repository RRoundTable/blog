---

title: "angles and orthogonality 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'linear algebra']
layout: post
---





## Angles

inner product는 angle에 대한 설명도 할 수 있다. 

일반적으로 Cauchy-Schwarz inequality를 사용해서 두 개의 vector x,y의 inner product space상의 angle을 설명한다.

- $x \neq 0$
- $y \neq 0$

$$
-1 \leqslant \frac{<x, y>}{\rVert x\rVert \rVert y \rVert} = cos \mathcal{w} \leqslant 1
$$



위의 식은 -1 부터 1의 범위의 값을 가진다. 아래의 cosine 그래프를 보면, 해당하는 값의 범위가 unique하게 결정된다는 것을 알 수 있다.

<img src="/Users/makinarocks/Library/Application Support/typora-user-images/image-20201010194402984.png" alt="image-20201010194402984" style="zoom:50%;" />



직관적으로 두 vector간의 angle은 서로간의 유사도를 표현한다. (Cosine similarity)

![](https://www.oreilly.com/library/view/statistics-for-machine/9781788295758/assets/2b4a7a82-ad4c-4b2a-b808-e423a334de6f.png)



또한, angle은 orthogonality과 연관이 깊다.



## Orthogonality

### Definition: orthogonality and orthonormality

두 vector x,y는 inner product $<x, y> = 0$ 일 때, 서로 orthgonal 하며, 수학적인 기호로 $x \bot y$로 나타낸다. 

또한, $\rVert x \rVert  = \rVert y \rVert = 1$이면, orthnormal하다고 한다.

**REMARK**

orthogonal을 정의할 때, dot product 뿐만 아니라 inner product에 대해서 정의된다.



### Definition: Orthogonal Matrix

Square matrix $A \in R^{n \times n}$ 는 orthogonal matrix이며, 이는 아래와 같이 나타낸다.
$$
AA^T  = A^T A = I
$$

$$
A^{-1} = A^T
$$



orthonormal matrix에 의한 변환은 vector의 length를 유지한다는 특징을 가진다.
$$
\rVert Ax\rVert ^ 2 =(Ax)^T (Ax) = x^T A^T A x = x^T I x = \rVert x\rVert^2
$$


그리고, orthonormal matrix로 변환한 두 vector간의 angle도 변하지 않는다.
$$
\cos w = \frac{<Ax, Ay>}{\rVert Ax\rVert \rVert Ay\rVert} = \frac{x^T y}{\rVert x \rVert \rVert y\rVert}
$$


정리하면, orthonormal matrix는 

- distance를 보존하며
- angle도 유지한다.