---

title: "eigenvalue 와 enginevector 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['math', 'linear algebra']
layout: post
---



## Eigenvalue 와 Eigenvector

### Definition 4.6

square matrix $A \in R^{n \times n}$가 주어졌을 때, 아래의 식을 만족하면 $\lamda \in R$는  engienvalue이고  $x \in R^n/\{ 0\}$는 eigenvector 이다. 그리고 아래의 식을 eigenvalue equation이라고 한다.
$$
Ax = \lambda x
$$


아래의 명제들은 모두 동치이다.

- $\lamdba \in R$ 은 eigenvalue이다.
- $Ax = \lambda x$ 혹은 $(A -\lambda I)x = 0$의 해가 non-trivialy하다. ($x \in R ^ n / \{0\}$)
- $rk(A -\lambda I) < n$: eigenvector x가 non-trivial한 해이므로 
- $det(A -\lambda I) = 0$



### Definition 4.7 (collinearity and codirection)

두 vector가 같은 방향인 것을 codirection, 반대인 것을 colinearity라고 정의한다.



### Remark: non-uniqueness of eigenvector

- $c \in R/ \{0\}$
- $x \ in R^n$: eigenvector

$$
Acx = cAx = c \lambda x = \lambda (cx)
$$



### Theorem 4.8 

$\lambda$ 는 *characteristic polynomial* $p_A(\lambda)$ of A의 해일 때만, eigenvalue이다.
$$
p_A(\lambda) = det(A - \lambda I) = 0
$$


이는 $(A - \lambda I) x = 0$에서 $A - \lambda I$가 0 vector가 아닌 해를 가져야 하기 때문이다.

### Definition 4.9

square matrix A가 eigenvalue $\lambda$를 가진다고 가정해보자.

$\lambda$의 algebraric multiplicity는 동일한 eigenvalue와 대응하는 eigenvector의 수이다.



### Definition 4.10 (Eigenspace and Eigenspectrum)

- Eigenspace: 모든 eigenvector들로 span 되는 공간
- Eigenspectrum: 모든 eigenvalue의 집합



### Useful Properties of eigenvalue and eigenvector

- $A, A^T$는 동일한 eigenvalue를 가지지만, eigenvector는 꼭 그렇지 않다.

- $$
  Ax = \lambda x \iff Ax - \lambda x = 0 \iff (A - \lambda I) x = 0 \iff x \in ker(A - \lambda I)
  $$

- similar matrices는 모두 동일한 eigenvalue를 가진다. 따라서, linear mapping $\Phi$는 basis 선택과 독립적인 eigenvalue를 가진다고 해석할 수 있다.
  $$
  B = P^{-1}AP \\
  A = PBP^{-1}
  $$

  $$
  Av = PBP^{-1}v = \lambda v \iff BP^{-1}v = P^{-1}\lambda v
  $$

  

- Symmetric, positive matrix는 항상 실수의 eigenvalue값을 가진다.

  

### Example 4.5

### Definition 4.11

$\lambda_i$를 square matrix A의 eigenvalue라고 가정하자. 이때 $\lambda_i$의 geometric multiplicity는 eigenvalue $\lambda_i$와 연관되는  linear independent eigenvector들의 수이다.

따라서, eigenvalue의 algebraric multiplicity는 geometric multiplicity보다 항상 크거나 같다.



### Geometric Intuition in Two Dimensions

![](/Users/makinarocks/Library/ApplicationSupport/typora-user-images/image-20201115210914855.png)





