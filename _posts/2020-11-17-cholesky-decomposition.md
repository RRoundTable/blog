---

title: "Cholesky Decomposition 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['math', 'linear algebra']
layout: post
---

matrix를 decomposition하는데 다양한 방법이 있다. 이번 글에서는 그 중 cholesky decomposition에 대해서 다룰 것이다.

## Cholesky Decomposition

양의 실수가 주어져있다면, 제곱근을 구할 수 있다. 마찬가지로 matrix에서는 symmetric, positive definite matrix라면 cholesky decomposition을 이용하여 분해할 수 있다.



### Definition 4.18 (cholesky decomposition)

Symmetric, positive definite matrix A는 $A=LL^T$로 분해될 수 있다. 이 때, $L$은 traiagular matrix with positive diagonal element이다.

그리고 L을 cholesky factor라고 부르며 unique하다. 


$$
\begin{bmatrix} a_{11} & \cdots & a_{13} \\ \vdots & \ddots & \vdots \\ a_{31} & \cdots & a_{33} \\\end{bmatrix} = \begin{bmatrix} l_{11} & \cdots & 0 \\ \vdots & \ddots & \vdots \\ l_{31} & \cdots & l_{33} \\\end{bmatrix}  \begin{bmatrix} l_{11} & \cdots & l_{13} \\ \vdots & \ddots & \vdots \\ 0 & \cdots & l_{33} \\\end{bmatrix} ^T
$$



​	


cholesky decomposition은 machine learning 에서 numerical computation 과정에서 활용되기도 한다. (Symmetric, positive definite)

- corvariance matrix of gaussian multivariate  variable

  - corvariance matrix는 symmetric, positive definite하기 때문에 cholesky decomposition이 가능하다. 추후에 다루겠지만, 이는 gaussian distribution으로부터 sampling할 수 있도록 하며, random variable의 linear transpormation이 가능하도록한다.

    $\Sigma= AA^T$: covariance matrix

    $y ~ \mathcal{N}(u, \Sigma)$에서 random variable을 구하기 위해서 아래와 같이 할 수 있다.

    $x ~ \mathcal{N}(0, I)$
    $$
    y = Ax + u
    $$

  - 반면에, VAE에서는 미분가능한 sampling 함수를 만들기 위해서 상당히 무거운 계산을 한다. 

- Determinant 계산을 쉽게 한다.

  - $$
    A = LL^T \\
    det(A) = det(L)det(L^T) = det(L)^2
    $$

    
  
  - L은 triagular matrix이기 때문에
  
    - $$
      det(L) = \prod_{i} l_{ii} \\
    det(A) = \prod_{i} l_{ii}^2 \\
      $$
  
      

