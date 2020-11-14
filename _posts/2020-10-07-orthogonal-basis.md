---

title: "orthogonal basis"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'linear algebra']
layout: post
---



## Orthonormal Basis

n-dimensional vector space V와 V의 basis $\\{b_1, \cdots, b_n \}$ 이 있다고 했을때, 다음과 같은 조건을 만족하면, Orthonomal basis이다.
$$
<b_i, b_j> = 0 \text{  for     }  i \neq j
$$

$$
<b_i, b_i> = 1  
$$



##  *Gram-Schmidt process*: Orthonormal Basis를 구해보자.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Gram%E2%80%93Schmidt_process.svg/1200px-Gram%E2%80%93Schmidt_process.svg.png)



## Orthogonal Complement

D dimension의 vector space V와 vector subspace $U \subset V$(M dimension)이 있다고 해보자.

이때 U의 orthogonal complement는 $U^{\perp}$로 표현하며, U와 orthogonal하다. 그리고 dimension은 D - M 이다. 
$$
U \cap U^{\perp} = \{ 0\}
$$


그리고, $x \in V$는 결국 $U$와 $U^{\perp}$의 vector들로 표현이 가능하다.
$$
x = \sum_{i=1}^M \lambda_i b_i + \sum_{i=1}^{D - M} \psi_i b_i^{\perp}, \lambda_i, \psi_i \in R
$$


다음과 같은  한 평면에 수직인 vector가 있다고 해보자. 파란색의 vector와 수직인 vector는 모두 파란색 평면에 mapping된다. 이를 파란색 vector가 파란색 평면의 normal vector라고 표현하며, orthogonal complement이다. 파란색 vector 기준으로 orthogonal complement는 파란색 평면이며, 파랑색 평면은 파란색 백터의 hyperplane이다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Surface_normal_illustration.svg/440px-Surface_normal_illustration.svg.png)