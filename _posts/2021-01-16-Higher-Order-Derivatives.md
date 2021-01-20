---
title: "Higher Order Derivatives"
toc: true
branch: master
badges: true
comments: true
categories: ['math']
layout: post
---



## Higher Order Derivatives

함수 $f: R^2 \rightarrow R$인 함수가 있다고 가정해보자. 아래와 같은 표기로 Higher Order Derivatives를 표기할 것이다.



- $x$에 대한 $f$의 Second Partial Derivative 
  $$
  \frac{\partial^2 f }{\partial x^2}
  $$

- $x$에 대한 $f$의 n-th Partial Derivative 
  $$
  \frac{\partial^n f }{\partial x^n}
  $$
  

- First Partial Derivative로 부터 얻은 Partial Derivatives
  $$
  \frac{\partial^2 f }{\partial y x} = \frac{\partial}{\partial y}\frac{\partial f }{\partial x}
  $$
  
  $$
  \frac{\partial^2 f }{\partial x y} = \frac{\partial}{\partial x}\frac{\partial f }{\partial y}
  $$



**Hessian**이라고 정의하는 것은 **Second Order Partial Derivatives**를 모두 모아논 것이다. 하나의 예시를 살펴보겠다.

만약 $f(x,y)$가 두 번 미분가능한 함수라면, 아래와 같이 무엇으로 먼저 미분하던 결과는 동일하다.
$$
\frac{\partial^2 f }{\partial y x} = \frac{\partial^2 f }{\partial x y}
$$
그리고 Hessian matrix는 아래와 같이 표현된다.
$$
H = \nabla^2_{x, y}f(x, y) = \begin{bmatrix} \frac{\partial^2 f }{\partial x^2} & \frac{\partial^2 f }{\partial y x} \\
\frac{\partial^2 f }{\partial y x} & \frac{\partial^2 f }{\partial y^2}

\end{bmatrix}
$$


일반적으로 $f: R^n \rightarrow R$인 함수에서 Hessian은 $R^{n \times n}$의 꼴을 가지고 있다. 만약 $: R^n \rightarrow R^m$이라면, $R^{m \times n \times n}$의 모양을 가진다.

