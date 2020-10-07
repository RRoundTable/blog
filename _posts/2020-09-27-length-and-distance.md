---

title: "length and distance 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'linear algebra']
layout: post
---



vector의 length는 norm으로 정의할 수 있으며, inner product로 설명할 수 있기도 하다. (하지만, 모든 norm을 inner product로 표현하는 것은 아니다. L1 norm)
$$
\rVert x \rVert:= \sqrt{<x, x>}
$$
이 책에서는 주로 inner product로 정의될 수 있는 norm에 대해서 다룰 예정이다.

### Remark: Cauchy-Schwarz Inequality

- inner product space: $(V, <\cdot, \cdot>)$
  $$
  \rvert <x, y> \rvert \leqslant \rvert x \rvert \rvert y\rvert
  $$
  

## Example 3.5

- $x = [1, 1]^T \in R^2$
- Inner product: dot product

$$
\rVert x \rVert = \sqrt{x^Tx} = \sqrt{1 + 1} = \sqrt{2}
$$



만약에 다른 inner product를 고른다면?

- inner product: 
  $$
  <x, y> = x^T \begin{bmatrix}
  1 & -\frac{1}{2}  \\  -\frac{1}{2} & 1 \end{bmatrix} y = x_1y_1 - \frac{1}{2}(x_1y_2 + x_2 y_1) + x_2 y_2
  $$

  $$
  <x, x> = x_1^2 - x_1x_2 + x_2^2 = 1 - 1 + 1 = 1 = \rvert x \rvert 
  $$

  

## Definition: distance and metric

- inner product space: $(V, <\cdot, \cdot>)$
- $x, y in V$

$$
d(x, y) := \rVert x - y \rVert = \sqrt{<x - y, x -y>}
$$

아래의 D와 같은 mapping을 metric이라고 부르며, d(x,y)는 distance이다.
$$
D: V \times V \rightarrow R \\
(x, y) \rightarrow d(x, y)
$$


distance또한 length와 마찬가지로 norm으로 정의하며, inner product의 개념을 필요로 하지 않는다. 하지만, norm자체가 inner product의 영향을 받는 경우라면, inner product의 선택에 따라서 distance가 변할 수 있다.





## Distance의 조건

- d는 positive definite하다.

  - $$
    d(x,y) \geqslant 0 \ \forall x, y \in V \\ d(x, y) = 0 \iff x = y
    $$

    

- d는 symmetric하다.

  - $$
    d(x, y) = d(y, x)
    $$

- Triangle inequality

  - $$
    d(x, z) \leqslant d(x, y) + d(y + z) \ \forall x, y, z \in V
    $$

    

