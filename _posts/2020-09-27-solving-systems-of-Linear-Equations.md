---
title: "Solving Systems of Linear Equations 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'linear algebra']
layout: post
---



## Particular solution에서 General solution으로 확장하기


$$
\begin{bmatrix}
1 & 0 &  8 & -4\\
0 & 1 &  2 &  12\\
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
x_4\\
\end{bmatrix} = 
\begin{bmatrix}
42 \\8
\end{bmatrix}
$$


위와 같은 linear equation이 있다고 생각해보겠습니다.

우선 particular solution을 구해보겠습니다.  우선 column의 linear combination으로 다음과 같이 해를 구할 수 있습니다.


$$
b =\begin{bmatrix}
42 \\
8
\end{bmatrix} = 42 \begin{bmatrix}
1\\
0
\end{bmatrix} + 
8\begin{bmatrix}
0 \\1
\end{bmatrix}
$$

$$
[x_1, x_2, x_3, x_4] ^T = [42, 8, 0, 0] ^T
$$



그리고, 위의 matrix를 잘 살펴보면, 일부 column의 linear combination을 통해서 다른 column vector를 표현할 수 있음을 알 수 있습니다.


$$
\begin{bmatrix}
8 \\
2
\end{bmatrix} = 8 \begin{bmatrix}
1\\
0
\end{bmatrix} + 
2\begin{bmatrix}
0 \\1
\end{bmatrix}
$$

$$
\begin{bmatrix}
-4 \\
12
\end{bmatrix} = -4 \begin{bmatrix}
1\\
0
\end{bmatrix} + 
12\begin{bmatrix}
0 \\1
\end{bmatrix}
$$





이것을 바탕으로 $Ax=0$인 해와 $Ax = b$의 해를 결합한 general solution을 구할 수 있습니다.


$$
[42, 8, 0, 0] ^T + \lambda_1 [8, 2, -1, 0] ^T + \lambda_2[-4, 12, 0, -1]^T
$$


## Elementary Transformation

$$
2x_1 + 4x_2 - 2x_3 - \ x_4 + 4x_5 = -3 \\
4x_1 + 8x_2  + 3x_3 - 3x_4 + x_5 =2 \\
x_1  -2x_2 + x_3 - x_4 + x_5 = 0 \\
x_1  - 2x_2 - 3x_4 +4x_5 = a
$$

$$
[A \mid b] = \begin{bmatrix}
2 & 4 & -2 & -1 & 4 \mid  & -3\\
4 & 8 & 3 & -3 & 1  \mid &2\\
1 & -2 & 1 & -1 & 1  \mid & 0\\
1 & -2 & 0 & -3 & 4  \mid & a\\
\end{bmatrix}
$$



# Row echelon form

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/3743aca294b2e5346c167819fd9ee0bcb79ef22c)

Matrix는 row chelon form이라고 정의할 수 있다.

- 모든 요소가 zero인 row들은 matrix 하단에 위치한다. 동시에, 모든 요소가 zero가 아닌 row는 모든 요소가 zero인 row보다 위에 있다.
- 모든 요소가 zero가 아닌 row에서 첫 번째로 zero가 아닌 요소는 pivot이라고 부른다.



- basic variable: pivot variable

- Free variable: 나머지



Row echelon form은 particular solution을 구하는데 유용하다.


$$
\lambda_1 \begin{bmatrix}
1 \\
0 \\
0 \\
0
\end{bmatrix} + \lambda_2 \begin{bmatrix}
1 \\
1 \\
0 \\
0
\end{bmatrix} + \lambda_3 \begin{bmatrix}
-1 \\
-1 \\
1 \\
0
\end{bmatrix} =
\begin{bmatrix}
0 \\
-2 \\
1 \\
0
\end{bmatrix}
$$


먼저, $\lambda_3$을 쉽게 구할 수 있으며, 연쇄적으로 $\lambda_2, \lambda_1$을 계산할 수 있다.





### Reduced row echelon form

- Row echelon form
- every pivot = 1
- Pivot 은 한 column에서 유일한 non-zero element이다.





## Minus -1 trick

$$
A = \begin{bmatrix}
1 & 3 & 0 &0 & 3 \\
0 & 0 & 1 & 0 & 9  \\
0 & 0 & 0 & 1 & -4  \\
\end{bmatrix}
$$

$$
\tilde{A} = \begin{bmatrix}
1 & 3 & 0 &0 & 3 \\
0 & -1 & 0 &0 & 0 \\
0 & 0 & 1 & 0 & 9  \\
0 & 0 & 0 & 1 & -4  \\
0 & 0 & 0 & 0 & -1  \\
\end{bmatrix}
$$



$Ax=0$의 해는 아래와 같습니다.
$$
x = \lambda_1 \begin{bmatrix}
3 \\
-1 \\
0 \\
0 \\
0
\end{bmatrix} + \lambda_2 \begin{bmatrix}
3 \\
0 \\
9 \\
-4 \\
-1
\end{bmatrix} , \lambda_1, \lambda_2 \in R
$$




### **Calculating an Inverse Matrix by Gaussian Elimination**

$$
A = \begin{bmatrix}
1 & 0 & 2 &0  \\
1 & 1 & 0 & 0   \\
1 & 2 & 0 & 1 \\
1 & 1 & 1 & 1 \\
\end{bmatrix}
$$


$$
[A \mid I] = \begin{bmatrix}
1 & 0 & 2 &0  \mid 1 & 0 & 0 & 0\\
1 & 1 & 0 & 0 \mid 0 & 1 & 0 & 0\\
1 & 2 & 0 & 1 \mid 0 & 0 & 1 & 0\\
1 & 1 & 1 & 1 \mid 0 & 0 & 0 & 1\\
\end{bmatrix}
$$


위의 matrix를 reduced row echelon form으로 변화시킨다.
$$
\begin{bmatrix}
1 & 0 & 0 & 0 \mid &-1 & 2 & -2 & 2\\
0 & 1 & 0 & 0 \mid &1 & -1 & 2 & -2\\
0 & 0 & 1 & 0 \mid& 1 & -1 & 1 & -1\\
0 & 0 & 0 & 1 \mid& -1 & 0 & -1 & 2\\
\end{bmatrix}
$$

$$
A^{-1} =\begin{bmatrix}
-1 & 2 & -2 & 2\\
1 & -1 & 2 & -2\\
1 & -1 & 1 & -1\\
-1 & 0 & -1 & 2\\
\end{bmatrix}
$$



## **Algorithms for Solving a System of Linear Equations**

- least square solutions

$$
Ax =b  \iff A^T A x = A^T b \iff x = (A^TA)^{-1}A^Tb
$$

- Gaussian eliminations
- Iterative solution

​	