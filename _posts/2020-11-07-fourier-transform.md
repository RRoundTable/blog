---

title: "Fourier transform 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['math', 'Fourier transform']
layout: post
---



## 시계열 데이터란,

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Random-data-plus-trend-r2.png/500px-Random-data-plus-trend-r2.png)

시간에 따라 순차적으로 관측한 데이터이다.  금융, 생체신호 등 다양한 도메인의 영역에서 시계열 데이터를 다루며, 시간 domain에서 데이터를 다룬다.



## Frequency Domain

![](https://www.radartutorial.eu/10.processing/pic/Time_vs_Frequency_domain.png)

위의 이미지는 여러가지 주기의 신호를 나타낸다. 위의 t는 시간을 나타내고, f는 frequency를 나타낸다. 즉 하나의 시계열 데이터를 time 혹은 fequency domain에서 접근할 수 있다는 것이다.



**Remark** 

- $T$: 한 주기

$$
f = \frac{1}{T}
$$





## Fourier Series

Fourier Series는 신호 데이터를 무한개의 주기함수로 나타낸다.

- $A_n$: $\cos nx$신호의 강도
- $B_n$: $\sin nx$ 신호의 강도
- $A_0$: 
- $2T$: 주기

$$
f(x) = A_0 + \sum_{n=1}^{\infty}(A_n \cos nx + B_n \sin nx)
$$

위의 식은  어떤 신호 $f(x)$를 여러개의 주기함수로 분해한 것이다. 이제 각 요소에 대해서 살펴보자.

아래는 $A_0$에 대한 수식이다. 
$$
A_0 = \frac{1}{2T} \int_{-T}^{T}f(x)dx
$$
이에 대해서 수식을 다음과 같이 전개해보면, 쉽게 이해할 수 있다.
$$
\int_{T}^{-T}f(x)dx = \int_{-T}^{T} A_0 d_x + 0 = A_0 \times 2T
$$




$A_n, B_n$은 각 $\frac{n}{2}$주기에 함수와 함수 $f(x)$간의 inner product이다. 해석하자면, 전체신호와 각 주기함수간의 유사도를 나타내며, 전체신호에서 어느정도의 비중을 차지하는지 표현할 수 있다.
$$
A_n = \frac{1}{2T} \int_{-T}^{T}f(x) \cos nx dx
$$

$$
B_n = \frac{1}{2T} \int_{-T}^{T}f(x) \sin nx dx
$$

$A_n$에 대해서도 다음과 같이 전개해보겠다. 

$f(x) = A_0 + \sum_{n=1}^{\infty}(A_n \cos nx + B_n \sin nx)$ 양변에 $\cos nx$를 곱한뒤 적분을 하겠다.
$$
\int_{-T}^{T}f(x) \cos nx dx = \int_{-T}^{T}A_0 \cos nx dx + \int_{-T}^{T}\sum_{i=-\infty}^{\infty} A_i \cos nx  \cos ix dx +  \int_{-T}^{T}\sum_{i=-\infty}^{\infty} B_i \sin nx  \cos ix dx
$$

$$
= \int_{-T}^{T}A_n (\cos nx)^2 dx
$$



- 삼각함수의 직교성에 의하여 서로 다른 주기 함수의 적분 값은 0이 된다.

- 주기함수를 주기 범위로 적분하면 그 결과는 0이 된다. 

  - $$
    \int_{-T}^{T}A_0 \cos nx dx = 0
    $$

    

- $\int_{-T}^{T} (\cos nx)^2 = 2T$





### 함수의 내적

함수를 무한차원의 vector처럼 생각하게 되면, vector와 마찬가지로 inner product를 적용할 수 있다.

- $f(x)$
- $g(x)$

$$
<f, g> = \int_{-\infty}^{\infty} f(x)g(x) dx
$$



### 삼각함수의 직교성

서로 다른 주기함수는 서로 직교한다.
$$
\int_{0}^{2\pi} \sin nx  \cos mx dx = \int_{0}^{2\pi} \frac{1}{2}(\sin(n+m) x + \sin(n -m)x) dx
$$




## Fourier Transform

fourier series에서는 주기신호를 주기신호들로 분해했다. 그렇다면, 비주기의 신호는 어떻게 처리할 수 있을까?

우선, 주기신호와 비주기 신호는 수학적으로 어떻게 나타낼 수 있을까?

우선 주기신호는 T의 주기를 가진다. 비주기 신호는 무한대의 주기를 가지는 신호라고 볼 수 있다.

`주기가 무한대인 주기함수 = 비주기함수`

### 주기가 무한대일 때, $A_n$은?

$$
A_n = \lim_{T \rightarrow \infty} \frac{1}{T}\int_{-\frac{T}{2}}^{\frac{T}{2}}f(x) \cos nx dx
$$



$\int_{-\frac{T}{2}}^{\frac{T}{2}}f(x) \cos nx dx$은 수렴하므로, $A_n$은 0에 수렴한다.



### 주기가 무한대일 때, $T A_n$은?

$$
T A_n = \lim_{T \rightarrow \infty} \int_{-\frac{T}{2}}^{\frac{T}{2}}f(x) \cos nx dx
$$



0이 아닌 값에 수렴한다.



### 유도과정

- 주기: $T$

$$
T A_n =  \int_{-\frac{T}{2}}^{\frac{T}{2}}f_T(x)_n e^{-jw_0t} dx
$$



T가 무한대로 간다면? 다음과 같이 치환할 수 있습니다.
$$
f_T(x) \triangleq f(x)
$$ {\\}

$$
A_nT \triangleq X(nw_0), w_0 = \frac{2\pi}{T}
$$

$$
nw_0 \triangleq w
$$



이를 바탕으로 위의 식을 다시 작성해보면,
$$
X(w) = \int_{-\infty}^{\infty} f(x)e^{-jwt} dt
$$


$X(w)$는 주파수 도메인의 입력을 받게 됩니다. $\int_{-\infty}^{\infty} f(x)e^{-jwt} dt$는 시간도메인의 함수를 적분을 통하여 주파수 도메인의 함수로 바꿉니다.

그렇다면, 주파수 도메인에서 시간 도메인으로는 어떻게 변환할까? 
$$
f(x) = \frac{1}{2\pi}\int_{-\infty}^{\infty} X(w)e^{-jwt} dt
$$


위와 같이, 주파수 도메인의 함수를 주파수를 기준으로 적분하면, 시간 도메인의 함수가 된다.





## Reference

[1] [헥펜파임](https://www.youtube.com/watch?v=KueJtenJ2SI&ab_channel=%ED%98%81%ED%8E%9C%ED%95%98%EC%9E%84)

