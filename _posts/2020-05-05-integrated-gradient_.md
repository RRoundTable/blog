---
title: "Integrated Gradient 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'XAI']
---



# Integrated Gradient 정리글



## Abstract

해당 논문에서는 Attribution method의 방법론의 Axiom을 정리하였다.

1. **Sensitivity**

2. **Implementation Invariance**

그리고 위의 두 가지 조건을 만족하는 **Integrated gradient** 방법론을 제시한다.



## Motivation



해당 연구의 목표는 input-output간의 관계를 파악하는 것이다. deep network prediction이 있을 때, 각 input feature가 어떤 영향을 끼쳤는지 알고 싶다.



아래와 같은 형식으로 attribution은 정의될 수 있다.

- deep network: $F: R^n \rightarrow [0, 1]$
- input: $x = (x_1, \cdots, x_n) \in R^n$
- baseline input: $\acute{x}$

$$
A_F(x, \acute{x}) = (a_1, \cdots, a_n) \in R^n
$$

여기서 baseline의 역할은 비교대상이다. 예를 들어, object recognition 과제가 있을 때, input image에서 어떤 pixel이 특정 class라고 판단하게 하는지 구할 수 있다.  아래의 이미지처럼, attribution을 알고싶다면, baseline을 모두 0으로 처리할 수 있다.

![{{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/ex1.png "Example"]()

baseline은 model의 행동의 원인을 파악하고자 필요한 개념이다. prediction이 중립적인 상황을 가정하며, image의 경우에는 위와 같이 나타낼 수 있다.



## Two Fundamental Axioms



### Gradients

gradient는 일반적으로 deep network의 model coefficient를 알 수 있는 방법이다. 특정 input feature의 gradient가 높게 나온다면, 해당 input feature가 중요한 역할을 한다고 생각할 수 있다. 하지만, **sensitivity**라는 성질에서 좋지 않은 결과를 보여준다.

### Sensitivity

sensitivity를 만족했다는 것은 다음을 의미한다. input하고 baseline의 차이가 오직 하나의 feature이고 서로 다른 예측을 한다면, 차이나는 feature에서는 non-zero attribution이 있다.

하지만, gradient는 이러한 특성을 반영하지 못한다. 아래의 이미지를 살펴보자.

![{{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/gradient.png "gradient"]()

- baseline: $x=0$

위의 상황에서 $x=2$ 일 때, gradient의 값은 0이다. 하지만, baseline을 고려해보면, 분명히 함수 값의 차이가 있다. ($y=0 \rightarrow y = 1$) 따라서 $x=2$에서 attribution은 0이면 안된다.

gradient의 한계로 인해서, 연관성 없는 feature에 gradient 값이 높게 나오기도 한다.

### Implementation Invariance

attribution이 두 개의 기능적으로 동일한 network상에서 항상 같아야 한다는 것이다.

![{{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/implementation_invariance.png]()

우선, gradient 자체로는 implementation invariant하다.  아래의 수식을 살펴보자. 

- input: $f$
- output: $g$
- network: $h$

$$
\frac{\partial f}{\partial g} = \frac{\partial f}{\partial h} \cdot \frac{\partial h}{\partial g}
$$

하지만, LRP 혹은 DeepLift와 같은 방법론에서는 변형된 discrete gradient를 사용한다. 하지만, 이와 같은 discrete gradient에서는 chain rule이 성립하지 않는다.


$$
\frac{f(x_1) - f(x_0)}{g(x_1) - g(x_0)} \ne \frac{f(x_1) - f(x_0)}{h(x_1) - h(x_0)} \cdot \frac{h(x_1) - h(x_0)}{g(x_1) - g(x_0)} \text{   for all  } x_1, x_0
$$
![{{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/figure7.png]()



- $f(x_1, x_2) = ReLU(h(x1, x2)) = ReLU(k(x_1, x_2))^3$
- $h(x_1, x_2) = ReLU(x_1) - 1 - ReLU(x_2)$
- $k(x_1, x_2) = ReLU(x_1 - 1) - ReLU(x_2)$
- $h$와 $k$는 서로 다르지만, $f$ 와$g$는 동일한 함수이다.

증명: $ReLU(x_1) - 1 \ne ReLU(x_1 - 1)$ 라면,  $f$ 와$g$는 동일한 함수가 아니다. 이는 $x_1 < 1$인 경우이고 이 경우에는  $f$ 와$g$ 모두 0 값을 가진다. 따라서 동일한 함수이다.

implementation에 따라서 attribution이 달라진다면, 중요하지 않은 feature에 집중할수 있다.





## Integrated Gradients

정리해보면, gradient는 sensitivity하지 않다. 이를 해결하기 위해서 discrete하게 gradient를 구하게 되면, implementation에 따라 다른 결과가 나올 수 있다. 이러한 문제를 해결하기 위해서 integrated gradient를 제안한다.


$$
IntegratedGrads_i(x) = (x_i - \acute{x}_i) \times \int_{\alpha=0}^1 \frac{\partial F(\acute{x} + \alpha(x-\acute{x}) )}{\partial x_i} d\alpha
$$

### Completeness

$$
\sum_{i=1}^n IntegratedGrads_i(x) = F(x) - F(\acute{x})
$$

각 dimension의 모든 integrated gradient값을 더하면, 함수값의 차이가 된다. 이는 결국 각 attribution의 합이 함수의 값의 차이와 동일하다는 것을 의미한다.







## Uniqueness of Integrated Gradients

object recognition task에서 attribution score top-k pixel을 제거해가면서 성능의 저하가 있는지 확인한다. 좋은 attribution method라면 점수는 급격히 떨어질 것이다. 하지만, 이러한 방법에는 문제가 있다.

unnatural한 data가 만들어진다. 따라서 성능의 저하가 단순히 attribution 때문이 아니라 처음 본 데이터 형식이라서 그럴 수 있다.

여기서는 두 가지 단계가 논리를 전개한다.

1. path method 소개
2. integrated gradient가 왜 path method중에서선택되었는지 설명



### Path Methods

![{{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/figure1.png]()

- $\gamma = (\gamma_1, \cdots, \gamma_n) : [0, 1] \rightarrow R^n$
- $\gamma(0) = \acute{x}$
- $\gamma(1) = x$

$$
PathIntegratedGrads_i^\gamma(x) = \int_{\alpha=0}^1 \frac{\partial F(\gamma(\alpha))}{\partial \gamma_i(\alpha)} \frac{\partial \gamma_i(\alpha)}{\partial \alpha} d\alpha
$$



모든 path methods는 implementation invariance 성질을 만족한다. 그리고 integrated gradient도 path method중 하나이며,  위의 이미지에서 $P2$ linear combination의 path에 해당한다.

### Integrated Gradients is Symmetry-Preserving

두 input variable이 서로 교환하여도 fucntion의 output의 변화가 없다면, symmetry하다고 한다.
$$
F(x, y) = F(y, x)
$$
attribution method는 동일한 symmetry value를 가지고 있고 baseline의 symmetric variable이 동일한 attribution을 가진다면, symmetry preserving하다고 한다. 

예시) 
$$
Sigmoid(x1 + x2, \cdots)
$$
$x_1, x_2$는 symmetric variable이고 input에서는 $x_1=x_2=1$ 이며, basline에서는 $x_1=x_2=0$이다. symmetry preserving하다면, $x_1, x_2$에 모두 동일한 attribution 값이 나와야한다.



그리고, integrated gradient는 이러한 조건을 만족한다. 아래를 간략히 정리하면, non-straightline은 symmetry preserving하지 않다는 것이다.

![{{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/proof1.png]()

