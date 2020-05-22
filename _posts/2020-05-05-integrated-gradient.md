---
title: "Integrated Gradient 정리글"
toc: true
branch: master
badges: true
comments: true
categories: ['deeplearning', 'XAI']
layout: post
---

# Integrated Gradient 정리글

XAI을 위한 다양한 방법론들이 있습니다. 그 중 input feature의 attribution을 파악하는 다양한 기법들이 있습니다. attribution이란 어떤 input feature가 모델의 예측에 영향을 끼쳤는지 나타내주는 값입니다.

하지만, 정량적으로 이들을 비교하고 성능을 평가하는 것은 쉽지 않습니다. 일반적인 모델들은 accuracy를 가지고 판단할 수 있지만, attribution을 생성하는 모델의 경우에는 정량적으로 평가하기 쉽지 않습니다.

2017년도 ICML에서 소개된 **'Axiomatic Attribution for Deep Networks'** 에서는 이런 문제의식을 가지고 연구를 진행했습니다. 

Attribution 방법론에서 정량적인 평가보다 정성적인 조건들에 대해서 정리해보고,이러한 요구조건에 부합하는 integrated gradient에 대해서 설명드리겠습니다. 또한 이를 위해서, baseline이라는 개념이 사용되는데 이에 대해서도 다룰 예정입니다.


## Attribution이란?

Attribution을 파악한다는 것은 모델의 input과 output간의 관계를 파악하는 것입니다. 즉, model이 예측을 할 때, 어떤 input feature가 해당 예측에 큰 영향을 주었는지 파악하는 것이 주요한 목적입니다.

본 논문에서 Attribution은 아래와 같이 정의될 수 있습니다. 주목할 점은 attribution을 구할 때, input과 baseline을 함께 사용한다는 점입니다.

- deep network: $F: R^n \rightarrow [0, 1]$
- input: $x = (x_1, \cdots, x_n) \in R^n$
- baseline input: $\acute{x} = (\acute{x_1}, \cdots, \acute{x_n}) \in R^n$

Attribution은 아래와 같이 수식으로 나타낼 수 있습니다.

$$
A_F(x, \acute{x}) = (a_1, \cdots, a_n) \in R^n
$$

$a_1, \cdots, a_n$은 feature importance와 유사한 개념입니다.


baseline이란, 일종의 비교대상입니다. 아래의 gradient 이미지를 보면, 직관적으로 받아들 일 수 있으리라 생각됩니다. 좋은 baseline이란, 모델에 대해서 중립적인 의미를 가지는 데이터포인트입니다.

예를 들어, object recognition에 경우에 input image의 어느 pixel이 특정 class라고 판단하게 하는지 구할 수 있습니다.  일반적인 경우에는 baseline 이미지는 zero pixel로 두어 구하기도 합니다. 자세한 사항은 아래의 gradient 부분에서 다루도록 하겠습니다.

![]({{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/ex1.png "Example1")



## Motivation: 정량적 평가가 아닌 Two Fundamental Axioms을 통해서 

 많은 attribution method의 문제는 평가하는 것이 어렵다는 것입니다. 정량적으로 평가하는 방법들이 있지만, 한계가 명확합니다. attribution이 높은 feature들을 제거하면서 모델의 성능이 얼마나 떨어지는지 파악하기도 합니다. 언뜻 생각해보면, 상당히 직관적인 방법입니다. 하지만, 일부 feature가 제거된 모델의 성능이 떨어지는 이유가 단지 중요한 feature여서이기보다는 모델이 학습시키지 않은 분포의 데이터이기 때문에 성능하락이 나타날 수 있습니다.

아래의 이미지를 예시로 설명드리겠습니다. input image를 개로 분류하는데 주요한 feauture가 코라고 가정해보겠습니다. 그럼 아래의 이미지처럼 코 부분을 zero-pixel로 만들고 model metric이 떨어지는지 파악합니다. 하지만, 코부분이 zero-pixel인 이미지는 인공적으로 만들어낸 데이터이기 때문에 좋지 않은 방법이라는 것입니다.

![]({{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/remove_feature.png "Example2")

본 연구에서는 이런 한계를 극복하기 위해서 정량적인 평가보다는 정성적인 조건을 정의하고 이를 충족시키는 intergrated gradient라는 방법을 제안합니다.

그리고 그 두가지 조건은 아래와 같습니다.

- Sensitivity
- Implementation Invariance

### Sensitivity이란

baseline과 input과의 차이가 오직 하나의 feature이고 baseline의 예측과 input의 예측이 다른경우를 가정해보겠습니다. 이런 상황에서는 차이나는 feature가 모델의 예측에 영향을 끼쳤다고 생각할 수 있습니다.  이렇게 차이나는 feature는 non-zero의 attribution의 값을 가져야합니다. 그리고 이러한 조건이 만족된다면, sensitivity 조건을 만족하게 됩니다.


#### Gradients는 sensitivity를 충족하지 못합니다.

gradient는 model coefficient를 쉽게 알 수 있는 방법입니다. backpropagation을 통해서 작업하게 되면, 쉽게 input의 gradient값을 구할 수 있습니다. 하지만, sensitivity 하지 않다는 단점을 가집니다.

![]({{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/gradient.png "gradient")

위와 같은 함수가 있다고 가정해보겠습니다.  baseline $x=0$이 주어졌을때, input $x=2$의 gradient를 구해보겠습니다.  $x=2 $에서 함수는 평평하므로 gradient값은 0이 됩니다. 하지만, baseline을 고려해보면, 함수값의 차이는 1이 되므로, attribution은 non-zero여야 합니다.



### Impementation Invariance란?

서로 다른 network이지만 같은 input ~ output 관계를 가진다면, 두 network는 동일한 attribution을 가져야 합니다. 이러한 특성을 implementation invariance라고 합니다.

![]({{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/implementation_invariance.png)

gradient는 sensitivity하지는 않지만, chain rule이 성립하기 때문에, implementation invariant하다는 장점을 가지고 있습니다.  수식으로 살펴보겠습니다.

- model output: $f$
- model input: $g$
- network: $h$

$$
\frac{\partial f}{\partial g} = \frac{\partial f}{\partial h} \cdot \frac{\partial h}{\partial g}
$$



하지만, LRP 혹은 DeepLIFT와 같은 방법론은 chain rule이 성립하지 않습니다.


$$
\frac{f(x_1) - f(x_0)}{g(x_1) - g(x_0)} \ne \frac{f(x_1) - f(x_0)}{h(x_1) - h(x_0)} \cdot \frac{h(x_1) - h(x_0)}{g(x_1) - g(x_0)} \text{   for all  } x_1, x_0
$$


아래의 이미지는 이를 실험한 결과입니다.

![]({{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/figure7.png)

실험결과는 integrated gradient를 제외한 나머지 discreted gradient를 사용한 모델은 implementation에 따라서 attribution의 값이 달라진 것을 확인할 수 있습니다.

만약, implementation에 따라서 attribution이 달라진다면, input-output관계가 아닌 것에 영향을 받는 것입니다. 동일한 input-output 관계를 가진 두 모델이 서로 다른 attribution을 보여준다면, 신뢰도 있는 해석은 기대하기 힘듭니다.


## Integrated Gradients를 소개합니다.

위에서 gradient가 implementation invariant하지만 sensitivity의 속성은 충족시키지 못하는 것을 살펴봤습니다.  integrated gradient는 이러한 한계를 극복하면서 implementation invariance를 유지하는 방법입니다.

![]({{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/integrated_gradient.png) 

직관적으로 살펴보면, integrated gradient는 baseline에서 input까지의 모든 gradient를 고려하는 방법입니다. 결과적으로 각 path의 gradient를 모두 고려하므로 특정 지점에서 gradient값이 0이 되는 이슈를 해결할 수 있으면서, gradient를 활용하므로 implementation invariance합니다.


$$
IntegratedGrads_i(x) = (x_i - \acute{x}_i) \times \int_{\alpha=0}^1 \frac{\partial F(\acute{x} + \alpha(x-\acute{x}) )}{\partial x_i} d\alpha
$$


### Integrated Gradients여야 하는 이유

이미지 인식에서 attribution method를 평가하는 방법으로 다음과 같은 방법이 있습니다.

- Attribution score가 높은 pixel을 제거해나가면서, 성능의 저하를 확인합니다.
- 좋은 attribution 방법이라면, model score가 급격히 떨어질 것으로 기대합니다.


하지만, 이런 방법은 문제가 있습니다. pixel을 제거한 이미지의 성능저하가 attribution이 좋아서 그런 것인지 아니면 처음 본 이미지 형식이라서 그런지 알 수 없다는 것입니다. 이런 이유로 해당논문은 수치적으로 attribution 방법을 증명하기 보다는 sensitivity와 implementation invariance의 특성을 만족하는 것으로 integrated gradient의 정당성을 설득합니다.


이런 문제의식을 바탕으로 해당 연구에서는 두 가지 단계의 논리를 전개합니다.

1. path method 소개
2. path method중에서 integrated gradient가 선택된 이유



### Path Methods는 implementation invariance합니다.

모든 path methods는**implementation invariance** 성질을 만족합니다.  또한 path method만이 sensitivity와 implementation invariance를 모두 만족할 수 있다고 주장합니다.

> Theorem 1 (Friedman, 2004))  
>
> Path methods are the only attribution methods that always satisfy
> Implementation Invariance, Sensitivity, Linearity, and Completeness.  

integrated gradient도 path method중 하나이며,  아래의 이미지에서 $P2$ linear combination의 path에 해당합니다. 아래의 그림처럼 비선형적인 path도 path methods 중 일부입니다.

![]({{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/figure1.png)

여기서 $\gamma$는 path를 정의하는 함수 입니다. $\gamma(0)$은 baseline을 의미하고, $\gamma(1)$은 input을 의미합니다. 아래 이미지의 예시에서는 feature의 차원은 2이고, path가 지나는 n개의 point가 있다고 가정했습니다. 

- $\gamma = (\gamma_1, \cdots, \gamma_n) : [0, 1] \rightarrow R^n$
- $\gamma(0) = \acute{x}$
- $\gamma(1) = x$

따라서, $\gamma$의 path integrated gradient는 아래와 정리할 수 있습니다.

$$
PathIntegratedGrads_i^\gamma(x) = \int_{\alpha=0}^1 \frac{\partial F(\gamma(\alpha))}{\partial \gamma_i(\alpha)} \frac{\partial \gamma_i(\alpha)}{\partial \alpha} d\alpha
$$


### 많은 Path method중에서 Integrated Gradients: Symmetry-Preserving

$x, y$가 함수 $F$에 대해서 대칭이라면, 다음과 같이 나타낼 수 있습니다.

$$
F(x, y) = F(y, x)
$$

attribution method는 다음과 같은 조건이 지켜지면 symmetry preserving하다고 표현합니다.

- input: symmetric variable이 모두 동일한 값을 가진다.
- baseline: symmetric variable이 모두 동일한 값을 가진다.
- 각 symmetry variable은 모두 같은 attribution을 가져야한다.



다음과 같은 예시를 통해서 이해해보겠습니다.
$$
Sigmoid(x1 + x2, \cdots)
$$
$x_1, x_2$는 symmetric variable이고 input에서는 $x_1=x_2=1$ 이며, basline에서는 $x_1=x_2=0$입니다. symmetry preserving하다면, $x_1, x_2$에 모두 동일한 attribution 값이 나와야합니다.



그리고, integrated gradient는 이러한 조건을 만족합니다. integrated gradient 하에서는 path가 선형적이기 때문에, 동일한 값을 가지는 $x_1, x_2$는 동일한 attribution을 가지게 됩니다.

하지만, path가 비선형적이라면 이는 성립하지 않습니다.  아래의 수식의 $\frac{\partial \gamma_i(\alpha)}{\partial \alpha}$ 부분을 고려해보면 알 수 있습니다.  integrated gradient의 경우 선형적이기때문에 이 값이 상수값이며 모든 dimension에서 동일합니다. 하지만 비선형적인 path를 가진다면 이 값은 dimension마다 다른 값을 가지게 될 것입니다.


$$
PathIntegratedGrads_i^\gamma(x) = \int_{\alpha=0}^1 \frac{\partial F(\gamma(\alpha))}{\partial \gamma_i(\alpha)} \frac{\partial \gamma_i(\alpha)}{\partial \alpha} d\alpha
$$



아래의 증명은 non-straightline은 symmetry preserving하지 않다는 것을 반례를 통해서 보여주고 있습니다.

![]({{ site.baseurl }}/images/2020-05-05-Integrated-gradient-정리글/proof1.png)


### integrated gradient의 기타 특징들

#### Completeness란?

integrated gradient는 completeness라는 재밌는 특성을 가집니다.  integrated gradient로 나오는 attribution들을 모두 더하면, 결국은 두 모델의 예측값의 차이가 됩니다.

$$
\sum_{i=1}^n IntegratedGrads_i(x) = F(x) - F(\acute{x})
$$


참고로, Completness는 sensitivity를 내포하고 있습니다. Completeness는 함수값의 차이는 attribution의 합이어야 합니다. sensitivity는 한 feature가 예측값의 차이를 발생시켰다면 해당 attribution은 non-zero어야합니다. 

#### Linearity란?

$f_1, f_2$ 의 두 모델이 있다고 가정해보겠습니다. 그리고 이를 바탕으로 $f_3 = a \times f_1 + b \times f_2$를 만들었습니다. $f_3$에 대해서 attribution을 구하면, f1과 $f_2$의 attribution에서 각각 $a, b$만큼의 가중치를 부여해서 구할 수 있는 속성입니다.


## 실제로 Integrated Gradients 구하기

실제로 적용단계에서 integral gradient를 수식대로 구하는 것은 매우 높은 비용을 치루거나 불가능합니다. 따라서 아래와 같은 근사하는 방법을 사용합니다.  직관적으로는 gradient를 구하는 경로를 m등분하여 각 gradient의 평균값을 구하는 것입니다.
$$
IntegratedGrads_i^{approx}(x) = (x_i - \acute{x_i}) \times \sum_{k=1}^m \frac{\partial F(\acute{x} + \frac{k}{m} \times (x-\acute{x}) )}{\partial x_i} \times \frac{1}{m}
$$

- $m$: the number of steps in the Riemman approximation


## 마무리 하며...

이번 글에서는 Attribution method가 갖춰야할 주요한 조건들과 integrated gradient라는 기법에 대해서 알아보았습니다. 추후에 이어질 포스팅에서는 다양한 XAI기법들에 대해서 다룰 예정입니다.



## Reference

- [1] Mukund Sundararajan, Axiomatic Attribution for Deep Networks, ICML, 2017