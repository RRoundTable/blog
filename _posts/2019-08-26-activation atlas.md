---

title: "Activation atlas 정리글"
toc: true
branch: master
badges: true
categories: ['deeplearning', 'interpretability']
---



# Activation atlas 정리글



## Introduction

What have these networks learned that allows them to classify images so well?

네트워크가 classification을 잘하는 이유를 찾기 위해서 다음과 같은 시도를 하였다.

기본적으로 네트워크를 시각적으로 분석할려고 노력했다.

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figure1.png "Figure1")



- [individual neurons](https://distill.pub/2017/feature-visualization/)

  뉴런들을 독립적으로 시각화

- [Interaction between Neurons](https://distill.pub/2017/feature-visualization/#interaction)

  뉴런은 독립적으로 움직이는 것이 아니기 때문에 simple feature combination을 시각화함. 이런 시도는 문제점을 가지고 있었다.

  예를 들어, 수 많은 combination중 어떤 combination을 살펴봐야하는지 어떻게 알 수 있는가?

- [spatial activation](https://distill.pub/2018/building-blocks/#ActivationGridSingle)

  위의 질문에 대한 답은 activation을 시각화하는 것에 있다. 특정 input tensor에 대해서 activation되는 뉴런들의 combination을 시각화하는 것이다.

위의 접근방법은 hidden layer를 다루는데 탁월하지만, 치명적인 결함이 있다. 하나의 input에 대해서만 시각화한다는 점이다. 이는 각 network의 전반적인 시각적 분석을 하기 힘들다는 뜻이다.

해당 논문은 이런 문제의식을 가지고 "Activation Atlas"를 기획하였다.

이런 global view를 얻기 위한 방법으로는 다음과 같은 방법이 있다.

- [CNN code visualization](https://cs.stanford.edu/people/karpathy/cnnembed/)

- 

- ![]({{ site.baseurl }}/images/2019-08-26-activation atlas/tsne.jpg "t-sne")

  

  t-SNE 기반의 시각화 방법이다. 간략히 설명하면, 각 input tensor 혹은 activation value에 대해서 t-SNE로 mapping 시킬 좌표를 구하고 해당 좌표에 위와 같이 이미지를 시각화 하는 것이다.

activation atlas의 경우 위와 t-SNE와 유사한 방법을 이용하였으나, 주된 차이는 input tensor가 아닌 각 feature를 시각화하는 것에 있다. 각 feature를 위와 같이 시각화하면 feature간의 관계를 파악할 수 있다는 장점이 있다.

activation atlas는

- 각 feature의 관계를 잘 파악할 수 있다는 강점을 가지고 있으나,
- data distribution에 영향을 받는다는 단점 또한 가지고 있다. 



## Looking at single images

activation atlas를 살펴보기전에 activation vector를 시각화하는 [spatial activation](https://distill.pub/2018/building-blocks/)부터 살펴볼 것이다. 사용할 모델은 InceptionV1이이며, 시각화 과정은 다음과 같다.

1. feed the image into InceptionV1

2. collect activations

   여기서 수집한 activation은 단순한 vector이기 때문에 인간의 눈으로 해석하기 힘들다. 여기서 feature visualization이 필요하다. 단순하게 생각하면, [feature visualization](https://distill.pub/2017/feature-visualization/)은 model이 생각하는 특정 activation vector를 생성하는 image를 시각화한 것이다. 일반적으로 image를 activation vector로 바꾸는 흐름과 다르게 activation atlas에서는 activation vector에서 image를 재현하는 흐름으로 간다고 생각하면 된다.

InceptionV1은 convolution layers로 이루어져 있으므로, 각 layer마다 복수의 activation vector가 존재한다. (Filter의 수만큼) 또한 아래의 이미지 처럼 하나의 뉴런이 각 patch를 이동하면서 activation vector를 생성한다. (Parameter-sharing) 

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/filter.jpg "CNN filter")



그러므로, network에 input image를 넣으면 하나의 뉴런은 많은 수의 evaluation을 받는다. 우리는 이를 각 뉴런이 각 patch에 대해서 얼마나 활성화됐는지 평가할 수 있다.

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figure2.png "Figure2")

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figure3.jpg "Figure3")





## Aggregating Multiple Images

위의 방법론은 single image에 대해서만 접근한 것이다. 하지만, global view를 얻고 싶다면 어떻게 해야할까?

모든 이미지에 대해서 위의 방법론을 적용할 수도 있겠으나, 그러한 방법은 scale-up할 수 없으며 인간의 두뇌는 구조적인 정리없이 수많은 이미지를 모두 인지할 수 없다.

우선, 먼저 수많은 이미지로부터 activation value를 수집해보자. 이는 위와 동일한 방법을 반복하면 된다. 수집한 activation은 위와 동일하게 feature visualization을 적용한다.

이렇게 수집된 vector는 high-dimension(512 dim)의 성격을 가진다. 이를 dimensionality reduction방법론을 적용해서 2차원으로 mapping 하면 아래의 이미지처럼 나타나게 된다.

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figure4.jpg "Figure4")



Feature visualization을 적용할 때, regularization을 사용하였다.(ex: [transformation robustness](https://distill.pub/2017/feature-visualization/#regularizer-playground-robust)) 

다른 objective를 사용하기도 하였다. activation space $v$ 를 시각화 하기 위해서, point $x, y$의 activation vector $h_{x,y}$ 를 dot product하였다. ( $h_{x, y} \cdot v$)

해당 논문은 dot product에 cosine similarity를 곱하여 anlge을 강조하는 것이 효과적이라는 것을 발견하였다.
$$
\frac{(h_{x,y} .v)^{n+1}}{(\rVert h_{x,y}  \rVert  \cdot \rVert v \rVert)^{n+1}}
$$

> We also find that whitening the activation space to unstretch it can help improve feature visualization. ???

> cosine similarity
> $$
> similarity = \cos(\theta)= \frac{A\cdot B}{\rVert A \rVert \rVert B \rVert}
> $$
> ![]({{ site.baseurl }}/images/2019-08-26-activation atlas/similarity.png "Similarity")

각 activation vector마다 attribution vector를 구할 수 있다. attribution vector란, 각 calss에 대한 항목이 있으며 각 class의 logit에 영향을 받은 activation vector의 값을 근사한다. attribution vector는 주변 contex에 영향을 받는다.
$$
h_{x, y} \cdot \nabla_{h_{x_y}}logit_c
$$

- Class c logit: $logit_c$
- 해당 수식은 뉴런이 logit에 미치는 영향을 측정하는 것
- GradCam과 유사하지만, gradient spatial averaging을 사용하지 않고 gradient의 noise를 continuous relaxtion을 통해서 감소 시켰다.
- Code: https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/building-blocks/AttrSpatial.ipynb



![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figure5.png "Figure5")



위에서 보이는 이미지는 오른쪽의 feature space에 상단 좌측에 위치한 average attribution을 시각화 한 것이다. 위의 이미지는 모두 조금씩 다르지만, 비슷한 류의 동물의 형상을 하고 있다. 특히, 눈, 털, 코 등의 특징을 잡아내고 있다. 주의 할 점은 앞단의 레이어에서 실행하면, 상당히 혼란스러울수 있다는 점이다. (앞단의 레이어에서는 위와 같은 특징을 못잡을 수도 있다.)



![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figure6.png "Figure6")



위의 이미지는 좌측 하단의 위치한 average attribution vector이다. 위의 이미지와 다르게 바다 해변의 형상을 가지고 있다. 

> seashore class를 확인하기 위한 activation이 starfish나 sealion과 같은 class를 확인하는데도 쓰이는 것을 확인할 수 있었다.

위의 두 사례를 보았을 때, 해당 activation atlas가 유의미한 2차원 좌표(semantic)를 가지고 있음을 확인할 수 있다.



## Looking at Multiple Layers

위에서는 하나의 레이어에서 다른 object가 어떻게 시각화되는지 확인하였다면, 이번 세션에서는 유사한 object에 대해서 서로 다른 layer에서 어떻게 나타타는지 알아볼 것이다.

사용할 레이어는 다음과 같다.(강조된 부분)

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figure7.png "Figure7")

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figure8.png "Figure8")



위의 이미지는 cabbage class를 시각화한 것이다. 왼쪽에서 오른쪽으로 갈 수록 더 cabbage처럼 구체적이고 복잡해지는 것을 확인할 수 있다. 이는 해당연구에서 기대했던 바인데 이유는 다음과 같다.

- 뒷 단의 레이어일수록 receptive field가 크기 때문

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur9.png "Figure9")



위의 이미지는 sand와 water 그리고 sandbar의 이미지의 activation value를 나타낸 것이다. sandbar를 보면, 앞의 두 이미지를 합친 것과 유사해 보인다.



## Focusing on a Single Classification

이제부터는 network가 classification하는 것에 대해서 살펴볼 차례이다. 

예를 들어서, network가 어떤 과정을 거쳐서 'fireboat'라는 class로 결정하는지 살펴볼 것이다.



![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur10.jpg "Figure10")



먼저 last layer(mixed5b)를 살펴볼 것이다. 뚜렷하게 보이는 부분일수록 'fireboat'로 결정하는데 큰 기여를 한 activation이다. classification 전의 layer이기 때문에 'fireboat'와 매우 유사한 이미지들이 진하게 보인다는 것을 확인 할 수 있다.

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur11.png "Figure11")



아래의 이미지는 mixed4d의 이미지로 여러가지 부분의 조합으로 'fireboat'로 인식하고 있음을 확인할 수 있다. 각 부분들은 fireboat와 유사해 보이지 않지만, 위의 fireboat사진을 보면 이렇게 인식하는 이유를 이해할 수 있다.

fireboat를 보면 창문 + 기중기 + 물로 이루어져 있음을 알 수 있다. 해당 부분도 물, 기중기, 창문들로 이루어져 있다.

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur12.png "Figure12")





이러한 특성은 'fireboat'와 'streetcar'와 비교해보면 잘 알 수 있다. (조금 유사하지만 다른 object)

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur13.png "Figure13")



해당 이미지를 보면 streetcar는 기중기나 물에서는 약한 activation을 가지고 있으나 창문과 집에 대한 activation에서는 매우 강한 activation을 가진다. 반대로 fireboat는 물과 기중기, 창문에서는 강한 activation을 가지지만 집에 대한 activation에서는 약한 activation을 가지고 있음을 확인할 수 있다.



## Further Isolating Classes

특정 class에 기여하는 activation만을 확인하고 싶다면, 다른 activation을 완전히 제외할 수 있다. 이를 class-specific activation이라고 한다.



![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur14.png "Figure14")



- 스노쿨링 이미지

- Code: https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/activation-atlas/class-activation-atlas.ipynb

class activation atlas는 특정 class에 대해서 어떤 detector가 더 많은 기여를 했는지 명확하게 보여준다. 위의 스노쿨링 예시에서 강한 attribution만 보여주는 것이 아니라, strength가 약하더라도 해당 class에 전반적인 영향을 끼친 attribution도 보여준다. 특정 경우 우리가 보고 싶어하는 object와 매우 강하게 상관관계가 있는 object가 있다. (스노쿨러 - 물고기) 물고기는 우리가 보고 싶어하는 부분과 다른 부분이다. 따라서 적절한 filtering 방법이 필요하다



![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur15.png "Figure15")



위의 이미지 다른 필터링을 적용한 것이다. 위의 설명을 참고 바란다.

이제는 유사한 두 클레스를 비교해볼 것이다. (magnitude 기준으로)



![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur16.png "Figure16")



위의 이미지를 보면 두 클레스를 구분하기 힘들것이다. 아래의 이미지를 보면 도움이 될 것이다.





![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur17.png "Figure17")



>To help make the comparison easier, we can combine the two views into one. We’ll plot the difference between the attributions of the “snorkel” and “scuba diver” horizontally, and use t-SNE  to cluster similar activations vertically.

위에 주목할 점은 locomotive(기관차)가 스쿠버 다이버와 연관이 깊게 나온다는 것이다. 이를 바탕으로 다음과 같은 실험을 진행하였다.

스노쿨링 이미지에 조금씩 기관차 이미지를 사이즈 업하여 더한 것이다.



![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur18.png "Figure18")



해당 이미지를 보면 조금 더 하면 스쿠버 다이버의 softmax값이 올라가나 일정수준이 넘으면 기관차로 인식함을 알 수 있다. 아마도 기관차의 스팀이 그런역할을 한 것으로 보이며 이와 같은 feature를 multi-use feature라고 칭한다.(시각적으로 유사해 보여도 서로 다른 시각적으로 다른 class에 반응)

위와 같은 실험을 attack의 개념으로 1000여번을 진행했다.



![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur19.png "Figure19")

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur20.png "Figure20")

![]({{ site.baseurl }}/images/2019-08-26-activation atlas/figur21.png "Figure21")



위의 공격은 모든 클레스에 대해서 효과적인것은 아니었으나, 다섯개의 이미지에서 2개 정도로 target image로 인식하게 만들 수 있었다.

## Conclusion

- ### Surfacing Inner Properties of Models

- ### New interfaces

  - using AI to augment Human intelligence: 인간지능 보조
  - 이미지의 알파벳처럼 activation을 조합할 수 있다.
  - classification 모델을 generative model처럼 ...
  - Style transfer
  - Query large image datasets
  - Histogram
  - 새로운 데이터 셋 탐색





#### reference

- https://distill.pub/2019/activation-atlas/