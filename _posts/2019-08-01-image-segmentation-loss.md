---
title: "Image segmentation에서 사용되는 loss"
---



# "Image segmentation에서 사용되는 loss"



## Basic: Cross Entropy

일반적으로 classification문제에서 사용되는 term이다.
$$
CE(p, \hat{p}) = -(p \log(\hat{p}) + (1- p)\log(1-\hat{p}))
$$

## Weighted cross entropy

$$
WCE(p, \hat{p}) = -(\beta p \log(\hat{p}) + (1- p)\log(1-\hat{p}))
$$

위의 cross entropy의 변형으로 모든 positive samples는 가중치를 부여 받는다. 이는 class imbalace문제를 다루기 위해서 사용한다. 만약 positive sample이 is 90%고 negative sample이 10%라면 cross entropy는 잘 작동하기 힘들다. 위와 같은 상황에서는  positive sample에 $\beta$ 를 1보다 작은 값을 부여하여 해결하고 만약 negative sample이 더 많다면 반대로 $\beta$ 에 1보다 큰 값을 부여하여 해결한다.



## Balanced cross entropy

위의 WCE와 유사한 방법론이다. 차이점은 negative sample에도 특정 가중치를 곱하는데에 있다.
$$
BCE(p, \hat{p}) = -(\beta p \log(\hat{p}) + (1-\beta)(1- p)\log(1-\hat{p}))
$$
주의할 점은 만약 $\beta$가 1의 값을 가진다면 negative sample을 전혀 고려하지 못하게 된다. 이는 $\beta$ 가 fixed value가 아닐때 발생하는데, 예시는 다음과 같다.

~~~python
beta = tf.reduce_sum(1 - y_true) / (BATCH_SIZE * HEIGHT * WIDTH)
~~~

이러한 문제는 small value $\epsilon$ 을 임의로 더해주거나 tf.clip_by_value를 이용하여 해결한다.

## Focal loss

주된 목적은 쉬운 example에 대해서는 down-weight를 부여하여 모델이 조금 더 어려운 example에 집중하는 것이다.
$$
FL(p, \hat{p}) = -(\alpha(1-\hat{p}) ^ \gamma p \log(\hat{p}) + (1-\alpha)\hat{p}^\gamma(1- p)\log(1-\hat{p}))
$$

- 만약 $\gamma$ 가 0이라면, BCE를 얻게 된다. 

  여기서 어려운 exapmle이란, 모델이 예측하기 어려운 class를 의미한다.

<수식 전개 및 코드 구현 작성할까>



## Distance to the nearest cell

cross entropy 개념에 거리 개념을 더하여 모델이 붙어있는 object에 대해서 더 잘 구별하도록 하는 것이 목적이다.
$$
BCE(p, \hat{p}) + w_0 \cdot \exp(- \frac{(d_1(x) + d_2(x))^2}{2 \sigma^2})
$$

- $d_1(x), d_2(x)$ 는 가장 가까운 cell과의 거리, 두 번째로 가까운 cell과의 거리를 나타낸다. 거리가 멀어질 수록 total loss는 줄어들고, 거리가 가까울 수록 total loss가 증가하게 된다.

- 여기서 cell은 다른 class를 의미한다.

  

# Overlap measures

## Dice Loss /F1 score

dice coefficient는 IOU와 유사한 term이다. 
$$
DC = \frac{2TP}{2TP + FP + FN} = \frac{2 \rvert X \cap Y \rvert}{|X| + |Y|}
$$

$$
IOU = \frac{TP}{TP + FP +FN} = \frac{\rvert X \cap Y \rvert}{|X| + |Y| - \rvert X \cap Y \rvert}
$$

​	여기서 $DC \ge IOU$ 라는 것을 알 수 있다.

Dice coefficient는 loss function으로 정의될 수 있다.
$$
DL(p, \hat{p}) = \frac{2<p, \hat{p}>}{\rVert p \rVert_2^2 + \rVert \hat{p} \rVert_2^2}
$$

~~~python
def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred)
  # some implementations don't square y_pred
  denominator = tf.reduce_sum(y_true + tf.square(y_pred))

  return numerator / (denominator + tf.keras.backend.epsilon()
~~~



## Tversky loss

dice loss의 generalization이다. FP와 FN에 weight를 곱하는 것이다.

$$
DL(p, \hat{p}) = \frac{2 <p, \hat{p}>}{2<p, \hat{p}> + <1-p, \hat{p}> + <p, 1 - \hat{p}>}
$$

$$
TL(p, \hat{p}) = \frac{2 <p, \hat{p}>}{2<p, \hat{p}> + \beta<1-p, \hat{p}> + (1-\beta)<p, 1 - \hat{p}>}
$$

## Lovasz-softmax

DL과 TL은 $\hat{p} \in \{ 0, 1\} ^n$ 이라는 강한 제한을 둔다. 하지만 Lovasz-softmax에서는 surrogate loss function을 사용한다.

> https://github.com/bermanmaxim/LovaszSoftmax/blob/master/tensorflow/lovasz_losses_tf.py



#### Reference

- https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/#references