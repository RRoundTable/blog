---
title: "word representation"
toc: true
branch: master
badges: true
categories: ['nlp']
---

# Word Representation 정리글

## Word2vec: distributed representation

> 분산 표현(distributed representation) 방법은 기본적으로 분포 가설(distributional hypothesis)이라는 가정 하에 만들어진 표현 방법입니다. 이 가정은 **'비슷한 위치에서 등장하는 단어들은 비슷한 의미를 가진다'**라는 가정입니다. 
>
> reference: https://wikidocs.net/22660



### 1. CBOW(Continuous Bag of Words)

주변에 있는 단어로 중간에 있는 단어를 예측하는 방법입니다. 중심단어를 예측하기 위해서 주변단어를 보는 범위를 window라고 합니다. 예를 들어서, 아래 이미지의 첫 번째 글에서 파란색 부분이 'fat', 'cat'이므로 window는 2입니다.

<img src="https://wikidocs.net/images/page/22660/%EB%8B%A8%EC%96%B4.PNG">

> 보통 딥 러닝이라함은, 입력층과 출력층 사이의 은닉층의 개수가 충분히 쌓인 신경망을 학습할 때를 말하는데 Word2Vec는 입력층과 출력층 사이에 하나의 은닉층만이 존재합니다. 이렇게 은닉층(hidden Layer)이 1개인 경우에는 일반적으로 심층신경망(Deep Neural Network)이 아니라 얕은신경망(Shallow Neural Network)이라고 부릅니다. 또한 Word2Vec의 은닉층은 일반적인 은닉층과는 달리 활성화 함수가 존재하지 않으며 룩업 테이블이라는 연산을 담당하는 층으로 일반적인 은닉층과 구분하기 위해 투사층(projection layer)이라고 부르기도 합니다.
>
> reference: https://wikidocs.net/22660

<img src="https://wikidocs.net/images/page/22660/word2vec_renew_2.PNG">



아래의 이미지를 보면 연산과정을 알 수 있다. 

- $x_{cat}$: word one-hot vector
- $W_{V * M}$:  input layer와 prejection layer사이의 matrix
- $W'_{M* V}$: projection layer와 output layer간의 matrix
- $V, M$: 단어의 개수, embedding 차원

<img src="https://wikidocs.net/images/page/22660/word2vec_renew_3.PNG">



<img src="https://wikidocs.net/images/page/22660/word2vec_renew_5.PNG">



loss function으로는 cross-entropy 함수를 사용하게 됩니다.



이제 역전파(Back Propagation)를 수행하면 W와 W'가 학습이 되는데, 학습이 다 되었다면 M차원의 크기를 갖는 W의 행이나 W'의 열로부터 어떤 것을 임베딩 벡터로 사용할지를 결정하면 됩니다. 때로는 W와 W'의 평균치를 가지고 임베딩 벡터를 선택하기도 합니다.



### 2. Skip-Gram

중간에 있는 단어로 주변의 단어들을 예측하는 방법입니다.

<img src="https://wikidocs.net/images/page/22660/word2vec_renew_6.PNG">



### 3. Negative Sampling



대부분 word2vec은 negative sampling을 함께 사용합니다. word2vec의 학습과정을 잘 살펴보면, 출력층에 있는 softmax 함수는 단어집합 크기의 vector내의 모든 값을 0과 1사이의 값이면서 모두 더하면 1이 되도록 바꾸는 작업을 수행합니다. 이는 그 단어가 주변단어와 전혀 상관이 없는 단어라도 똑같이 적용되는 부분입니다.

만약 마지막 단계에서 '강아지'와 '고양이'와 같은 단어에 집중하고 있다면, Word2Vec은 사실 '돈가스'나 '컴퓨터'와 같은 연관 관계가 없는 수많은 단어의 임베딩을 조정할 필요가 없습니다. 전체 단어집합이 아니라 일부 단어집합에 대해서만 고려해도 되지 않을까요?

>  '강아지', '고양이', '애교'와 같은 주변 단어들을 가져옵니다. 그리고 여기에 '돈가스', '컴퓨터', '회의실'과 같은 랜덤으로 선택된 주변 단어가 아닌 상관없는 단어들을 일부만 갖고옵니다. 이렇게 전체 단어 집합보다 훨씬 작은 단어 집합을 만들어놓고 마지막 단계를 이진 분류 문제로 바꿔버리는 겁니다. 즉, Word2Vec은 주변 단어들을 긍정(positive)으로 두고 랜덤으로 샘플링 된 단어들을 부정(negative)으로 둔 다음에 이진 분류 문제를 수행합니다.
>
> reference: https://wikidocs.net/22660

이는 기존의 다중 클래스 분류 문제를 이진 분류 문제로 바꾸면서도 연산량에 있어서 훨씬 효율적입니다.



**negative sampling은 word2vec을 만들때 현재 문장에 없는 단어를 전체 데이터셋에서 추출하는 방법이다. 목적 단어와 연관성이 없을 것이라고 추정되는 단어를 추출한다.**

- 자주 뽑히는 단어일 수록 연관성이 낮다고 본다. 흔한 단어일수록 목적 단어와의 관계는 강하지 않기 때문



아래는 한 단어 $w_i$가  negative sample로 뽑힐 확률이다. 

- $f(w_i)$ : $w_i$의 등장빈도

$$
P(w_i) =\frac{f(w_i)^{0.75}}{\sum_{j=0}^n (f(w_j)^{0.75})}
$$



#### word2vec의 한계점

- out of vocalbulary에 대해서는 word representation을 얻을 수 없다.
- infrequent words는 학습이 불안정하다.





## Glove

카운트 기반과 예측기반을 모두 사용하는 방법론입니다.

> - LSA: 카운트 기반, 단어의미 유추 성능이 떨어짐.
>
> - Word2Vec: 예측기반, 전체적인 통계정보를 반영하지 못함.





단어의 동시 등장 행렬은 행과 열을 전체 단어 집합의 단어들로 구성하고, window size내에서 k단어가 등장한 횟수를 기록하는 행렬을 말합니다. 이 행렬은 transpose를 해도 동일합니다. (대칭행렬)

> 예시
>
> I like deep learning
>
> I like NLP
>
> I enjoy flying

| 카운트   | I    | like | enjoy | deep | learning | NLP  | flying |
| :------- | :--- | :--- | :---- | :--- | :------- | :--- | :----- |
| I        | 0    | 2    | 1     | 0    | 0        | 0    | 0      |
| like     | 2    | 0    | 0     | 1    | 0        | 1    | 0      |
| enjoy    | 1    | 0    | 0     | 0    | 0        | 0    | 1      |
| deep     | 0    | 1    | 0     | 0    | 1        | 0    | 0      |
| learning | 0    | 0    | 0     | 1    | 0        | 0    | 0      |
| NLP      | 0    | 1    | 0     | 0    | 0        | 0    | 0      |
| flying   | 0    | 0    | 1     | 0    | 0        | 0    | 0      |



### Co-occurrence probability

동시 등장 확률 $P(k | i), P(k | i)$는 동시 등장 행렬로부터 특정 단어 i의 전체 등장 횟수를 카운트하고, 특정 단어 i가 등장했을 때 어떤 단어 k가 등장한 횟수를 카운트하여 계산한 조건부 확률입니다.



예를 들어서, 위의 동시등장행렬을 바탕으로 P('I'|'like')를 구해보자. 'I' 는 총 3번 등장하였다. 'I'와 'like'가 동시에 등장한 횟수는 2번이다. 따라서 P('I'|'like') 는 2/3이다.
$$
P(A|B) = \frac{P(A) \cap P(B)}{P(B)}
$$

> | 동시 등장 확률과 크기 관계 비(ratio) | k=solid  | k=gas    | k=water | k=fasion |
> | :----------------------------------- | :------- | :------- | :------ | :------- |
> | P(k l ice)                           | 0.00019  | 0.000066 | 0.003   | 0.000017 |
> | P(k l steam)                         | 0.000022 | 0.00078  | 0.0022  | 0.000018 |
> | P(k l ice) / P(k l steam)            | 8.9      | 0.085    | 1.36    | 0.96     |
>
> 위의 표를 통해 알 수 있는 사실은 solid가 등장했을 때 ice가 등장할 확률 0.00019은 solid가 등장했을 때 steam이 등장할 확률인 0.000022보다 약 8.9배 크다는 겁니다. 그도 그럴 것이 solid는 '단단한'이라는 의미를 가졌으니까 '증기'라는 의미를 가지는 steam보다는 당연히 '얼음'이라는 의미를 가지는 ice라는 단어와 더 자주 등장할 겁니다.



### Loss function

Embedding 된 중심단어와 주변 단어 벡터의 내적이 전체 코퍼스에서 동시 등장 확률이 되도록 모델을 학습시킨다.
$$
\mbox{doc product}(W_i, W_k) \approx P(k|i) = P_{ik}
$$

실제 학습에서는 아래의 식을 활용한다.

$$
\mbox{doc product}(W_i, W_k) \approx \log P(k|i) = \log P_{ik}
$$



- $X$ : 동시 등장 행렬(Co-occurrence Matrix)
- $X_{ij}$ : 중심 단어 i가 등장했을 때 윈도우 내 주변 단어 j가 등장하는 횟수
- $X_i$:$\sum_{j}X_{ij}$: 동시 등장 행렬에서 i행의 값을 모두 더한 값
- $P_{ik}$ :$P(k|i)$ = $\frac{X_{ik}}{X_i}$ : 중심 단어 i가 등장했을 때 윈도우 내 주변 단어 k가 등장할 확률
  Ex) P(solid l ice) = 단어 ice가 등장했을 때 단어 solid가 등장할 확률
- $\frac{P_{ik}}{P_{jk}}$ : $P_{ik}$를  $P_{jk}$로 나눠준 값
  Ex) P(solid l ice) / P(solid l steam) = 8.9
- $W_i$ : 중심 단어 i의 임베딩 벡터
- $W_k$ : 주변 단어 k의 임베딩 벡터



embedding vector의 목적은 단어간의 관계를 잘 표현하는데 있습니다. 위에서 살펴본 $\frac{P_{ik}}{P_{jk}}$ 를 목적함수에 사용합니다. 먼저 함수의 input과 output을 정의해봅니다.
$$
F(W_i, W_j, W_k) =\frac{P_{ik}}{P_{jk}}
$$

input에서 output을 도출하는 방법은 많겠지만, 해당 연구에서는 두 단어간의 차이를 input으로 넣는 것을 제안합니다.

$$
F(W_i - W_j, W_k) =\frac{P_{ik}}{P_{jk}}
$$

그리고  선형 공간에서 두 vector간의 유사도를 보기 위해서 dot product를 선택했습니다. 
$$
F((W_i -W_j) ^ T W_k) =\frac{P_{ik}}{P_{jk}}
$$

정리하자면 선형공간에서 단어의 의미 관계를 표현하기 위해 뺄셈과 내적(dot procut)를 활용했습니다.

여기서 함수 F는 중심단어와 주변단어의 선택기준이 무작위이기 때문에, 이 둘의 관계는 함수 F안에서 자유롭게 교환가능해야 합니다. 이 조건을 만족하기 위해서는 Homomorphism이라는 조건을 만족해야합니다.
$$
F(a+b) = F(a)F(b) \ \forall a, b \in R
$$

> In [algebra](https://en.wikipedia.org/wiki/Algebra), a **homomorphism** is a [structure-preserving](https://en.wikipedia.org/wiki/Morphism) [map](https://en.wikipedia.org/wiki/Map_(mathematics)) between two [algebraic structures](https://en.wikipedia.org/wiki/Algebraic_structure) of the same type (such as two [groups](https://en.wikipedia.org/wiki/Group_(mathematics)), two [rings](https://en.wikipedia.org/wiki/Ring_(mathematics)), or two [vector spaces](https://en.wikipedia.org/wiki/Vector_space)).

 Homomorphism의 조건하에서 a와 b가 각각 vector라면 scalar값이 나올 수 없지만 내적값이라고 하면 scalar값이 나올 수 있습니다.

v1, v2, v3, v4 모두 vector입니다.
$$
F(v1^T v2 + v3^Tv4) = F(v1^T v2) F(v3^Tv4) \ \ \forall v1, v2, v3, v4 \in V
$$
F는 두 vector의 차이를  받았기 때문에, 뺄셈에 대한 homomorphism으로 변경했습니다. 

(간단하게 덧셈 ~ 곱셈 관계를 뺄셈 ~ 나누기 관계로 치환하였습니다.)
$$
F(v1^T v2 - v3^Tv4) = \frac{F(v1^T v2)}{F(v3^Tv4)} \ \ \forall v1, v2, v3, v4 \in V
$$
이제 glove식에 적용해보겠습니다.
$$
F((w_i - w_j)^Tw_k) = \frac{F(w_i^T w_k)}{F(w_j^T w_k)}= \frac{P_{ik}}{P_{jk}}
$$
위의 식에서 조금 더 자세히 살펴보면,
$$
F(w_i^Tw_k) = P_{ik} = \frac{X_{ik}}{X_i}
$$
또한 좌변을 풀어쓰면,
$$
F((w_i - w_j)^Tw_k) = F(w_i^T w_k - w_j^T w_k) = \frac{F(w_i^T w_k)}{F(w_j^T w_k)}
$$
따라서 homomorphism을 형태와 일치하게 됩니다.



그리고 이러한 조건을 만족시키는 함수는 지수 함수(Exponential function)입니다.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/Exp.svg/200px-Exp.svg.png">

- $x^{a + b} = x ^ a * x ^ b$
- $x^{a - b} = x ^ a / x ^ b$



이제 F를 $\exp$라고 해봅시다.
$$
\exp(w_i^T w_k - w_j^T w_k) = \frac{\exp(w_i^T w_k)}{\exp(w_j^T w_k)}
$$

학습의 안정성을 위해서, log를 사용합니다.

$$
w_i^T w_k  = \log P_{ik}=\log (\frac{X_{ik}}{X_i})=\log X_{ik} - \log X_i
$$

위의 식은 homomorphism이 성립해야하지만, $\log X_i$ 때문에 성립하지 않게 됩니다. ($a - b \ne b-  a$)그래서 glove 연구팀은 $\log X_i$항을 bias $b_i$라는 상수항으로 대체합니다. 같은 이유로 $w_k$에 대한 bias $b_k$를 추가합니다.
$$
w_i^T w_k + b_i + b_k = \log X_{ik}
$$


$$
\mbox{loss function} = \sum_{m,n = 1}^V (w_m ^ T w_n + b_m + b_n - \log X_{mn})^2
$$

- $V$ : 단어집합의 크기
- $X_{ik}$ 이 0이 될수도 있으므로 $\log(1+X_{ik})$로 바꾼다.



또한 동시등장행렬이 희소행렬일 가능성이 높다. glove 연구진은 동시 등장 행렬에서 등장 빈도의 값 $X_{ik}$가 매우 낮은 경우에는 정보가 거의 도움이 되지 않는다고 판단합니다. 따라서 동시등장행렬을 바탕으로 가중치함수를 구상하게 됩니다.

가중치 함수의 그래프는 아래와 같습니다. 특정 값보다 크다면 모두 같은 가중치를 주게 됩니다.

<img src="https://wikidocs.net/images/page/22885/%EA%B0%80%EC%A4%91%EC%B9%98.PNG">
$$
f(x) = \min(1, (\frac{x}{x_{\max}})^{0.75})
$$
최종적으로 목적함수는 아래의 식과 같습니다.
$$
\mbox{loss function} = \sum_{m,n = 1}^V f(X_{\min})(w_m ^ T w_n + b_m + b_n - \log X_{mn})^2
$$



## FastText

FastText는 단어를 구성하는 Subwords(substrings)의 vector 합으로 단어 vector를 표현합니다. 이 방법은 typo(오식)가 있는 단어라 할지라도 비슷한 representation을 얻을 수 있으며, 새로운 단어에 대해서도 형태적 유사성을 고려한 적당한 word representation을 얻도록 도와줍니다.

자연어 처리에서 자주 등장하는 문제는 (1) out of vocabulary, (2) infrequent words(모호성) 입니다. word2vec에서는 앞/뒤에 등장하는 단어로 가운데 단어를 예측하게 학습함으로서 문맥이 비슷한 단어를 유사한 vector로 표현합니다.

