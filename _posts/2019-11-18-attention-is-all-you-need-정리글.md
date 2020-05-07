---
title: attention is all you need 정리글
toc: true
badges: true
branch: master
categories: ['deeplearning', 'nlp', 'transformer']
---

# attention is all you need 정리글



## Transformer 구조



<img src="https://user-images.githubusercontent.com/27891090/69061756-2196f900-0a5d-11ea-9249-2401a60830d6.png" style="width:50%">



위의 이미지는 Transformer의 구조이다. encoder와 decoder 구조로 이루어져 있다. 마지막 encoder layer의 output이 각 decoder stack에 input으로 들어가게 된다. (**residual connection**) 각 encoder layer와 decoder layer는 모두 동일한 구조를 가지나 서로 parameter를 공유하지 않는다.  아래의 그림처럼 논문에서는 각 6개의 layer를 가지고 있다.



<img src="https://user-images.githubusercontent.com/27891090/69062040-aaae3000-0a5d-11ea-9d43-e5162aa70eb6.png" style="width:50%">

아래의 이미지는 encoder와  decoder의 세부 구조이다.

<img src="http://jalammar.github.io/images/t/Transformer_decoder.png" style="width: 80%">

각 세부 layer사이에는  Normalization 및  bias를 더하는 과정이 추가된다.



> ## Matrix Calculation of Self-Attention
>
> 이제  복수의 embeddinb vector를 matrix 연산으로 대체하는 과정을 살펴보자. 위의 그림과는 다르게 embedding vector가 matrix형태로 제공되어서 병렬 연산이 가능해졌다. 아래의 이미지 참고.
>
> <img src="http://jalammar.github.io/images/t/self-attention-matrix-calculation.png" style="width: 50%">
>
> <img src="http://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png" style="width: 50%">

##  Attention

<img src="https://user-images.githubusercontent.com/27891090/69063207-72a7ec80-0a5f-11ea-8e62-d4acc8bb5044.png">



### 1. Scaled Dot-Product Attention

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}}) V
$$



> Query, Key -  Value의 역할
>
> - Query: , matrix
> - Key: 각 embedding vector의 key, matrix
> - Value: 각 key가 가지고 있는 value, matrix
>
>  **추가적인 설명** 우선 query와 key, value에 대해서 설명하면 query가 어떤 단어와 관련되어 있는지 찾기 위해서 모든 key들과 연산한다. 여기서 실제 연산을 보면 query와 key를 dot-product한뒤 softmax를 취하는데, 의미하는 것은 하나의 query가 모든 key들과 연관성을 계산한뒤 그 값들을 확률 값으로 만들어 주는 것이다. 따라서 query가 어떤 key와 높은 확률로 연관성을 가지는지 알게 되는 것이다. 이제 구한 확률값을 value에 곱해서 value에 대해 scaling한다고 생각하면된다.

> **추가적인 설명** key와 value는 사실상 같은 단어를 의미한다. 하지만 두개로 나눈 이유는 key값을 위한 vector와 value를 위한 vector를 따로 만들어서 사용한다. key를 통해서는 각 단어와 연관성의 확률을 계산하고 value는 그 확률을 사용해서 attention 값을 계산하는 용도이다.
>
> reference:  https://reniew.github.io/43/ 




아래의 이미지는 scaled dot product attention과정의 일부이다. query, key 그리고 value는 각 $W^Q, W^K, W^V$matrix와 dot product를 진행한 결과이다.

<img src="http://jalammar.github.io/images/t/transformer_self_attention_vectors.png">

그리고  query와 key의 dot product의 결과를 $\sqrt{d_k}$만큼 scaling 해준다.

<img src="http://jalammar.github.io/images/t/self-attention-output.png">\



### 2. Multi-Head Attention

$$
MultiHead(Q, K, V) = Concat(head_1, \cdots, head_h) W^O \ \\\ where \ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

- $W_i^Q \in R_{d_{model} \times d_k}$
- $W_i^K \in R_{d_{model} \times d_k}$
- $W_i^V \in R_{d_{model} \times d_k}$

해당 연구에서는 multi head attention을 적용하였다. 이는 두 가지 방식으로 성능향상에 기여하였다.

- model이 다른 위치에 집중할 수 있는 능력을 향상시켰다.  “The animal didn’t cross the street because it was too tired” 과 같은 문장을 번역하는데 효과적인데 그 이유는 it이 가르키는 것이 무엇인지 중요하기 때문이다.
- layer multiple representation subspace를 제공한다. 복수의 Q, K, V matrix를 가지게 되고 이는 random하게 초기화된다.



아래의 그림은 두 개의 embedding vector(Thinking, Machines)의 복수의 head를 가지게 되는 과정을 시각화 한 것이다.

<img src="http://jalammar.github.io/images/t/transformer_self-attention_visualization_3.png">



## Representing The Order of The Sequence Using Positional Encoding



위에서의 attention 과정에서 word의 위치정보를 잃어버리게 된다. 이를 어떻게 복구할 것인가? 

이런 문제를 극복하기 위해서 Transformer에서는 input embedding vector에 특별한 vector를 더한다.  이는 각 word의 위치를 파악하는데 도움을 주거나 각 word의 distance를 구하는데 도움을 줄 것이다.

<img src="http://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png" style="width: 80%">

<img src="http://jalammar.github.io/images/t/transformer_positional_encoding_example.png">

아래의 이미지는 실제 20개의 word의 positional encoding의 시각화 결과이다. (512 dimension)

<img src="http://jalammar.github.io/images/t/transformer_positional_encoding_large_example.png">



$$
PE_{(pos, 2i)}=sin(pos/10000^{2i/d_{model}})
$$

$$
PE_{(pos, 2i+1)}=cos(pos/10000^{2i/d_{model}})
$$

- pos는 word의 위치를 나타낸다.
- i 는 dimension의 index를 나타낸다.

#### Reference

 http://jalammar.github.io/illustrated-transformer/ 