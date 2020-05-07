---
title: "MLE 와 MAP에 대하여"
toc: true
branch: master
badges: true
comments: true
categories: ['interview', 'machine learning']
metadata_key1: MLE
metadata_key2: MAP
---



# MLE: Maximum Likelihood Estimation



*liklihood*란, 이미 주어진 표본적 증거로 비추어보았을 때, 모집단에 관해 어떠한 통계적 추정이 그럴듯한 정도를  의미합니다. 수식으로 나타내면, $P(D \mid W)$ 처럼 나타낼 수 있다. (D = observation, W = parameters) . 

동전던지기 예시를 생각해보면, 쉽게 이해할 수 있다. 일반적으로 동전 던지기를 해서 앞면이 나올 확률은 0.5라고 생각합니다. 하지만, 이는 우리가 가정한 값이지 실제의 값은 아닙니다. 이런 이유로 몇 번의 수행결과로 동전의 앞면이 나올 확률 $P(H)$를 정의하고자 합니다.

만약 동전을 100번 던졌을 때, 동전의 앞면이 56번 나왔다면 '동전의 앞면이 나올 확률'은 몇이라고 얘기할 수 있을까?  이 문제의 해답이 **Maximum Likelihood**를 구하는 것이다. 즉, observation이 주어졌을 때, 가장 그럴듯한 가설(혹은 parameter)를 찾는 문제가 **Maximum Likelihood Estimation**이다.

동전 던지기는 이항분포를 따릅니다. x는 100번 던졌을 때 앞면이 나온 횟수를 의미하고 p는 앞면이 나올 확률을 의미합니다.
$$
p(x) = \begin{pmatrix} n \\ x \end{pmatrix} p^x(1-p)^{n-x}
$$
likelihood는 다음과 같이 정의됩니다. ($X$는 앞면이 나온 횟수, $\theta$는 앞면이 나올 확률)
$$
P(X \mid \theta)
$$
아래는 likelihood의 그래프입니다.

![]({{ site.baseurl }}/images/2019-07-07-MLE Maximum Likelihood Estimation/binomial.png "Binomial distribution for coin toss")



앞면이 나올 확률이 0. 56인 지점에서 가장 높은 likelihood를 가지고 있음을 알 수 있습니다.

$$
\hat{\theta} = argmax_{\theta}P(X \mid \theta)
$$

$$
\hat{\theta} = 0.56
$$



## Classification: MLE와 KL-divergence, Cross Entropy

KL-divergence는 다음과 같이 정의됩니다.

$$
D_{KL}(P \rVert Q) = -\sum_{x \in X}P(x)\log{\frac{Q(x)}{P(x)}}
$$

$$
D_{KL}(P \rVert Q) = -\sum_{x \in X}P(x)(\log{Q(x)} - \log{P(x)})
$$

즉, Q distribution과 P distribution간의 차이를 나타내주는 역할을 합니다. 만약 두 분포간의 차이를 나타내는데 필요한 정보량이 많다면, 두 분포는 차이가 크다고 할 수 있습니다.

Cross entropy는 다음과 같이 정의 됩니다.

$$
H(p, q) = -\sum_{x \in X}p(x)\log{q(x)} = E_p[-\log{q}]
$$

실제 분포 $p$에 대해서 $-\log{q}$를 평균내는 것입니다. 만약, $q$가  p와 유사할 수록 cross entropy는 낮게 나올 것입니다.  사실 cross entropy는 kl-divergence에서 출발했다고 해석할 수도 있습니다. 아래 수식을 보면 $\log{P(x)}$는 실제 분포가 주어지면, constant term이 됩니다. 따라서 이를 제외하고 본 것이 cross entropy라고 해석할 수 있습니다.

$$
D_{KL}(P \rVert Q) = -\sum_{x \in X}P(x)(\log{Q(x)} - \log{P(x)})
$$

그렇다면, MLE와 KLD, cross entropy는 어떤 관계를 가질까요? 결론부터 말씀드리면, 이들은 본질적으로 같은 일을 합니다. 아래는 수식입니다.
$$
\begin{align}
\theta_{ML} =\arg\max_{\theta}P_{model}(X \mid \theta) = \arg\max_{\theta}E_{X \sim P_{data}}[\log{P_{model}(x \mid \theta)}]
\end{align}
$$


$$
D_{KL}(P \rVert Q) = E_{x \sim P_{data}}[\log{P_{data}(x)} - \log{P_{model}(x)}]
$$

$$
H(p, q) =  E_p[-\log{q}]
$$

KLD를 최소화한다는 것은 결국 cross entropy를 최소화하는 것과 같습니다. 그리고 cross entropy를 최소화 하는 것은 결국 likelihood를 최대화하는 것과 같습니다. 따라서 MLE가 하는 것은 모델이 추정한 데이터의 분포와 실제 데이터의 분포를 가장 유사하게 만들어주는 parameter를 찾는 것이라고 해석할 수 있습니다.



## Regression: MLE와 Method of Least Squares 

regression task에서는 MLE를 다음과 같이 정의할 수 있습니다.

$$
\begin{align}
\theta_{ML} = \arg\max_{\theta} P_{model}(Y \mid X;\theta)=\arg\max_{\theta}\sum_{i=1}^{m}\log{P_{model}(y_i \mid x_i;\theta)}
\end{align}
$$


$P_{model}$이 Gaussian distribution이라고 가정하겠습니다. 위의 log term은 아래와 같이 전개됩니다.

$$
\log{P_{model}(y_i \mid x_i;\theta)} = -m\log{\sigma} - \frac{m}{2}\log{2\pi}-\sum_{i=1}^m\frac{\rVert\hat{y}_i - y_i\rVert^2}{2\sigma^2}
$$

$\sigma^2$(분산)은 고정되어 있다고 가정하면, 결국  $\rVert\hat{y}_i - y_i\rVert^2$를 최소화하는 parameter $\theta$를 찾는 것이 MLE입니다. 이는 아래와 같이 결국 least square와 같은 일을 하게 됩니다.

Least squares는 모델에 input을 넣었을때 나오는 output이 실제 target y와 차이를 나타내는 수식이며 이를 가장 작게하는 파라미터 $\theta$를 찾는 것이 관건입니다.
$$
MSE = \frac{1}{m}\sum_{i=1}^m\rVert\hat{y}_i - y_i\rVert ^ 2
$$


# MAP: Maximum a Posterior Estimation

MAP는 MLE와 다르게, prior라는 assumption을 사용합니다.

posterior $P(w \mid D)$ 는 아래와 같이 정의됩니다. ($P(w)$는 prior $P(D \mid W)$는 likelihood)
$$
P(w \mid D) = \frac{P(D \mid w) P(w)}{P(D)}
$$
위의 동전 던지기 예를 다시 살펴보겠습니다.

만약 동전을 던졌을 때, 앞면이 나올확률이 0.5이라는 가정을 했다고 생각해봅시다. 그리고 실제 100번을 던졌을 때, 앞면이 70번이 나온상황을 보면 posterior는 다음과 같습니다.

$$
P(T=0.5 \mid E=0.7)=\frac{P(E=0.7 \mid T=0.5)P(T=0.5)}{P(E=0.7)}
$$

만약 Maximum a posterior estimation을 한다고 하면, prior를 일종의 variable로 설정하고 구할 수 있습니다.

$$
P(T=x \mid E=0.7)=\frac{P(E=0.7 \mid T=x)P(T=x)}{P(E=0.7)}
$$
결국은 가장 데이터에 어울리는 prior를 구하는 과정이 MAP라고 할 수 있습니다.



## Bayesian interpretation of Ridge regularization



![]({{ site.baseurl }}/images/2019-07-07-MLE Maximum Likelihood Estimation/lasso_ridge.png "Left: Lasso Right: Ridge")



regression task를 가정했을 때, residual sum of squares(RSS)가 최소화되는 parameter $\beta$를 찾아야합니다. 
$$
\hat{\beta} =\arg\min_{\beta}(y-X\beta)^T(y-X\beta)
$$

여기서 ridge regression은 다음과 같이 penalty term의 역할을 합니다. 위의 식과 다르게 RSS와 $\beta$의 크기도 함께 고려해야 하는 것을 의미합니다. ($\rVert\beta\rVert_2^2 = \beta_1^2 + \beta_2^2 + \cdots \beta_p^2$)  결국 model의 complexity를 제한 하는 역할을 하는 것으로 해석할 수 있습니다.

$$
\hat{\beta} =\arg \min_{\beta}(y-X\beta)^T(y-X\beta) + \lambda\rVert\beta\rVert_2^2
$$


ridge regression은 bayesian 관점에서 해석할 수 있습니다.  다음은 regression task에 gaussian distribution을 가정한 것입니다.
$$
y \mid X, \beta \sim \mathcal{N}(X\beta, \sigma^2I)
$$
frequentism에서는 $\beta$가 고정된 값이지만, bayesian에서 $\beta$는 prior distribution을 가진다.  $\beta$를 Normal distributioni으로 가정해보겠습니다.
$$
\beta \sim \mathcal{N}(0, \tau^2 I)
$$
posterior는 다음과 같이 전개 됩니다.
$$
\begin{align}
p(\beta \mid y, X)& \varpropto p(\beta)\cdotp(y \mid X, \beta) \\
&\varpropto \exp[-\frac{1}{2}(\beta-0)^T\frac{1}{\tau^2}I(\beta - 0)]\cdot\exp[-\frac{1}{2}(y-X\beta)^T\frac{1}{\sigma^2}(y-X\beta) \\
&=\exp[-\frac{1}{2\sigma^2}(y-X\beta)^T(y-X\beta) -\frac{1}{2\tau^2}\rVert\beta\rVert_2^2]
\end{align}
$$
위의 식을 통해서 maximum a posterior를 구할 수 있습니다.
$$
\begin{align}
\hat{\beta} &= \arg\max_{\beta}\exp[-\frac{1}{2\sigma^2}(y-X\beta)^T(y-X\beta) -\frac{1}{2\tau^2}\rVert\beta\rVert_2^2]\\
&= \arg\min \frac{1}{\sigma^2}(y-X\beta)^T(y-X\beta) + \frac{1}{\tau^2}\rVert\beta\rVert_2^2\\
&= \arg\min (y-X\beta)^T(y-X\beta) + \frac{\sigma^2}{\tau^2}\rVert\beta\rVert_2^2
\end{align}
$$
위의 식을 통해서 $\lambda = \frac{\sigma^2}{\tau^2}$으로 구할 수 있습니다.

참고로 lasso regularization은 prior distribution을 laplace distribution을 사용하면 위와 같이 유도할 수 있습니다.

**Reference**

- https://ratsgo.github.io/statistics/2017/09/23/MLE/
- [베이즈 정리와 MLE, MAP](http://databaser.net/moniwiki/pds/BayesianStatistic/베이즈_정리와_MLE.pdf)
- https://statisticaloddsandends.wordpress.com/2018/12/29/bayesian-interpretation-of-ridge-regression/