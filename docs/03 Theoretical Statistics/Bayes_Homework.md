


###1. $Beta(\alpha,\beta)$ 분포의 경우 $\alpha=\beta=1$이면, 그 beta분포는 (0,1) 사이의 Uniform distribution이 됨을 보이시오.


Beta 분포의 pdf는 아래와 같다.

$Beta(\alpha,\beta) \sim \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}{x}^{\alpha-1}(1-x)^{\beta-1},\ 0<x<1$

여기서 $\alpha, \beta$에 각각 1을 대입하면 pdf의 상수항 꼴인 $\Gamma()$ 부분들이 1로 계산되고, $x^0*(1-x)^0=1$의 꼴로 바뀌어 pdf가 $0<x<1$ 사이에서 1인 형태가 되는데, 이는 $Unif(0,1)$ 의 pdf와 같다.


###2. Binomical model $B(1,\theta)$ 에서 모수 $\Theta$ 에 대한 사전분포로 $\alpha=\beta=1$인 $Beta(\alpha,\beta)$ 분포를 가정하면, 사전분포가 Bayes estimator에 아무런 영향을 미치지 않을 것이라고 예상하는 이유는 무엇인가?


일반적으로 우리가 사전분포에 대해 아무런 지식이 없다는 상황을 가정할 때 우리는 $Unif(0,1)$분포를 사용하게 된다.

이는 모든 구간에 균등한 확률을 갖는 가장 기본적인 분포를 가정한 것인데, 이는 즉 사전분포에 대해 아무런 정보가 없기에 어떤 선택을 하더라도

균일하게 사건이 발생할 것이라는 믿음 (곧 아무 정보가 없어서 어떤 결과나 나오더라도 공평한 결과가 나올 것이라는 믿음) 즉 사전 믿음이 없음을 의미하는 것이므로, $Unif(0,1)$을 사전분포로 사용할 때 사전분포가 Bayes estimator에 아무런 영향을 미치 않을 것이라고 예상하는 것이다.





###3. 질문 2에 대한 답변에도 불구하고, 실제로는 Bayes estimator가 사전분포에 영향을 받음을 보이시오.


하지만 모수 $\theta$에 대한 Bayes estimator는 관찰된 표본으로부터 얻은 최대우도추정량과 사전분포의 평균의 가중평균으로 얻어진 결과로써,

위와 같은 Binomial case에서 Bayes estimator를 구할 때 $\alpha=\beta=1$로 두었을 때 Bayes esitmator는 우리가 사전에 대한 믿음이 아예 없다고 가정했을 때

예상되는 결과(사전믿음이 없으므로 이때의 Bayes estimator는 MLE와 같게 나와야 일리가 있을 것이다.)와는 다른 결과가 나오므로, 우리의 예상과는 벗어나는 결과를 보여준다.

질문 2의 케이스를 살펴보면 이 때의 $\theta$에 대한 Bayes estimator는 아래와 같이 표현된다.

$$(\frac{n}{\alpha+\beta+n})\frac{y}{n}+(\frac{\alpha+\beta}{\alpha+\beta+n})\frac{\alpha}{\alpha+\beta}=(\frac{n}{n+2})\frac{y}{n}+(\frac{2}{n+2})\frac{1}{2}$$

여기서 $y$ 는 $\theta$에 대한 충분통계량이다. 우리의 예상과는 다르게 MLE가 prior mean쪽으로 살짝 끌어 당겨지고, 여전히 prior에 대해 영향을 받고 있음을 알 수 있다. 

엄밀히 사전믿음에 아무런 영향을 받지 않으려면 위의 Bayes estimator는 정확히 $\frac{y}{n}$이 나와야할 것이다.


###4. Bayes estimator가 사전분포에 영향을 받지 않게 하는 방법 중의 하나가 $\alpha=\beta=0$으로 놓은 것임을 보이시오.

만약 우리가 위의 예시에서 $\alpha=\beta=0$으로 둔다면, **shirinkage** estimate는 MLE인 $y/n$으로 reduce 될 것이다. 

이러한 prior를 두는 것은 추론에 아무런 영향을 주지 않는다.

하지만 $Beta(0,0)$은 pdf의 꼴이 아니다.

###5. 질문 4에서 $\alpha=\beta=0$인 beta(\alpha,\beta)분포는 pdf가 되지 않음을 보이시오.

먼저 우리가 아는 Beta 분포에서 모수 $\alpha, \beta$의 모수공간은 0보다 큰 곳에서 존재하므로 pdf의 꼴이 될 수 없다.

또는 간단한 수식을 통해서 적분값이 상수꼴이 나오지 않음을, 또한 1의 값이 나오지 않음을 보일 수 있다.

$\int_0^1\frac{1}{\theta(1-\theta)}d\theta=\int_0^1\frac{1}{\theta}d\theta+ \int_0^1\frac{1}{1-\theta}d\theta$

 theta의 범위는 0과 1 사이이므로, improper integral에 의해 위의 적분을 아래와 같이 표현할 수 있다.
 
 $$\lim_{a\to\ 0+}\int_a^1\frac{1}{\theta}d\theta+\lim_{b\to\ 1-}\int_0^b\frac{1}{1-\theta}d\theta$$
 
 $$\lim_{a\to\ 0+}[log(\theta)]_a^1-\lim_{b\to\ 1-}[log(1-\theta)]_0^b$$
 
 $$=\infty$$
 
 고로 $Beta(0,0)$은 pdf가 아니다.
 
 
