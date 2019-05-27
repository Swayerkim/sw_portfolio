
###1-1. 강의노트 14페이지 참고, 왜 $\theta_3$에 대한 conjugate prior가 Gamma$(\alpha,\beta)$인지 설명하시오.


정규분포의 케이스에서 충분통계량은 $\bar{X}$와 $S^2$이다. 이들을 이용하여 likelihood를 살짝 변형하여 아래와 같이 표현할 수 있다.


$L(\theta_1,\theta_3|\mathbf{x}) \propto (\frac{\theta_3}{2\pi})^{n/2}exp[-{\frac{1}{2}} \{(n-1)s^2+n(\bar{x}-\theta_1)^2\}\theta_3]$

이는 다시 보면 감마분포인 $\frac{1}{\Gamma{(\alpha})\beta^{\alpha}}x^{\alpha-1}exp(-\frac{x}{\beta})$의 꼴로 표현이 가능함을 확인할 수 있고, 모수 $theta_3$에 대해 likelihood가 감마분포의 kernel로 이루어짐을 알 수 있다. 

고로 $\theta_3$에 대한 conjugate prior가 Gamma$(\alpha,\beta)$이다.



###1-2. $\theta_3$가 주어진 경우, $\theta_1$에 대한 prior로 $N(\theta_0,\frac{1}{n_0\theta_3})$를 사용하면 어떤 점이 편리해지는지 설명하시오.

$\theta_3$이 주어진 경우 $\theta_1$에 대한 prior를 위와 같이 정하면,

posterior joint pdf를 아래와 같은 꼴로 표현할 수 있다.

$k(\theta_1,\theta_3|\bar{x},s^2) \propto L(\theta_1,\theta_3|\mathbf{x})h(\theta_3)h(\theta_1|\theta_3)$

또한 이는 이어서 쓰면 

$\propto \theta_3^{\alpha+\frac{n}{2}+\frac{1}{2}-1}exp[-\frac{1}{2}Q(\theta_1)\theta_3]$

$where\ Q(\theta_1)=\frac{2}{\beta}+n_0(\theta_1-\theta_0)^2+[(n-1)s^2+n(\bar{x}-\theta_1)^2]$

posterior joint pdf가 위처럼 나올 시, 이는 $\theta_3$에 대한 Gamma kernel 꼴이므로, 

$\theta_1$에 대한 prior를 $N(\theta_0,\frac{1}{n_0\theta_3})$로 주는 것이 편리한 이유는

우리가 알고자하는 모수 $\theta_1$의 marginal pdf를 매우 편하게 구할 수 있기 때문이다.



###1-3. $\theta_1$과 $\theta_3$에 대한 posterior joint pdf가 14페이지 맨 아래의 식처럼 주어짐을 보이시오.

위에서 우리는 $h(\theta_1,\theta_3)=h(\theta_3)h_1(\theta_1|\theta_3)$

$\propto \theta_3^{\alpha-1}exp(-\frac{1}{\beta}\theta_3)(n_0\theta_3)^{1/2}exp(-\frac{n_0\theta_3}{2}(\theta_1-\theta_0)^2)$로 표현하였고,

이에 $g(\bar{x},s^2|\theta_1,\theta_3)$을 곱하여 joint posterior pdf를 다음처럼 표현할 수 있다.

$k(\theta_1,\theta_3|\bar{x},s^2) \propto g(\bar{x},s^2|\theta_1,\theta_3)h(\theta_1,\theta_3)$

$\propto (\theta_3)^{\frac{n}{2}}exp[-\frac{\theta_3}{2}\{(n-1)s^2+n(\bar{x}-\theta_1)^2\}] \theta_3^{\alpha-1}exp(-\frac{1}{\beta}\theta_3)(n_0\theta_3)^{1/2}exp(-\frac{n_0\theta_3}{2}(\theta_1-\theta_0)^2)$

$\propto (\theta_3)^{\alpha+\frac{n}{2}+\frac{1}{2}-1}exp[-\frac{\theta_3}{2}\{\{(n-1)s^2+n(\bar{x}-\theta_1)^2\}+n_0(\theta_1-\theta_0)^2+\frac{2}{\beta}\}]$

$\propto (\theta_3)^{\alpha+\frac{n}{2}+\frac{1}{2}-1}exp[-\frac{\theta_3}{2}Q(\theta_1)]$

$where\ Q(\theta_1)=\frac{2}{\beta}+n_0(\theta_1-\theta_0)^2+[(n-1)s^2+n(\bar{x}-\theta_1)^2]$




###2. Prior로 logistic distribution을 사용하고 있다. 만일 squared-error loss function을 사용한다면, $\theta$에 대한 Bayes estimator를 구하기 위해서는 적분을 두번이나 해야한다고 강의노트에 적혀있다. 왜 그런지 설명하고, 그 적분이 closed form으로 구할 수 없음도 설명하시오.

책의 예제는 다음과 같다.

$$X_1,...,X_n \stackrel{iid}{\sim} N(\theta_0,\sigma^2),\ \sigma^2\ is\ known$$

$$Y=\bar{X},\ a\ sufficient\ statistic\ for\ \theta$$

$Y|\theta \sim N(\theta, \frac{\sigma^2}{n})$

$\Theta \sim h(\theta)=\frac{1}{b}\frac{exp\{-(\theta-a)/b\}}{[1+exp\{-(\theta-a)/b\}]^2}$

where $-\infty<\theta<\infty,\ a\ and\ b >0\ are\ known$

이는 prior의 분포를 logistic분포로 가정한 것인데, 베이즈 정리를 통해 사후분포를 나타내면 아래와 같다.

$$k(\theta|y)=\frac{g(\theta|y)h(\theta)}{g_1(y)}=\frac{\frac{\sqrt{n}}{\sqrt{2\pi}\sigma}exp\{-\frac{1}{2}\frac{(y-\theta)^2}{\sigma^2/n}\}\frac{b^{-1}exp\{-(\theta-a)/b\}}{[1+exp\{-(\theta-a)/b\}]^2}}{\int_{-\infty}^{\infty}\frac{\sqrt{n}}{\sqrt{2\pi}\sigma}exp\{-\frac{1}{2}\frac{(y-\theta)^2}{\sigma^2/n}\}\frac{b^{-1}exp\{-(\theta-a)/b\}}{[1+exp\{-(\theta-a)/b\}]^2}d\theta}$$

즉 posterior pdf를 구하는 과정에서 분자의 Y와 $\Theta$의 joint pdf에서 Y의 marginal로 나눠주기 위해 $\theta$를 marginalize하는 과정의 적분이 포함된다.

이후 squared-error loss function으로 계산되는 $\theta$에 대한 Bayes estimate는 사후분포의 평균이기에, 이 평균을 계산하는 과정에서 필요한 적분과정이 하나 추가 되어 최종적으로 두번의 적분과정이 포함되게 되는 것이다.

$\delta(y)=E[\Theta|y]$

$={\int_{-\infty}^\infty}\theta k(\theta|y)d\theta$

$={\int_{-\infty}^\infty}\theta \frac{\frac{\sqrt{n}}{\sqrt{2\pi}\sigma}exp\{-\frac{1}{2}\frac{(y-\theta)^2}{\sigma^2/n}\}\frac{b^{-1}exp\{-(\theta-a)/b\}}{[1+exp\{-(\theta-a)/b\}]^2}}{\int_{-\infty}^{\infty}\frac{\sqrt{n}}{\sqrt{2\pi}\sigma}exp\{-\frac{1}{2}\frac{(y-\theta)^2}{\sigma^2/n}\}\frac{b^{-1}exp\{-(\theta-a)/b\}}{[1+exp\{-(\theta-a)/b\}]^2}d\theta}$

그리고 켤례사전분포는 이로 인해 연산되는 사후분포가 같은 family에 속하여 계산이 closed form으로 표현될 수 있는 것인데,

logistic distribution의 pdf는 normal과 conjugate를 이루지 않기 때문에 적분계산이 closed form으로 얻어지지 않는다.




###3. 강의 노트 18페이지에는 "inverse of the logistic cdf"를 구하였다. 왜 그것을 구해야 하는지 설명하시오.

inverse of the logistic cdf를 구한 이유는 $Unif(0,1)$를 만들어 샘플링을 하기 위해서이다. 

이를 구한 이유는, 위의 문제에서 Bayes estimate를 계산하는데 적분을 closed form으로 얻을 수 없으므로,

이를 해결하기 위해 $w(\theta)=f(y|\theta)=\frac{\sqrt{n}}{\sqrt{2\pi}{\sigma}}exp\{-\frac{1}{2}\frac{(y-\theta)^2}{\sigma^2/n}\}$으로 정의하여 Bayes estimate를 다음와 같이 표현한다.

$\delta(y)=\frac{E[\Theta w(\Theta)]}{E[w(\Theta)]}$

Monte Carlo 기법으로 위의 $\frac{E[\Theta w(\Theta)]}{E[w(\Theta)]}$ 로 확률수렴하는 

$\frac{m^{-1}\sum_{i=1}^m\Theta w(\Theta_i)}{m^{-1}\sum_{i=1}^mw(\Theta_i)}$를 계산하기 위해 $\Theta_i$들을 로지스틱 분포에서 추출해야하며, 이러한 과정을 위해 아래의 기법이 필요하다.



continuous random variable과 그것의 cdf 또한 연속인인 $X$를 생각해볼 때, 우리는 이것의 cdf $F(X)$의 **분포**가 $[0,1]$을 범위로 갖는 Uniform Distribution이라는 것을 알 수있는데, 간단히 수식으로 표현하면 아래와 같다.

 
위에서 설명한 확률변수의 누적확률분포를 $F(X)$라고 칭하자.
  
그렇다면 $0<x<1$을 만족하는 임의의 $x$에 대하여 아래와 같은 식전개가 가능하다.
  
  $$P(F(X) \leq x)$$
  
  $$=P(X \leq F^{-1}(x))$$
  
  $$=F(F^{-1}(x))$$
  
  $$=x$$
  
여기서 $X=F^{-1}(U)\  has\ distribution\ function\ F,\ where U~ Unif(0,1)$이다.


###4. WLLN가 뭔지 기술하고, WLLN가 어떻게 Bayes estimator를 시뮬레이션을 이용하여 구하는데 사용되었는지 설명하시오.

Let ${X_n}$ be a sequence of iid random variables having common mean $\mu$ and variance $\sigma^2<\infty$. Let $\bar{X_n}=\frac{\sum_{i=1}^nX_i}{n}$. Then 

$$\bar{X_n}\xrightarrow{p} \mu$$

WLLN는 표본의 크기가 커짐에 따라 표본평균이 모평균으로 확률수렴하는 것을 의미한다.

이러한 WLLN가 Bayes estimator를 시뮬레이션을 이용하여 구하는데 사용될 수 있는 이유는 squared error loss function 하에서 Bayes estimator가 사후분포의 기댓값이기 때문이고,

기댓값은 적분으로 표현할 수 있기 때문이다. 따라서 위에서 구한 logistic prior의 예시같은 적분 계산의 어려움을 해결하기 위해 충분통계량의 pdf를 $\theta$에 대한 함수 $w(\theta)$로 취급하여 $\frac{E[\Theta w(\Theta)]}{E[w(\Theta)]}$으로 $\delta(y)$를 나타내어 $\Theta_i$만 시뮬레이션 할 수 있으면, WLLN에 의해 m이 커짐에 따라 $\theta(y)$에 대한 consistent estimator로의 $\frac{m^{-1}\sum_{i=1}^m\Theta w(\Theta_i)}{m^{-1}\sum_{i=1}^mw(\Theta_i)}$를 계산할 수 있다.


###5. HMC 11.4.5

Suppose (X,Y) has the mixed discrete-continuous pdf

$$f(x,y)= \frac{1}{\Gamma(\alpha)}\frac{1}{x!}y^{\alpha+x-1}e^{-2y}, y>0,\ x=0,1,2,..,\ \alpha >0$$

1. Prove Y ~ Gamma($\alpha$,1)

$f_Y(y)= \sum_{x=0}^\infty \frac{1}{\Gamma(\alpha)}\frac{1}{x!}y^{\alpha+x-1}e^{-2y}= \frac{1}{\Gamma(\alpha)}e^{-2y}y^{\alpha-1}\sum_{x=0}^{\infty}\frac{1}{x!}y^x$,

$e^x=\sum_{k=0}^{\infty}\frac{1}{k!}x^k$이므로, 

$f_Y(y)=\frac{1}{\Gamma(\alpha)}e^{-y}y^{\alpha-1},\ y>0,\ \alpha>0$ 이고 이는 

$Gamma(\alpha,1)$의 pdf이다.


2. Prove X is negative Binomial 

$fX(x)= \int_0^{\infty}f(x,y)$

여기서 $y^{\alpha+x-1}e^{-2y}$는 $\Gamma(\alpha+x,\frac{1}{2})$의 kernel이다.

고로 $f_X(x)= \frac{1}{\Gamma(\alpha)}\frac{1}{x!}\Gamma{(\alpha+x)2^{-(\alpha+x)}}$

$= \frac{(\alpha+x-1)!}{x!(\alpha-1)!}2^{-(\alpha+x)}$이다.

이는 성공횟수가 x고 실패횟수가 $\alpha$, 성공확률이 $\frac{1}{2}$인 음이항분포의 pdf이다.


###6. 각자 Hogg 책 648p에 있는 gibbser2.s라는 프로그램을 실행하여, 유사한 결과가 나오는지 확인하시오.



```r
gibbser2 = function(alpha,m,n){ x0 = 1
yc = rep(0,m+n)
xc = c(x0,rep(0,m-1+n))
for(i in 2:(m+n)){yc[i] = rgamma(1,alpha+xc[i-1],2)
  xc[i] = rpois(1,yc[i])}
y1=yc[1:m]
y2=yc[(m+1):(m+n)]
x1=xc[1:m]
x2=xc[(m+1):(m+n)]
list(y1 = y1,y2=y2,x1=x1,x2=x2)
}

set.seed(2013122059)
result=gibbser2(10,3000,3000)
yhat <- result[['y2']]
xhat <- result[['x2']]

yse <- 1.96*(sd(yhat)/sqrt(3000))
yCI <- paste0('(',round(mean(yhat)-yse,3),',',
  round(mean(yhat)+yse,3),')')
              
        
xse <-1.96*(sd(yhat)/sqrt(3000))               
xCI <- paste0('(',round(mean(xhat)-xse,3),',',
  round(mean(xhat)+xse,3),')')

mean(xhat) ; mean(yhat)
```

```
## [1] 10.15433
```

```
## [1] 10.10725
```

```r
var(xhat) ; var(yhat)
```

```
## [1] 20.31862
```

```
## [1] 10.46886
```

```r
xCI ; yCI
```

```
## [1] "(10.039,10.27)"
```

```
## [1] "(9.991,10.223)"
```

책에 있는 gibbs sampler 프로그램을 실행하여 $\alpha=10,\  m=3000,\ n=6000$로 시뮬레이션을 해본 결과 lecture note의 값과 비슷하게 나오는 것을 확인할 수 있다.


