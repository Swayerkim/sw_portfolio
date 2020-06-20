---
title: "R Notebook"
output: 
  html_document:
    keep_md:  true
---

##Intermediate Applied Statistics Quiz III

####1.



####2.(a) Write the likelihood functions

 X와 Y가 독립이므로 둘의 joint likelihood는 아래와 같다.
 
$X \sim B(m,p_1),\ Y \sim B(n,p_2)$ 
 
 $L(p_1,p_2)=\ p_1^x(1-p_1)^{(m-x)}p_2^y(1-p_2)^{(n-y)}$
 

####2.(b) Rewrite the likelihood function in terms of $\theta = log\frac{p_1(1-p_1)}{p_2(1-p_2)}\  and\  \eta = log\frac{p_2}{1-p_2}$


$p_1 = \frac{e^{\eta}}{1+e^{\eta}},\ p_2 = \frac{e^{\theta + \eta}}{1+ e^{\theta+\eta}}.$


$L(\theta, \eta) = (\frac{p_1}{1-p_1})^x(1-p_1)^m(\frac{p_2}{1-p_2})^y(1-p_2)^n$

$= (\frac{p_1/(1-p_1)}{p_2/(1-p_2)})^x(\frac{p_2}{1-p_2})^{x+y}(1-p_1)^m(1-p_2)^n$

$= e^{{\theta}x}e^{\eta(x+y)}(1+e^{\theta+\eta})^{-m}(1-p_2)^n$


####2.(c) Use the profile likelihood to get the mle of $\theta$

여기서 target parameter는 $\theta$일 것이고, $\eta$가 nuisance parameter이기 때문에 at each fixed value $\theta$에서 $\eta$의 mle를 구하는 것이 profile likelihood를 찾는 것이다

이를 계산해보면

$Let\ \frac{\partial{L(\theta,\eta)}}{\partial\eta} = {\theta}x + \eta(x+y) -mlog(1+e^{\theta+\eta}) - nlog(1+e^{\eta}) = 0$

허나 이 식은 MLE에 대한 closed form을 갖지 않으므로 numerical 한 방법으로 profile likelihood를 구할 수 있다.

$L(\theta)= max_\eta{L(\theta,\eta)}$.

$\theta$의 MLE는 invariance property에 의해 다음과 같다.

$\hat{\theta} = log\frac{x/(m-x)}{y/(n-y)}$.


####3.(a)-(e)


```r
set.seed(2020311194)
obs <- c(2.08, 2.6, 2.67, 2.7, 2.94, 3.08, 3.71, 4.66, 4.71, 5.2)
exp(mean(obs))
```

```
## [1] 31.03141
```

```r
iter <- 5000
esti <- rep(0,iter)

###Assume parent Normal###
for ( i in 1:iter) esti[i] <- exp(mean(rnorm(10,mean(obs),sd(obs))))
print(sd(esti))
```

```
## [1] 11.49627
```

```r
###Assume Exponential###
for ( i in 1:iter) esti[i] <- exp(mean(rexp(10,1/mean(obs))))
print(sd(esti))
```

```
## [1] 253.4486
```

```r
###Approximation method###

((exp(mean(obs)))^2)*(var(obs)/length(obs))
```

```
## [1] 110.7054
```

```r
###Bootstrap###
for ( i in 1:iter) esti[i] = exp(mean(sample(obs,10,replace=T)))
print(sd(esti))
```

```
## [1] 11.3162
```

```r
###Jackknife###
n = length(obs)
esti <- rep(0,n)
lxbar <- exp(mean(obs))
for ( i in 1:n) esti[i] <- exp(mean(obs[-i]))
sqrt((n-1)*mean((esti-lxbar)^2))
```

```
## [1] 10.28417
```

###3-(f)
Bootstrap 방법이 분석자에 따라 다른 값들을 갖긴 하지만, sample 관측에 어려움을 느끼는 등의 제약이 있을 때 쉽게 응용할 수 있는 방법이므로, ootstrap의 방법을 채택하는 것을 선호한다.
