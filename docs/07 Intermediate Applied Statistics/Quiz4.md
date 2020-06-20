---
title: "R Notebook"
output: 
  html_document:
    keep_md:  true
---

##Intermediate Applied Statistics Quiz4

####1.(a)

Traffic Control Measure 설치 전 후의 관측한 Years가 다르기 떄문에 이 기간을 고려할 수 없으므로, $\bar{m_{i1}},\ \bar{m_{i2}}$간의 비교는 적절하지 않다.

####1.(b)


two-way ANOVA 역시 Trafiic Control Measure 설치 전 후의 Years간의 비교를 고려할 수 없으므로, 적절하지 않은 접근 방법이다.

####1.(c)

Poisson Regression model을 적용한다면 아래와 같은 식을 얻을 수 있다.

$log{\lambda_{ij}}=\lambda_0+l_i+\tau_j$

$L(\lambda_0,l,\tau;y_{ij})= \prod\frac{e^{-\lambda_{ij}(\lambda_{ij})^{y_{ij}}}}{y_{ij}!}$

$logL= -\sum{e^{(\lambda_0+l_i+\tau_j)}}+\sum{y_{ij}(\lambda_0+l_i+\tau_j)}-\sum log(y_{ij}!)$



```r
data <- matrix(0,16,3)


data[,1] <- c(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1)
data[,2] <- c(1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8)
data[,3] <- c(13,6,30,30,10,15,7,13,0,2,4,0,0,6,1,2)
data <- data.frame(data)
names(data) <- c("Traffic_Measure","Locations","Accidents")
data
```

```
##    Traffic_Measure Locations Accidents
## 1                0         1        13
## 2                0         2         6
## 3                0         3        30
## 4                0         4        30
## 5                0         5        10
## 6                0         6        15
## 7                0         7         7
## 8                0         8        13
## 9                1         1         0
## 10               1         2         2
## 11               1         3         4
## 12               1         4         0
## 13               1         5         0
## 14               1         6         6
## 15               1         7         1
## 16               1         8         2
```

```r
fit <- glm(Accidents~Traffic_Measure+Locations,family=poisson(link='log'),data=data)
summary(fit)
```

```
## 
## Call:
## glm(formula = Accidents ~ Traffic_Measure + Locations, family = poisson(link = "log"), 
##     data = data)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -3.0196  -1.9267  -0.4212   0.4995   3.2014  
## 
## Coefficients:
##                 Estimate Std. Error z value Pr(>|z|)    
## (Intercept)      2.87723    0.18421  15.619  < 2e-16 ***
## Traffic_Measure -2.11223    0.27336  -7.727  1.1e-14 ***
## Locations       -0.03086    0.03708  -0.832    0.405    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for poisson family taken to be 1)
## 
##     Null deviance: 154.691  on 15  degrees of freedom
## Residual deviance:  56.414  on 13  degrees of freedom
## AIC: 112.06
## 
## Number of Fisher Scoring iterations: 5
```

```r
exp(-2.112)
```

```
## [1] 0.1209957
```


parameter를 추정하면 위와 같은 식을 얻을 수 있으며, 계수의 해석은 
Traffic control measure 설치 후의 accident rate가 0.12 감소함을 의미한다.
