---
title: "R Notebook"
output: 
  html_document:
    keep_md:  true
---

$$f(x;\theta) = L(\theta;x)= \prod_{i=1}^nf(x_i;\theta),\ \ \ \theta \in \Omega$$

$$\hat\theta_{mle}=max_{\theta}(\theta)=max_{\theta}\prod_{i=1}^nf(x_i;\theta)$$


$$\hat\theta_{mle}=max_{\theta}logL(\theta)=max_{\theta}\sum_{i=1}^nlogf(x_i;\theta)$$

$$Mle\ can\ be\ obtained\ by\ solving\ this\ equation\ \frac{\partial{l(\theta)}}{\partial{\theta}}=0,\ where\ l(\theta)=logL(\theta)$$

$$Actually\ need\ to\ check\ \frac{\partial^2l(\theta)}{\partial{\theta}}|_{\theta=\hat{\theta}}<0$$


$$Assume\ that\ X_1,...,X_n\ satisfy\ the \ regularity\ conditions\ (R0)-(R2),\ where\ {\theta}_0\ is\ the\ true\ parameter,\\ and\ further\ that\ f(x;\theta)\ is\ differentiable\ with\ respect\ to\ {\theta}\ in {\Theta}.\\ Then,\ \frac{\partial{l(\theta)}}{\partial{\theta}}=0\ is\ a\ solution\ \hat\theta_n \xrightarrow{P} {\theta}_0$$

$$(R0)-(R5)\ are\ satisfied,\ suppose\ further\ that\ 0<I(\theta)<\infty,\\ Then\ any\ consistent\ sequence\ if\ solutions\ of\ the\ MLE\ equations\ satisfies\\ 
\sqrt{n}(\hat\theta_n-\theta_0)\xrightarrow{D}N(0,\frac{1}{I(\theta_0)}$$


$$Applying\ the\ Taylor\ Expansion\ on\ l'(\hat\theta_n),\\
l'(\hat\theta_n)= l'(\theta_0)+l''(\theta_0)(\hat\theta_n-\theta_0)+\frac{1}{2!}l'''(\theta_n^*)(\hat\theta_n-\theta_0)^2\ when\ \theta_n^*\ is\ between\ {\theta_0} \&\ \hat\theta_n $$

$$Because\ l'(\hat\theta_n)=0,\\ (\hat\theta_n-\theta_0)=\frac{-l'(\theta_0)}{l''(\theta_0)+\frac{1}{2}l'''(\theta_n^*)(\hat\theta_n-\theta_0)}\\
\rightarrow \sqrt{n}(\hat\theta_n-\theta_0) = \frac{\frac{1}{\sqrt{n}}l'(\theta_0)}{\frac{-1}{n}l''(\theta_0)-\frac{-1}{2n}l'''(\theta_n^*)(\hat\theta_n-\theta_0)}= \frac{A_n}{B_n+C_n} $$

$$A_n \xrightarrow{d} N(0,I(\theta_0)),\\
B_n \xrightarrow{p} I(\theta_0),\\
C_n\ is\ from\ the\ condition\ (R5)\ \&\ \hat\theta_n\xrightarrow{p}\theta_0,\\
Bounded*0 \xrightarrow{p} 0\\
So,\ \frac{A_n}{B_n+C_n}=\frac{1}{I(\theta_0)}N(0,I(\theta_0))= N(0,\frac{1}{I(\theta_0)})$$

$$The\ joint\ pdf\ is\\
L(\theta;x_1,..,x_n)={\theta}^n exp(-\theta\sum_{i=1}^nx_i),\ for,\ i=1,..,n.\\
From\ the\ factorization\ theorem,\ Y_1 = \sum_{i=1}^nX_i\ is\ sufficient(does\ not\ depend\ on\ \theta).\\
The\ log\ of\ the\ likelihood\ is\\
l(\theta)=nlog\theta-\theta\sum_{i=1}^nx_i.\\
Then\ the\ MLE\ Y_2=\frac{1}{\bar{X}}=n/Y_1\ is\ a\ function\ of\ sufficient\ statistic\ Y_1.
$$
