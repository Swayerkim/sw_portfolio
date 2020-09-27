---
title: "R Notebook"
output: 
  html_document:
    keep_md:  true
---

$Suppose\ X_1,X_2,..,X_n \sim N(\mu, \sigma^2)$

$pivot\ random\ variable\ for\ a\ CI\ is\  t=\frac{ \bar{X} - \mu }{S/\sqrt{n}}$


Theoratical CI is $(\bar{x}-t^{(1-\alpha/2)}\frac{s}{\sqrt{n}},\bar{x}-t^{(\alpha/2)}\frac{s}{\sqrt{n}})$.

Pivot Bootstrap CI is $(\bar{x}-t^{*(1-\alpha/2)}\frac{s}{\sqrt{n}},\bar{x}-t^{*(\alpha/2)}\frac{s}{\sqrt{n}})$.

$t^* = \frac{\bar{x^*}-\bar{x}}{s^*/\sqrt{n}},\ s^{*2} = (n-1)^{-1}\sum_{i=1}^n(x_i^*-\bar{x}^*)^2$
