
nav:
  - Home: index.md
  - Mathematical Statistics:
      - Likelihood: 000 CPA Data Analysis
      /likelihood.md

      # Likelihood Example

나는 김성완이다. 이 페이지는 텍스트와 LaTeX 수식 테스트용이다.

표본 평균은 $\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$ 로 정의된다.

정규분포를 가정하면 로그우도는 다음과 같다.

$$
\ell(\mu) = -\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2
$$

이를 최대화하면 MLE는 $\hat{\mu} = \bar{X}$ 이다.

git add .