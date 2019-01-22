---

#Ybigta 교육세션 ML 심화(1)

___

##The Basics of Decision Trees

&nbsp;&nbsp;&nbsp;&nbsp;Tree-based model은 단순하며 해석에 용이한 지도학습의 방법 중 하나이다. 순서는 Decision tree를 Regression과 classification 두 가지로 나누어 설명하며, 의사결정나무모형 이후에 Bagging과 random forests, boosting을 간단하게 설명하도록 할 것이다.


###Regression Trees


먼저 Regression Tree model로 간단한 예시를 *An Introduction to Statistical Learning with R(ISLR)*교재에서 가져와 보이겠다.

 ![](99photo_1.png)
 
  위는 *Hitters* 데이터 셋을 사용하였으며, 농구선수 들의 Salary를 선수들이 프로리그에서 경기한 햇수와 슛팅횟수를 통해 데이터를 분류한 간단한 사진이다.
  
  맨 윗줄의 분할 규칙을 보면(top split) 먼저 선수들의 집단을 프로리그에서 뛴 햇수가 4.5년 이상인지 이하인지로 구분을 함을 알 수 있다. 그 아래 node는 햇수로 선수들을 두 집단으로 구분한 뒤 , 슛팅횟수 117.5개를 기준으로 또 그룹 split을 함을 확인할 수 있다. 이 그림을 2차원의 평면 위에서 그림으로 표현한다면 아래와 같이 나타난다.
  
  ![](99photo_2.png)

&nbsp;&nbsp;&nbsp;&nbsp;위와 같은 세가지 regions에서 각각 region에 속해있는 player들의 mean response value를 구하는 것을 predicted Y로 설정하는 것이 기본적인 regression decision tree method의 매커니즘이다. 위의 그림에서 $R_1,R_2, R_3$의 region을 우리는 트리의 *terminal nodes*, 혹은 *leaves*라고 칭한다.

 위의 간단한 예시를 해석한다면 우리는 Salary를 결정하는 요소중 가장 중요한 factor는 Years라는 것을 알 수 있고, 리그에서 뛴지 4.5년이상이 된 플레이들 중에서는  이전 년도의 number of hits이 salary에 영향을 미친다는 것을 알 수 있다. 위 예시의 회귀 나무 모형은 Hits, Years, Salary간의 true relationship을 지나치게 단순화 하여 표현한 가능성이 높지만, 이는 회귀모형의 장점이며, 해석에 용이하고 시각화하기 편하다는 장점이 있다.
 
 
####Process of building regression tree
 
&nbsp;&nbsp;&nbsp;&nbsp;Rough하게 말해서 우리는 회귀트리를 만드는 것을 두가지 단계로 설명할 수 있다.

* predictor space를 set of possible values인 $X_1,X_2,...,X_{p}$가 $\mathbf{J}$ 개의 distinct한 non-overlapping 한 $R_1,R_2,...,R_{J}$구역에 속하게 정의한다.

* 모든 관측치는 region $R_{j}$에 속하며, $R_{j}$에 속하는 training observations들의 mean of response value로 예측을 시행한다.

 첫번째 Step에서 predictor Space가 two regions으로만 나뉘어져 있다고 가정을 해보고, $R_1$ 구역의 response mean이 10, $R_2$구역의 response mean이 20이라고 가정해보자. 여기서 Given observation이 $X=x$, if $x\in{R_1}$이면 우리는 10으로 예측을 하는 것이다.
 
 두번째 단계에서 우리는 regions을 어떻게 모양을 정의할 수 있을까? 실제로 region은 어느 모양으로도 형성될 수 있으며, 고차원에서는 우리가 생각하는 것 처럼 예쁜 직선으로만 경계를 나누지는 않을 것이다. 하지만 결국 region들의 모양이 어떻게 되던간의 우리의 목표는 RSS, 즉 잔차제곱합을 최소화하는 $R_1,..,R_{J}$를 찾는 것이다.
 
 
 $$\sum\limits_{j=1}^J\sum\limits_{i\in{R_j}}(y_{i}-\hat{y}_{R_{j}})^2$$
 RSS는 위처럼 표현되며, 여기서 $\hat{y}_{R_{j}}$는 mean response for the training observations within the *j*th box이다. 하지만 불행히도 **J** box에 대해서 가능한 모든 partition을 고려하는 것은 매우힘들다. 이러한 이유로 우리는 top-down, greedy 접근방법을 사용하는데 이는 *recursive binary splitting*으로 불린다. 
 
 Recursive binary splitting을 수행하기 위해 우리는 첫단계로 predictor $X_{j}$와 cutpoint *s*를 지정해주고 이는 predictor space를 $\{X|X_{j}<s\}$와 $\{X|X_{j}\ge{s}\}$ 로 나는데, 이때 predictor나 cutpoint설정은 RSS의 감소를 최대화하는 것으로 나누는 것이다. 이를 수식으로 좀 더 자세하게 표현하면 아래와 같이 표현가능하며, 이때의 *j*와 *s*를 찾는 것은 그 아래의 eqation을 최소화하는 j와 s를 찾는 것과 같다.
 
 $R_1(j,s) = \{X|X_{j} <s\}$ and $R_2(j,s) = \{X|X_{j} \geq s\}$,
 
 $\sum\limits_{i:x_{i}\in{R_{1}(j,s)}}(y_{i}-\hat{y}_{R_{1}})^2$ + $\sum\limits_{i:x_{i}\in{R_{2}(j,s)}}(y_{i}-\hat{y}_{R_{2}})^2$

 Input variables들의 사이즈가 크지않다면 위의 식을 구하는 것은 그렇게 오래 걸리지는 않을 것이다. 그 다음으로 우리는 이러한 과정을 계속해서 반복하여 모든 resulting regions내에서 RSS를 최소화하게 데이터를 분할하는 최적의 prediction 와 cutpoint를 찾으면 된다.
 
 만약 모든 regions이 정의가 되었다면, 우리는 새로 들어오는 test 관측치에 대해서 그 test 관측치가 속하는 regions의 train 관측치의 평균을 예측값으로 사용하면 된다.
 
 
 
####Tree Pruning 

 앞서 소개한 방식은 training set에 대해 좋은 예측값을 갖는 것 같지만, 자료에 대해 과적합할 가능성이 매우크고, test set에 대해 형편없는 결과를 가져오기도 한다. 이는 짠 트리모델이 너무 복잡하기 때문인데, 그래서 적은 분할을 한 작은 트리들은 편향을 줄이는 것을 희생하며 낮은 variance와 더 나은 용이한 해석을 이끈다.
 
 위와 같은 방식의 한가지 대안으로는, 각 분할에서 RSS의 감소가 특정 threshold를 넘는 split만 채택을 하여 tree를 줄이는 것을 예로 들 수 있다. 이런 대안은 조금 더 작은 tree를 만들지만, 이는 근시안적인 방법이다. 그 이유로는 트리형성의 각 초기단계에서는 형편없어 보이는 split이 나중에 tree전체를 만들어 놓은 이후 보았을 때는 large reduction in RSS를 가져올 수도 있기 때문이다.
 
 그러므로, 트리모형을 설계를 하면서 얻을 수 있는 더 나은 전략은, 먼저 가능한 매우 큰 나무 $T_0$을 만든 이후에, subtree를 얻기 위해 마지막에서 가지를 쳐내는 것이다. **어떻게 나무에 가지치기를 하는 것이 가장 좋은 방법일까?** 직관적으로 우리는 가장 낮은 test error rate를 갖는 subtree를 찾는 것이다. 
 
subtree를 얻은 후에, 우리는 교차검증 또는 validation set 접근을 통해서 test error를 추정할수 있다. 하지만 모든 subtree에 대해 너무나 많은 조합의 수가 존재하기에, 교차검증 error를 추정하는 것은 매우 소모적이고 비현실적이다.

 이를 위해 우리는 *Cost complexity pruning* 또는 *weakest link pruning* method로 알려진 방법을 사용할 수 있다. 모든 subtree의 경우의 수를 찾는 것이 아니고 음수가 아닌 튜닝 파라미터 $\alpha$로 인덱싱된 트리의 시퀀스를 고려해보는 것이다.
 
 알고리즘을 간단하게 표현하면 아래와 같은 단계로 이루어진다.
 
 ![](99photo_3.png)

 
* Recursive binary splitting을 통해 traninig data로 큰 트리를 만들면서, 각 terminal node가 일정 수준의 minimum level의 관측치 갯수를 갖게 된다면 트리형성을 멈춘다.
 
* 최적의 subtree 시퀀스를 얻기위해 $\alpha$의 함수를 이용하여 cost complecity pruning을 적용한다.

* K-교차검증을 이용하여 $\alpha$를 결정한다.

* 각 $\alpha$값들의 결과를 평균내어 average error를 최소화하는 $\alpha$를 채택한다.

 $\alpha$를 이용하여 RSS에 penalty를 부과하여 가지치기를 하는 것을 식으로 표현하면 아래와 같이 나타낼 수 있다.
 
 $$\sum\limits_{m=1}^{|T|}\sum\limits_{x_{i}\in{R_{m}}}(y_{i}-\hat{y}_{R_{m}})^2 + {\alpha}|T|$$
 
 앞서 일반적인 의사결정나무모형에서 RSS를 최소화하는 매커니즘은 똑같지만 $\alpha$에 대한 항이 더하기로 추가되었다. 여기서 $T$는 terminal nodes의 갯수이며, 
$\alpha$는 튜닝 파라미터이다. 이 $\alpha$는 subtree의 모델 complexity와 training data에 fitting하는 것과 trade-off한 관계를 갖는다. 
쉽게 설명을 하면 $\alpha$가 0이라면 위의 식은일반적으로 우리가 RSS를 구하는 식과 같으며 그럴때의 RSS를 최소화하는 식은 트리의 깊이를 최대한으로 깊게 만들어 모든 관측치들을 하나하나 terminal node로 삼는 경우일 것이다.

만약 모든 것이 같고 여기서 $\alpha$가 0이 아니라면, tree를 가장 깊게 뻗었을 때는 잔차제곱합에 추가로 $\alpha$항이 추가되었기 때문에, RSS를 최소화하는 경우와 일치하지 않게 된다. 덧붙혀 $\alpha$가 0에 근사한 작은 값이 아니고 적당히 큰 숫자를 갖는다면 잔차제곱을 나타내는 $\sum$안의 왼쪽 항보다 오른쪽항이 전체식에 영향을 더 크게 주기때문에 이럴 경우라면 terminal node의 수를 줄이는 것이(=make smaller subtree) 위 식의 quantity를 minimize 시킬 것이다. 

###Classification Trees

&nbsp;&nbsp;&nbsp;&nbsp; Classification tree는 regression tree와 양적 response가 아닌 질적 response를 예측한다는 것을 제외하고는 매우 유사하다. 회귀의 경우 관측치에 대한 response는 같은 terminal node에 속한 training 관측치의 평균 response로 예측을 하는데 반해, 분류의 경우에는 training 관측치가 속한 region에서 *most commonly occuring class*에 속한 개별 관측치를 예측한다.

 분류나무의 결과를 해석할 때는 특정 terminal node region에 상응하는 class prediction 뿐만 아니라, 각 region에 들어있는 training 관측치 사이에서 *class proportions* 또한 포함한다.
 
 트리를 만들 때에는 회귀의 경우와 비슷하게 **Recursive binary splitting**을 사용하지만, binary split의 기준이 RSS가 아닌 *classification error rate*를 사용한다. classification error rate는 단순하게 가장 공통적인 class에 속하지 않은 region에 포함된 training 관측치의 일부분으로 생각하면 된다.
 
 $$E= 1-max_{k}(\hat{p}_{mk})$$
 
 아래의 식에서 $\hat{p}_{mk}$는 *k*th class인 *m*번째 region안에 있는 training 관측치의 비율을 나타낸다. 하지만 기껏 설명했지만 이 분류에러비율은 tree-growing에 있어서 충분하게 sensitive하지 않아서 다른 두 가지 방법이 더 선호되는데 이는 우리가 잘 알고 있는 *Gini index*와 *entropy*가 있다.
 
 지니지수는 아래와 같이 표현되는데,
 
 $$G=\sum\limits_{k=1}^K\hat{p}_{mk}(1-\hat{p}_{mk})$$
 
 쉽게 말해서 이 인덱스는 node의 purity를 측정하는 것이며, 이 값이 작으면 이는 node가 대개 단일 클래스로부터 나온 관측치로 이루어져있다는 것을 의미한다.
 
 $$D=-\sum\limits_{k=1}^K\hat{p}_{mk}log\hat{p}_{mk}$$
 
 위의 식은 Entropy에 대한 설명인데, 이 또한 지니계수와 비슷하게 수치가 작을 수록 *m*번째 노드가 pure하다는 것을 의미한다.
 
&nbsp;&nbsp;&nbsp;&nbsp;분류에러비율과 지니계수, 엔트로피는 모두 나무에 가지치기를 할때 이용하지만 보통 일반적으로 최종 가지치기한 나무의 예측 정확성이 main goal이라면 분류에러비율을 criterion으로 사용하는 것이 선호된다.

&nbsp;&nbsp;&nbsp;&nbsp;간단하게 요약하면 Tree모형은 아래와 같은 장단점을 갖는다.

* 설명이 매우 용이하고, 어떨때는 선형회귀보다 쉽게 설명할 수 있다.

* 비전문가라도 tree의 사이즈가 너무 크지만 않다면 해석이 쉽고 시각화하여 볼 수있다.

* dummy variable들을 따로 생성하지 않아도 질적 predictors를 조작하기에 매우 쉽다.

* 하지만 이 책에서 다루는 다른 회귀나 분류를 다루는 방법들과 동일한 예측 정확성을 갖지는 않는다

* 가장 큰 단점으로는 트리모형은 아주아주 *non-robust*하다. 달리말해, 데이터가 조금만 바뀌어도 최종 예측되는 트리에 아주 큰 변화를 야기한다.

하지만 우리는 bagging, random forest, boosting 등의 방법으로 트리모형의 성능을 향상 시킬 수 있다!


##Bagging, Random Forests, Boosting

&nbsp;&nbsp;&nbsp;&nbsp;앞서 말한 것처럼 트리모델은 데이터셋이 조금만 달라도 아주 다른 결과값을 낳을 수도 있다고 언급했다. 이는 편향은 작지만 높은 분산을 갖는다.  

이러한 high-variance를 해결하기 위한 방법으로는 *bagging*, 즉 *bootstrap aggregation*을 통해 통계학습방법의 분산감소를 얻을 수있다.

$$\hat{f}_{avg}(x) = \frac{1}{B}\sum\limits_{b=1}^B\hat{f}^b(x)$$

부트스트랩을 이용하는 것은 위와 같이 표현할 수 있다. 풀어서 설명하면, 우리가 가지고 있는 training data set을 여러번 (많이 혹은 아주 많이) 복원추출을 통해 resampling하여 여러개의 표본 sample을 얻은 후, 이들을 각각 학습 알고리즘에 넣어 분류 혹은 회귀를 시행한 후 그 결과값(MSE등)의 평균을 통하여 prediction을 하는 방법이다. B개 만큼의 트리가 생기는 거고 이를 averaging하면 분산을 감소시키는 효과를 볼 수 있다.

여기서 number of trees **B**(resampling을 통해 그만큼 하나의 training set에서 여러개의 data set을 만들기에 그만큼 트리가 생기는 것) 는 bagging에서 엄청 중요시 여겨지는 파라미터는 아닌데, 그 이유는 B가 매우커져도 이것이 과적합과 직결되지는 않기 때문이다. B를 크게 늘린다는 것은 나무의 깊이를 깊게해 매우 많은 split을 통해 terminal node를 늘리는 것의 의미가 아니며 오히려 error를 더 안정화시키게 만든다. 


####Out-of-Bag Error Estimation
 
&nbsp;&nbsp;&nbsp;&nbsp;평균적으로 bootstrap을 진행하면 관측치의 2/3정도만 사용되어진다.

이론적으로 N개의 관측치에 data에서 N개의 표본을 복원추출하게 될 경우 각 데이터가 뽑힐 확률은 아래와 같고, 

$$1-(1-\frac{1}{N})^N$$

N이 매우 커지면 위 식은 $1-\frac{1}{e}$로 수렴하는데 이는 약 0.63정도이기에 관측치의 2/3정도만 확률적으로 사용되어진다는 뜻이다.

그렇다면 남은 관측치의 1/3은 bagged tree에서 fitting되는데 사용되지 못한다는 뜻을 의미하는데, 이를 *out-of-bag*(OOB)관측치라고 칭한다. 우리는 이러한 OOB샘플들을 활용하여 트리모형에서의 decision에 가중치를 조정할수도 있고, 분류모형에서는 오분류율을 추정하는 등 다양한 용도로 사용할 수있다.

특히 OOB error의 결과는 bagged model에서 test error를 추정하는데 있어서 한번도 fittingdㅔ 사용되지 않은 response들을 갖고 추정을 하기때문에 타당하다.

이러한 Bagging기법은 의사결정나무에 적용을 하면 가지치기 작업을 생략할수도 있게 해주고, error의 variance를 줄여준다는 장점이 있지만, single decision tree와는 다르게 어떠한 변수가 procedure에서 얼마나 중요한 영향력을 갖는지를 체크할 수 없다. 즉 Bagging은 **해석을 희생하여 예측의 정확도를 향상시키는 방법**이라고 말할 수 있다.

하지만 비록 단일 의사결정나무보다는 해석에 용이하지 않더라도, 회귀트리에서의 RSS와 분류트리에서의 Gini index등을 통해 전체적으로 개략적인 각 변수의 중요도는 가늠할 수 있다. 우리는 bagging에서  부트스트랩을 통해얻은 B개의 트리들을 평균함으로써 RSS나 gini index가 얼마만큼 감소했는지를에 대해 total amount를 체크할 수 있는데, 여기서 가장 큰 값들을 보이는 변수들이 중요한 변수들이 된다.

 ![](99photo_4.png)
 
 위의 그림은 분류트리의 예시인데, value값이 제일 큰 Thai, Ca, ChestPain이 largest mean decrease in Gini index를 갖는 변수이고 개중에 제일 중요한 변수라는 뜻을 의미한다.
 

###Random Forest 
