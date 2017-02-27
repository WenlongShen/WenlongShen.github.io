---
layout:     post
title:      "Expectation Maximization Algorithm"
subtitle:   "最大期望算法"
date:       2017-02-26
author:     "Wenlong Shen"
header-img: "img/bg/2017_2.jpg"
tags: 算法 2017
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

*掐指一算，概率最大的模型最有可能出现...*

#### 最大似然

建立模型，求得其中各个参数、分布，是对生物学问题的数学解决之道。然而如何做参数估计？“最大似然（maximum likelihood）”给出了这样一种理论，即从模型总体随机抽取n组样本观测值后，最合理的参数估计应该使得从模型中抽取该n组样本观测值的概率最大。为此，我们可以定义这样一个函数：

$$L(\theta)=L(x_1,...,x_n;\theta)=\prod_ip(x_i;\theta)$$

称为参数$$\theta$$相对于样本集X的似然函数（likelihood function）。我们的目的就是求得使$$L(\theta)$$最大的参数$$\theta$$，即：

$$\hat \theta=argmaxL(\theta)$$

为计算方便，我们可以进一步使用对数似然函数：

$$H(\theta)=logL(\theta)=log\prod_ip(x_i;\theta)=\sum_ilogp(x_i;\theta)$$

到这里，我们就可以通过求导数（偏导数）的方式，解出该似然函数的极值点，从而得到相关参数$$\theta$$了。最大似然估计，可以看作是已知观测结果，反推未知参数条件的过程。举个例子，假设我们有两个灌铅骰子A和B，分别投掷100次，我们就可以通过投掷结果来反推出这两个骰子各自的偏性。不过现实生活往往缺少如此直接的证据，很多数据我们观测不到，比如我们可能知道一共投掷了200次骰子，知道每次掷出的结果，但我们可能忘记了投掷的顺序，一会儿A一会儿B，使得我们无法直接计算似然函数，这时，我们就需要用到EM算法。

#### E一步，M一步

至此，由于存在未知变量z，我们原有的似然函数公式会发生变化，同时利用Jensen不等式，我们可以得到如下推导：

$$
\begin{align}
\sum_ilogp(x_i;\theta)&=\sum_ilog\sum_{z_i}p(x_i,z_i;\theta)\\
&=\sum_ilog\sum_{z_i}Q_i(z_i)\frac{p(x_i,z_i;\theta)}{Q_i(z_i)}\\
&\ge\sum_i\sum_{z_i}Q_i(z_i)log\frac{p(x_i,z_i;\theta)}{Q_i(z_i)}\\
\end{align}
$$

可见，为使$$L(\theta)$$达到最大值，就是不断地固定$$\theta$$，调整$$Q(z)$$，再固定$$Q(z)$$，调整$$\theta$$，直至收敛。其中，$$\frac{p(x_i,z_i;\theta)}{Q_i(z_i)}$$值为常数，且$$\sum_z{Q_i(z_i)}=1$$，可得如下推导：

$$
\begin{align}
Q_i(z_i)&=\frac{p(x_i,z_i;\theta)}{\sum_zp(x_i,z;\theta)}\\
&=\frac{p(x_i,z_i;\theta)}{p(x_i;\theta)}\\
&=p(z_i|x_i;\theta)\\
\end{align}
$$

即一旦固定$$\theta$$，$$Q(z)$$的计算公式就是后验概率问题。我们总结出一般的EM算法的步骤如下：

1. 初始化分布参数$$\theta$$；
2. 重复E步和M步直至收敛：
* E-step：根据初始参数或者上一次迭代出来的参数值，计算未知变量的后验概率（即期望），作为新的估计值：

$$Q_i(z_i)=p(z_i|x_i;\theta)$$

* M-step：根据未知变量新的估计值，最大化似然函数以得到新的参数：

$$\hat \theta=argmax\sum_i\sum_{z_i}Q_i(z_i)log\frac{p(x_i,z_i;\theta)}{Q_i(z_i)}$$

不过需要注意的是，EM算法实际需要知道样本的概率分布，否则似然函数里的p无法得到...

#### 应用

对于贝叶斯和隐马等模型，未知变量是常有的事儿，EM算法在这些模型的建立过程中发挥了至关重要的作用。这里仅仅举一个基因表达聚类的简单例子，已知部分基因的表达谱，且采样数据符合广义高斯分布，未知变量是各个基因所属的类别，模型参数是高斯分布的均值、协方差矩阵等。这时，在参数估计的过程中，E步就是根据上一次迭代数据，将各个基因概率划分为各个类别，M步就是根据当前划分的类别重新计算各分布的参数。
