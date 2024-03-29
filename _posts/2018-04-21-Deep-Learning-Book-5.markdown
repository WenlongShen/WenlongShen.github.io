---
layout:     post
title:      "Deep Learning Book 学习笔记（5）"
subtitle:   "Machine Learning Basics"
date:       2018-04-21
author:     "Wenlong Shen"
header-img: "img/bg/2018_3.jpg"
tags: 机器学习 读书笔记 2018
---

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=default"></script>

*学而不思则罔，思而不学则殆。*

#### 学习算法

所谓机器学习，“学习”是什么？Mitchell提供了一个简洁的定义：“对于某个任务T和性能度量P，一个计算机程序被认为可以从经验E中学习是指，通过经验E得到改进后，它在任务T上由性能度量P衡量的性能有所提升。”

**任务T**从实际操作的角度来说，指的是机器学习系统应该如何处理样本，这些样本往往是从某些事件或实验中量化得到的特征（feature）集合，可以用向量的形式来表示。常见的任务主要有以下几种：

* 分类：判断某些输入属于哪一类，或是属于不同类别的概率分布，比如图像识别等。
* 回归：拟合出函数曲线，以根据输入预测结果。
* 转译：将非结构化数据如图像、声音等，转译为文本或其它包含有信息的编码形式。
* 机器翻译：自然语言互译，也是机器学习的重要应用之一。
* 异常检测：标记筛选出非正常或非典型的个体。
* 缺失值插入：根据整体特征，填补样本的缺失值。
* 去噪：将信号分类，并试图降低噪声。

**性能度量P**可以包含两个部分，一个是评价指标，比如准确率、错误率、AUC等，另一个是测试集。但事实上，我们一方面很难选择一个满足各种性能要求的评价指标，一方面也很难构建一个能完全模拟现实样本数据的测试集。所以我们往往需要多种指标，建立稳定的CV机制。

**经验E**大致指是否有监督，有监督即是存在标签（label）或目标（target），无监督则需要根据样本数据，显式或隐式地学习出其概率分布，它们之间的界线其实越来越模糊。同时，机器学习也不局限于在一个固定的数据集上进行训练，比如强化学习算法会有交互、反馈回路等。

#### 容量、欠拟合和过拟合

机器学习的挑战是需要在新的数据集上表现良好，这种能力被称为**泛化**（generalization）。通常，算法面对训练集会有一些误差，我们不断优化的一个目的就是降低这些误差，但同时，我们也希望泛化误差很低，一般地，我们会从训练集中分出一部分作为测试集，来评估机器学习模型的泛化误差。既然我们只能观测到训练集，如何才能影响测试集的性能？从理论上，我们需要独立同分布假设，即样本之间独立，且训练集和测试集同分布，在这样一个框架下才能允许我们从数学上研究训练误差和泛化误差之间的关系。

因而决定机器学习模型是否好有两个因素：1、降低训练误差；2、缩小训练误差和泛化误差之间的差距。这就对应了**欠拟合**（underfitting，模型不能在训练集上获得足够低的误差）和**过拟合**（overfitting，训练误差和泛化误差之间的差距过大）。这里再引入**容量**（capacity）的概念，指的是机器学习模型拟合各种函数的能力，容量低的模型可能会比较难拟合训练集，而容量高的模型由于存在过多不适合于测试集的训练集性质，可能导致过拟合，这里先提一个重要的解决办法：**正则化**（regularization），即降低泛化误差而非训练误差，其中权重衰减是其重要策略之一。

慢慢地，在实用中有了这么几个原则：奥卡姆剃刀（Occam's razor），最简单的往往最有效；没有免费午餐（no free lunch），没有哪个算法总是比别的好。

#### 超参数和验证集

大多数机器学习算法都存在超参数来控制算法行为，超参数的值不是算法本身学习出来的，尽管我们可以设计另一个嵌套的学习过程，即一个学习算法为另一个学习算法学出最优超参数，但这样容易得到的超参数往往趋向于最大可能的模型容量而导致过拟合。为解决这一问题，我们需要验证集（validation set），区别于测试集用于估计泛化误差，验证集是为了挑选超参数。通常80%的数据用于训练，20%用于验证。

将数据集分成固定的训练集和测试集会有个问题，即如果测试集过小，则其误差估计可能会有统计不确定性。为此，我们可以采用重复训练和测试的办法，常见的如k-折交叉验证，即将数据集分成k个不重合的子集，在第i次测试时，使用的第i个子集作为测试集，其他的数据作为训练集，整体测试误差可以估计为k次计算后的平均测试误差。

#### 估计、偏差和方差

统计学理论在机器学习领域中发挥着各种各样的作用，如参数估计、偏差和方差，对于正式地刻画泛化、欠拟合和过拟合等都非常有帮助。这里不细数公式，慢慢消化学习吧。要注意的是，偏差度量着偏离真实函数或参数的误差期望，而方差度量着数据上任意特定采样可能导致的估计期望的偏差。

#### 最大似然估计

我们需要有些准则能够帮助我们从不同模型中得到特定函数作为好的估计，常用的准则即最大似然估计。考虑一组含有$$m$$个样本的数据集$$X$$，独立地由未知的真实分布$$p_{data}(x)$$生成。令$$p_{model}(x;\theta)$$为算法模型学习出来的概率分布，对$$\theta$$的最大似然估计定义为：

$$\theta_{ML}=arg\ max\ p_{model}(X;\theta)$$

所以我们可以将最大似然估计看作是使模型分布尽可能地和经验分布相匹配的尝试。其另一个重要运用是估计条件概率$$P(y\vert x; \theta)$$，从而给定$$x$$预测$$y$$，这也是大多数监督学习的基础。

最大似然估计最吸引人的地方在于，它被证明当样本数目足够大时，就收敛而言是最好的渐近估计。同时，在合适的条件下（1、真实分布必须在模型可估计范围内；2、真实分布必须刚好对应于一个$$\theta$$值），最大似然估计具有一致性，意味着当样本数目趋向于无穷大时，参数的最大似然估计会收敛到参数的真实值。

#### 贝叶斯统计

前面讨论的是频率派统计，基于估计单一$$\theta$$值的方法，另一种考虑所有可能的$$\theta$$，将其已知知识表示成先验概率分布，属于贝叶斯统计。

相对于最大似然估计，贝叶斯估计有两个重要区别。第一，不像最大似然方法预测时使用$$\theta$$的点估计，贝叶斯方法使用$$\theta$$的全分布。第二，贝叶斯先验分布能够影响概率质量密度朝参数空间中偏好先验的区域偏移。实践中，先验通常表现为偏好更简单或更光滑的模型。当训练数据很有限时，贝叶斯方法通常泛化得更好，但是当训练样本数目很大时，通常会有很大的计算代价。

确定好先验后，我们需要继续确定模型参数的后验分布。原则上，我们应该使用参数$$\theta$$的完整贝叶斯后验分布进行预测，但从计算上往往非常复杂，这就需要单点估计来提供一个可行的近似解。一种合理的方案就是最大后验（Maximum A Posteriori）点估计。

#### 监督学习算法

粗略地说，对于一组输入$$x$$和输出$$y$$，有监督就是$$y$$被人为定义或分类了。通常的监督学习算法都是基于估计概率分布$$p(y\vert x)$$的，可以利用最大似然估计找到有参分布族$$p(y\vert x;\theta)$$最好的参数向量$$\theta$$。二元变量分布常见的一种解决方法是使用logistic sigmoid函数，这个方法被称为逻辑回归（logistic regression）。

支持向量机（support vector machine，SVM）是监督学习中最有影响力的方法之一类似于逻辑回归，但不同的是支持向量机只输出类别。SVM重要的特点是核技巧，将很多函数写成点积的形式，进一步替换为核函数。

其他简单的监督学习算法还有决策树（decision tree），在每个子节点与输入空间的一个区域相关联。

#### 无监督学习算法

无监督，所以算法只处理输入的特征值。其实监督和无监督算法并没有严格的区别，无监督更多地尝试从数据本身找到最佳的“表示”，进而分类。

主成分分析（PCA）就可以看作是学习数据表示的无监督学习算法。通过正交线性变换等，将数据变换为彼此之间不相关的表示。

另一个简单的算法是k-均值聚类，提供了k维的one-hot编码，其通过初始化k个不同的中心点，然后迭代交换下面两个步骤直到收敛：一、将每个训练样本分配到离其最近的中心点所代表的聚类中；二、将每个中心点更新为该类中所有训练样本的均值。

需要注意的是，聚类本身存在一个问题，即没有单一的标准衡量聚类的结果在真实世界中的性质如何，也是见仁见智吧。

#### 随机梯度下降

梯度下降是优化过程的重要算法，但其时间复杂度是$$O(m)$$，在实际应用中往往存在大规模样本，考虑到计算时间成本，我们常使用随机梯度下降，即从训练集中均匀抽出一小批量（minibatch）样本来估计梯度$$g$$，然后使用如下的梯度下降：$$\theta \leftarrow \theta - \epsilon g$$，其中$$\epsilon$$是学习率。

#### 构建机器学习算法

几乎所有的机器学习算法都可以看作由这几个因素构成：数据集、代价函数、优化过程和模型。一般地，优化算法可以看作是求解代价函数梯度为零的方程。代价函数通常至少要含有一项使学习过程进行概率估计的成分。最常见的代价函数是负对数似然，最小化代价函数导致的最大似然估计。代价函数也可能含有附加项，如正则化项。例如，我们可以将权重衰减加到线性回归的代价函数中。在某些情况下，由于计算原因，我们不能实际计算代价函数。在这种情况下，只要我们有近似其梯度的方法，那么我们仍然可以使用迭代数值优化近似最小化目标。

#### 挑战

前述的大都是简单的机器学习算法，并不能很好地解决人工智能领域核心的一些问题，如图像识别、语音识别等。深度学习发展的一部分动机就是传统学习算法在这类人工智能问题上泛化能力的不足。通常，神经网络不会包含针对特定任务的假设，因此神经网络可以泛化到更广泛的各种结构中。同时，深度的分布式表示带来的指数增益也能有效地解决维数灾难带来的挑战。
