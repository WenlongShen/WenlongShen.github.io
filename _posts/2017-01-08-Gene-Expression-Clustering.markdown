---
layout:     post
title:      "Gene Expression Clustering"
subtitle:   "基因表达聚类分析"
date:       2017-01-08
author:     "Wenlong Shen"
header-img: "img/bg/2017_1.jpg"
tags: 生物信息 2017
---

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=default"></script>

*各位基因，请和谐表达，勿聚众闹事...*

#### 物以类聚

芯片、测序仪等高通量工具把人类对于基因的研究带入了组学时代，大量数据喷涌而来，如何解读成了重中之重。所谓物以类聚，从茫茫表达谱中抽提类似的基因，进而研究其协调/差异表达的原因和机制，考察富集的功能和通路等，不失为组学研究的好方法。

#### 何为相似

如何评价**“相似”**成了首要问题。这里有几个常用的距离指标（其中c表示某特定条件）：

Euclidean distance | $$d_{ij}=\sqrt{\sum_c(e_{i,c}-e_{j,c})^2}$$
Pearson correlation | $$d_{ij}=1-\frac{\sum_c(e_{i,c}- \overline e_i)(e_{j,c}- \overline e_j)}{\sqrt{\sum_c(e_{i,c}- \overline e_i)^2 \sum_c(e_{j,c}- \overline e_j)^2}}$$
Uncentered correlation (angular separation, cosine angle) | $$d_{ij}=1-\frac{\sum_c e_{i,c}e_{j,c}}{\sqrt{\sum_c e_{i,c}^2 \sum_c e_{j,c}^2}}$$

不同的距离函数有着不同的应用背景，各有各的用处，各有各的偏性，我们也可以自己提出合适的评价指标。然而基因表达涉及到的因素太过复杂，没有哪一个函数能够适用于所有的分析。

#### 何以聚类

把相似的基因聚在一起的算法有很多种，常用的主要有：

hierarchical clustering | 根据数据点之间的距离远近，构造具有层次关系的嵌套树
k-means | 将数据点预设分成K簇，先随机选择，再不断逼近
self organizing map (SOM) | 假设数据点之间存在一定的拓扑结构，不断循环计算数据点的同时保持拓扑结构的映射关系

#### 聚来聚去

不管用什么算法，目前都缺少一种指标来评价究竟哪种最好，也许是因为根本不知道基因表达分类的真实情况，我们只是自己强行将它们分成不同类别罢了。数据分析始终是仁者见仁，智者见智的事，多换几种算法，多做几组图表总是好的，当然也有前人的一些经验可以借鉴：

* 一般来说单链接聚类效果最差，全链接聚类有可能好于平均链接聚类
* 在数据量较大、类别较多的情况下，SOM优于k-means
* Euclidean distance更适于log尺度的数据，Pearson correlation适于真实值

聚类作为数据挖掘常用的手段之一，在做基因表达分析时也因遵循数据处理的一般原则，比如考察数据是否正态分布，处理奇异点、边界值和缺失值等。
