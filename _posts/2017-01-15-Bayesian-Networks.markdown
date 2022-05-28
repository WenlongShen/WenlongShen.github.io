---
layout:     post
title:      "Bayesian Networks"
subtitle:   "贝叶斯网络"
date:       2017-01-15
author:     "Wenlong Shen"
header-img: "img/bg/2017_1.jpg"
tags: 生物信息 算法 2017
---

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=default"></script>

*互相依赖又互相独立，生物网络在混沌之中维持着平衡，一切皆是概率，一切皆有分布...*

#### 万物皆网

转录调控、代谢通路、疾病诊断...有太多的生物学系统能够以网络的形式呈现，而面对各种变量间彼此因果、依赖、独立的关系，贝叶斯当仁不让成为首选之一。

将具有因果关系的两点用箭头连接，没有箭头就视为条件独立，依此构建出的有向无环图（directed acyclic graphs, or DAGs）再加上条件概率矩阵，即为贝叶斯网络。

#### 构造、学习、推断

作为图，贝叶斯网络中的节点条件独立于其所有非直接父节点，有一点Markov链的感觉，可以看作是Markov的非线性扩展。由于具有这样的特性，在贝叶斯网络中，联合条件概率分布可以写作：

$$P(x_1,x_2,...,x_n)=\prod_{i=1}^nP(x_i|Parents(x_i))$$

在生物研究中，往往不能知晓整个网络的全貌，或者具体的参数，大概会有以下四种情形：

结构 | 参数 | 方法
:-:|:-:|-
已知 | 完整 | 最大似然估计法（MLE）
已知 | 部分 | EM算法；Greedy Hill-climbing method
未知 | 完整 | 搜索整个模型空间
未知 | 部分 | 结构算法；EM算法；Bound contraction


