---
layout:     post
title:      "Support Vector Machine"
subtitle:   "支持向量机"
date:       2017-02-12
author:     "Wenlong Shen"
header-img: "img/bg/2017_2.jpg"
tags: 算法 2017
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

*给我几个支持向量，我可以支撑起整个超平面...*

#### 二分类

是与否，有跟无，调控非调控，变异非变异，乃至得病与否的判断，二分类是最直观的生物学问题。我们往往希望在基因检测之后，判断患者的病症、肿瘤类型，就好像在茫茫数据点中，用洪荒之力分出一个平面，这边是癌症，那边是健康。

Support Vector Machine是非常经典的二分类模型，其核心在于找到具有最大间隔的超平面（maximum-margin hyperplane）。由于只需要数个向量就可以描述一个边界，故这些向量被称为支持向量（support vector），这也是该算法名称的由来。

SVM的绝妙之处，在于将低维的线性不可分转变为高维的线性可分，这其中用到的便是核函数（kernel function），它仅仅是做隐式映射，并不显式地计算结果，这就基本避免了维度提升带来的计算爆炸。不过优点也往往就是缺点，复杂的数据转换以及难以阐述的超平面信息，是SVM被研究者诟病为black box的原因。

#### 还是二分类

SVM最早被研究用于二分类，然而现实生活中总是充斥着多分类问题。除了对于SVM算法的改进进行多分类研究外，我们也可以利用原本二分类器的组合达到目的：

* 一对多。即从整体中依次将各个类别识别出来。
* 一对一。即对每两个类别设计一个SVM（共需k(k-1)/2个），最后可利用投票或别的ensemble方法进行类别确定。
* 层次法。即不断地划分子类。
* 其他。
