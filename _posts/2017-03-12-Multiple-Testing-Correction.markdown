---
layout:     post
title:      "Multiple Testing Correction"
subtitle:   "多重检验校正"
date:       2017-03-12
author:     "Wenlong Shen"
header-img: "img/bg/2017_4.jpg"
tags: 统计学 算法 2017
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

*这么多年了，被p值折磨得死去活来...*

#### 检验，多重检验

组学研究，实际把生命科学带入了大数据时代，对统计学的要求越来越高。面对一个又一个的样本，一个接一个的实验，万一科学假设太离谱，或者手抖做的糙，怎么能知道数据结果是不是碰巧成功的？为此，p值频频出现在各种组学数据分析中，RNA-seq、ChIP、GWAS等等等等，研究者最关注的都是**p<0.05**。看似美好的背后，却最容易忽略一个事实：p值是针对单次实验做假设检验，而组学数据往往是同时进行多次检验。比如RNA-seq寻找差异表达的基因，实际上进行的假设检验是：基因A是否有差异？基因B是否有差异？基因C是否有差异？...<a href="https://www.xkcd.com/882/" target="_blank">xkcd</a>有个漫画讽刺了这件事（我稍稍修改了原图排版）：
![xkcd](/img/post/2017_03_12_significant.jpg)
如果取0.05作为p的阈值，大概每20次实验就能得到一个具有显著性差异的结果，换句话讲，即便所有的实验结果实际上都是非显著的，但你依然有64.15%的可能性“观测”到至少一个“显著差异”的结果：

$$
\begin{align}
p(at\ least\ one\ significant\ result)&=1-p(no\ significant\ results)\\
&=1-(1-0.05)^{20}\\
&\approx0.6415\\
\end{align}
$$

而在组学数据中，要做数以万计的检验，其中可能出现的假阳性结果的比例不容小觑。

#### 调整p值

p值本身并没有错，其原本常用的阈值（0.05、0.01）也只是为了在假阳性率和假阴性率之间做平衡。只是在多重检验中，传统阈值变得过于松弛，于是研究者们提出了各种各样新的阈值方法，如校正p值（adjusted p-value）、q值（q-value）、错误发现率FDR（false discovery rate）等，计算方法虽不同，但其目的是一致的：让实验结果更为可信。

Bonferroni校正就是一个简单常用且暴力的方法，为控制整体（共$$n$$个检验）犯第一类错误的可能性保持在原阈值$$\alpha$$，则将$$\alpha/n$$作为单个检验的阈值。推导如下（其中$$p_1,...,p_n$$为每一检验相对应的p值，$$n$$为零假设总数，$$n_0$$为实际为真的零假设总数）：

$$FWER=P\left\lbrace\bigcup_{i=1}^{n_0}(p_i\le\frac{\alpha}{n})\right\rbrace\le\sum_{i=1}^{n_0}\left\lbrace P(p_i\le\frac{\alpha}{n})\right\rbrace\le n_0\frac{\alpha}{n}\le n\frac{\alpha}{n}=\alpha$$

Bonferroni方法提供了全局校正，可以看到，对于漫画中的例子，在Bonferroni校正下，$$p(at\ least\ one\ significant\ result)=1-(1-0.0025)^{20}\approx0.0488$$，完美地将阈值控制在0.05。然而这种方法在实际应用中过于严格，特别是对于大规模组学数据，可能导致假阴性率的上升。为此，我们可选用另一种阈值方法FDR。

FDR最早于1995年由Benjamini和Hochberg提出，其定义是“被错误拒绝的零假设数除以所有被拒绝的零假设数”的期望。他们的方法也很简单，针对给定的原阈值$$\alpha$$，将$$p_1,...,p_n$$按从小到大排序后，找到最大的正整数$$k$$，使得$$p_k\le\frac{k}{n}\alpha$$即可。

#### 性价比

除了上面介绍的两种方法外，还有诸如Storey的q-value等其它算法，有些做分布估计，有些有参数控制。其实无论是对于p值还是fdr之类，计算方法和阈值的选取从来都是为了考虑性价比，<a href="https://www.xkcd.com/1478/" target="_blank">xkcd</a>还有个漫画嘲笑了人们对p值的选取：
![xkcd](/img/post/2017_03_12_p.png)
假阳性和假阴性是大数据处理必须面对的问题，如何更好地取舍就仁者见仁智者见智了。
