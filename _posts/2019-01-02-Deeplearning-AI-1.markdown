---
layout:     post
title:      "Deeplearning.ai 笔记（1）"
subtitle:   "Introduction to Deep Learning"
date:       2019-01-02
author:     "Wenlong Shen"
header-img: "img/bg/2019_1.jpg"
tags: 机器学习 读书笔记 2019
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

*该系列课程来自<a href="https://www.deeplearning.ai/" target="_blank">deeplearning.ai</a>可谓之业界翘楚，感谢<a href="https://github.com/fengdu78/deeplearning_ai_books" target="_blank">网络社区</a>共享了中文翻译，好好学习吧。*

#### Welcome

AI正在改变世界，深度学习正在改变数据处理工作，海量的医学和生物数据亟待深度挖掘。

#### What is a Neural Network

房价预测可以看作是一个经典的例子，利用回归，我们可以在房子面积和价格间拟合出一条不错的函数曲线。类似的，我们将面积作为输入，价格作为输出，函数作为输入到输出的中间结点，即神经元，我们就得到了一个最简单的神经网络。进一步扩展，我们可以引入更多的特征值作为输入，引入更复杂的神经元函数，将一个神经元的输出作为下一个神经元的输入等等，这样，我们就构建了更为复杂的深度神经网络。

#### Supervised Learning with Neural Networks

“学习”的本质是“分类”，过程最好有“监督指导”。事实上，目前几乎所有好用的神经网络，都离不开监督学习。监督学习使得我们可以更准确地得到输入和输出之间的关系，可以更好地训练内部的神经网络。同时，更高的深度和卷积等核函数的应用，使得神经网络可以更好地处理非结构化数据，成为数据分析的强大工具。
