---
layout:     post
title:      "Deep Learning Book 学习笔记（12）"
subtitle:   "Applications"
date:       2018-05-23
author:     "Wenlong Shen"
header-img: "img/bg/2018_3.jpg"
tags: 机器学习 读书笔记 2018
---

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=default"></script>

#### 大规模深度学习

深度学习的基本思想基于联结主义：尽管机器学习模型中单个生物性的神经元或者说是单个特征不是智能的，但是大量的神经元或者特征作用在一起往往能够表现出智能。规模的大小对于神经网络来说至关重要，因此深度学习需要高性能的硬件设施和软件实现。目前比较流行的如GPU、分布式计算等，甚至还有了专门的硬件设备。

#### 计算机视觉

计算机视觉就是深度学习应用中几个最活跃的研究方向之一，包括多种多样的处理图片的方式以及应用方向，从复现人类视觉能力（比如人脸识别）到创造全新的视觉能力。计算机视觉通常不需要特别复杂的预处理，但图像都应该被标准化，一方面使得像素都在相同并且合理的范围内，一方面使图像尺寸适合于计算模型。

#### 语音识别

语音识别任务在于将一段包括了自然语言发音的声学信号投影到对应说话人的词序列上。早期比较成功的模型是隐马尔可夫模型（HMM）和高斯混合模型（GMM）。随着更大更深的模型以及更大的数据集的出现，通过使用神经网络代替GMM-HMM来实现将声学特征转化为音素（或者子音素状态）的过程可以大大地提高识别的精度。其中一个创新点是卷积网络的应用，卷积网络在时域与频域上复用了权重，改进了之前的仅在时域上使用重复权值的时延神经网络。这种新的二维的卷积模型并不是将输入的频谱当作一个长的向量，而是当成是一个图像，其中一个轴对应着时间，另一个轴对应的是谱分量的频率。

#### 自然语言处理

自然语言处理让计算机能够使用人类语言，为了让简单的程序能够高效明确地解析，计算机程序通常读取和发出特殊化的语言。而自然的语言通常是模糊的，并且可能不遵循形式的描述。自然语言处理中的应用如机器翻译，学习者需要读取一种人类语言的句子，并用另一种人类语言发出等同的句子。许多NLP应用程序基于语言模型，语言模型定义了关于自然语言中的字、字符或字节序列的概率分布。常见的模型如n-gram、神经语言模型等。

#### 推荐系统

信息技术部门中机器学习的主要应用之一是向潜在用户或客户推荐项目，这可以分为两种主要的应用：在线广告和项目建议，两者都依赖于预测用户和项目之间的关联。通常，这种关联问题可以作为监督学习问题来处理：给出一些关于项目和关于用户的信息，预测感兴趣的行为（用户点击广告、输入评级、点击“喜欢”按钮、购买产品，在产品上花钱、花时间访问产品页面等）。通常这最终会归结到回归问题（预测一些条件期望值）或概率分类问题（预测一些离散事件的条件概率）。
