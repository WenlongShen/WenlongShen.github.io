---
layout:     post
title:      "Deep Learning Book 学习笔记（14）"
subtitle:   "Autoencoders"
date:       2018-06-07
author:     "Wenlong Shen"
header-img: "img/bg/2018_3.jpg"
tags: 机器学习 读书笔记 2018
---

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=default"></script>

自编码器是神经网络的一种，经过训练后能尝试将输入复制到输出。

#### 欠完备自编码器

将输入复制到输出听起来没什么用，但我们通常不关心解码器的输出。相反，我们希望通过训练自编码器对输入进行复制而使$$h$$获得有用的特性。从自编码器获得有用特征的一种方法是限制$$h$$的维度比$$x$$小，这种编码维度小于输入维度的自编码器称为欠完备自编码器。学习欠完备的表示将强制自编码器捕捉训练数据中最显著的特征。

当解码器是线性的且是均方误差，欠完备的自编码器会学习出与PCA相同的生成子空间。这种情况下，自编码器在训练来执行复制任务的同时学到了训练数据的主元子空间。因此，拥有非线性编码器函数和非线性解码器函数的自编码器能够学习出更强大的PCA非线性推广。

#### 正则自编码器

编码维数小于输入维数的欠完备自编码器可以学习数据分布最显著的特征。但如果赋予这类自编码器过大的容量，它就不能学到任何有用的信息。如果隐藏编码的维数允许与输入相等，或隐藏编码维数大于输入的过完备情况下，也会发生类似的问题。在这些情况下，即使是线性编码器和
线性解码器也可以学会将输入复制到输出，而学不到任何有关数据分布的有用信息。

理想情况下，根据要建模的数据分布的复杂性，选择合适的编码维数和编码器、解码器容量，就可以成功训练任意架构的自编码器。正则自编码器提供这样的能力。正则自编码器使用的损失函数可以鼓励模型学习其他特性（除了将输入复制到输出），而不必限制使用浅层的编码器和解码器以及小的编码维数来限制模型的容量。这些特性包括稀疏表示、表示的小导数、以及对噪声或输入缺失的鲁棒性。即使模型容量大到足以学习一个无意义的恒等函数，非线性且过完备的正则自编码器仍然能够从数据中学到一些关于数据分布的有用信息。

常见的有稀疏自编码器、去噪自编码器、收缩自编码器等。

#### 表示能力、层的大小和深度

自编码器通常只有单层的编码器和解码器，但这不是必然的，实际上深度编码器和解码器能提供更多优势。深度自编码器（编码器至少包含一层额外隐藏层）在给定足够多的隐藏单元的情况下，能以任意精度近似任何从输入到编码的映射。深度可以指数地降低表示某些函数的计算成本。深度也能指数地减少学习一些函数所需的训练数据量。

#### 随机编码器和解码器

自编码器本质上是一个前馈网络，可以使用与传统前馈网络相同的损失函数和输出单元。在随机自编码器中，编码器和解码器包括一些噪声注入，而不是简单的函数，这意味着可以将它们的输出视为来自分布的采样。

#### 去噪自编码器

去噪自编码器是一类接受损坏数据作为输入，并训练来预测原始未被损坏数据作为输出的自编码器。在某种意义上，去噪自编码器仅仅是被训练去噪的MLP。然而，其不仅仅可以学习去噪，还可以学到一个好的内部表示（作为学习去噪的副效用）。学习到的表示可以被用来预训练更深的无监督网络或监督网络。与稀疏自编码器、稀疏编码、收缩自编码器等正则化的自编码器类似，DAE的动机是允许学习容量很高的编码器，同时防止在编码器和解码器学习一个无用的恒等函数。

#### 使用自编码器学习流形

自编码器跟其他很多机器学习算法一样，也利用了数据集中在一个低维流形或者一小组这样的流形的思想。其中一些机器学习算法仅能学习到在流形上表现良好但给定不在流形上的输入会导致异常的函数。自编码器进一步借此想法，旨在学习流形的结构。

#### 收缩自编码器

收缩自编码器添加了显式的正则项，其与去噪自编码器之间存在一定联系，即在小高斯噪声的限制下，去噪重构误差与收缩惩罚项是等价的。换句话说，去噪自编码器能抵抗小且有限的输入扰动，而收缩自编码器使特征提取函数能抵抗极小的输入扰动。

#### 预测稀疏分解

预测稀疏分解是稀疏编码和参数化自编码器的混合模型。参数化编码器被训练为能预测迭代推断的输出。PSD被应用于图片和视频中对象识别的无监督特征学习，在音频中也有所应用。

#### 自编码器的应用

自编码器已成功应用于降维和信息检索任务。降维是表示学习和深度学习的第一批应用之一，它是研究自编码器早期驱动力之一。低维表示可以提高许多任务的性能，例如分类。小空间的模型消耗更少的内存和运行时间。

相比普通任务， 信息检索从降维中获益更多，此任务需要找到数据库中类似查询的条目。此任务不仅和其他任务一样从降维中获得一般益处，还使某些低维空间中的搜索变得极为高效。特别的，如果我们训练降维算法生成一个低维且二值的编码，那么我们就可以将所有数据库条目在哈希表映射为二值编码向量。这个哈希表允许我们返回具有相同二值编码的数据库条目作为查询结果进行信息检索。我们也可以非常高效地搜索稍有不同条目，只需反转查询编码的各个位。
