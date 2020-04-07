---
layout:     post
title:      "关于人类基因组的一些说明"
subtitle:   "Reference genome"
date:       2020-03-26
author:     "Wenlong Shen"
header-img: "img/bg/2020_3.jpg"
tags: 基因组 测序 生物信息 2020
---

*参考参考*

#### GRC

人类基因组计划之初，曾试图勾勒出一套完整的、一致性的基因组序列图谱，但无论是测序技术、组装算法还是基因组本身的多样性问题，都让这套“纯粹的”参考基因组无法实现。目前的人类参考基因组由Wellcome Sanger Institute、EBI、NCBI等多家研究机构成员组成的<a href="https://www.ncbi.nlm.nih.gov/grc/" target="_blank">Genome Reference Consortium</a>（GRC）负责更新和维护。

目前发布的人类参考基因组，主要包含以下序列：

* Assembled chromosomes：22+XY+M，即23对染色体和线粒体基因组，作为日常研究分析的主要序列
* Unlocalized sequences：已被定为到某条染色体上，但方向或具体位置仍未确定，以_random结尾
* Unplaced sequences：尚未被定位到某条染色体，以chrUn_开头
* Alternate loci：不同的单倍体型，一般以_alt结尾，也包括HLA序列
* EBV & decoy sequences：不属于人类基因组，但是高通量测序时会被测到的序列，标注为chrEBV及以_decoy结尾的序列

GRC会不定期以**major**或**minor**的形式对参考基因组进行更新。**major**表示基因组原序列碱基或坐标有相对较大的改动，会提供新的编号，如GRCh37、GRCh38等；**minor**则是patch的形式，如GRCh38.p13等，其基因组原序列无变动，主要是新增了fix patch和noval patch两种，其中fix patch是对原基因组序列的更新和修正，其内容会在下一次major中并入，noval patch则是新的单倍体型。

#### 版本选用

我们在下载使用参考基因组的时候，会发现即使是同样的GRCh38版本，NCBI、UCSC等却提供了各种各样的子版本形式。关于基因组版本的选择，<a href="https://lh3.github.io/2017/11/13/which-human-reference-genome-to-use/" target="_blank">Heng Li</a>有比较好的说明。同时，NCBI提供了analysis set版本用于分析，总结起来主要需注意这几点：

* Unlocalized and unplaced sequences：建议选用。这部分序列虽不知道具体位置，但依然为人类基因组上已确定包含的片段，不选用会导致错误比对结果
* ALT：除非特定关注ALT上的相关基因，否则不建议选用。这部分序列在aligner中往往对应为多比对序列，mapq值或很低或为零，会对后续分析带来干扰。如果需要使用，可参考GATK提供的含ALT基因组的相关<a href="https://gatk.broadinstitute.org/hc/en-us/articles/360037498992/" target="_blank">流程</a>，另外，bwa有相应的<a href="https://github.com/lh3/bwa/tree/master/bwakit/" target="_blank">bwakit</a>用于处理含ALT的基因组序列比对问题。
* PAR：拟常染色体区域，位于X和Y染色体，不会被单独标注，但是在不同的版本中序列呈现方式不同，如在analysis set中，Y染色体的PAR会被hard-masked为N
* EBV & decoy sequences：这些序列只会轻微影响比对结果，但亦可在质控时提供部分帮助
* 另外，analysis set还在rCRS、semi-ambiguous IUB codes等处略有不同

我们可以选择从<a href="ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/" target="_blank">NCBI</a>或者GATK的<a href="ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/hg38/" target="_blank">bundle</a>中下载人类基因组参考序列，并可通过fai文件来判断不同版本所包含的序列信息，以帮助自己选择。
