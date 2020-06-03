---
layout:     post
title:      "全基因组测序简要分析流程"
subtitle:   "A brief pipeline for wgs data analysis"
date:       2020-05-28
author:     "Wenlong Shen"
header-img: "img/bg/2020_5.jpg"
tags: 基因组 测序 生物信息 2020
---

*中国有人焉，非阴非阳，处于天地之间，直且为人，将反于宗。*

自人文哲学到生物分子，人类对于生命的追问，对于自身奥秘的探索，从未停止过。基因组，蕴含生命线索的密码书，漫漫其修远兮，吾将上下而求索。

我们知道人与人之间在基因组层面存在诸多差异，SNP、Indel、CNV、SV等，这些差异可能跟性格、疾病、寿命等有密切联系，借助WGS/WES（Whole Genome Sequencing，全基因组测序；Whole Exome Sequencing，全外显子测序），我们可以对自身有更好的了解。

这里我们走一遍测序后的基本分析和处理流程，让大家能有更直观的认识。

#### 软件环境

我们在Ubuntu18.04下构建docker镜像，主要用到的软件包有samtools、Picard、fastp、bwa、GATK、Strelka2等。人类基因组版本的选择可参考<a href="https://wenlongshen.github.io/2020/03/26/Reference-Genome/" target="_blank">我的这篇博文</a>，这里我们选取不含ALT的GRCh38作为参考基因组。

主要脚本代码参见<a href="https://github.com/WenlongShen/Shadow" target="_blank">我的github</a>。

#### 数据质控

拿到测序数据之后，我们首先要进行质控，以考察本次测序的质量是否合格，是否值得进行后续分析等。另外，如果是原始下机数据，我们还需要去除文库接头、去除低质量reads等。常见的工具有FastQC、trimmomatic、cutadapt、fastp等。这里我们推荐使用fastp，其使用简单、运速较快，默认可以不设置任何参数，就能够同时进行序列过滤、校正并产生质控报告。另外，比对完的数据还可利用Picard中的CollectWgsMetrics或CollectRawWgsMetrics来评价序列质量和覆盖度等。

```shell
fastp --in1 sample.R1.fq.gz --in2 sample.R2.fq.gz \
      --out1 sample.R1.clean.fq.gz --out2 sample.R2.clean.fq.gz
```

#### 唯一比对和冗余序列

比对和去冗余是高通量数据分析的常见步骤，关于这两步的相关内容，可以参考<a href="https://wenlongshen.github.io/2020/04/08/Unique-Duplicate/" target="_blank">我的这篇博文</a>。

在这里，我们使用BWA进行序列比对。首先我们需要建立参考基因组的索引`bwa index ...`，之后进行比对，常用命令形式如下，这里要注意记得加上`-M`才能兼容后续Picard进行冗余标记，`-R`后面设置数据的相关信息，为以后查询提供方便。

```shell
ID=sample_RG # Read Group, or Lane ID
PL=ILLUMINA # ILLUMINA, SLX, SOLEXA, SOLID, 454, LS454, COMPLETE, PACBIO, IONTORRENT, CAPILLARY, HELICOS, UNKNOWN
LB=sample_lib # Library
SM=sample # Sample
bwa mem -M \
	-R "@RG\tID:$ID\tPL:$PL\tLB:$LB\tSM:$SM" \
	${reference_bwa} \
	sample.R1.clean.fq.gz sample.R2.clean.fq.gz \
| samtools view -bS - \
| samtools sort -o sample.bam
```

另外，对于去冗余这一步，GATK等算法目前只是标注出冗余序列，并不直接去除，但在后续分析中不会使用冗余序列。要注意的是，如果算法本身不处理冗余序列，则需我们利用Picard等工具主动去除。

```shell
gatk MarkDuplicatesSpark \
	-I sample.bam \
	-O sample.markdup.bam \
	-M sample.markdup.metrics \
	--remove-all-duplicates false
```

#### 碱基质量校正

显而易见的，变异检测极其依赖于测序碱基的质量。这个质量值的高低直接决定了判断该碱基为真实变异的准确性。然而所有的测序系统或多或少地都存在偏差，再加上外部环境和实验过程带来的误差，系统偏性几乎是不可避免的。GATK提供了BQSR（Base Quality Score Recalibration）这一步以用于矫正测序碱基的“真实质量值”。BQSR会设置参考SNP数据集作为“真实”变异集，这些数据集之外的位点，若测序数据与参考基因组存在差异，则认为是“错误”，以此来判定并矫正测序碱基的质量值。

```shell
gatk BaseRecalibrator \
	-R ${reference_fa} \
	--known-sites dbsnp_146.hg38.vcf.gz \
	--known-sites 1000G_phase1.snps.high_confidence.hg38.vcf.gz \
	--known-sites Mills_and_1000G_gold_standard.indels.hg38.vcf.gz \
	-I sample.markdup.bam \
	-O sample.bqsr.recal.table
gatk ApplyBQSR \
	-R ${reference_fa} \
	-I sample.markdup.bam \
	--bqsr-recal-file sample.bqsr.recal.table \
	-O sample.bqsr.bam
```

#### 变异检测

接下来即是WGS/WES分析最关键的一步即变异检测，这里我们以SNP和Indel为例，使用GATK提供的HaplotypeCaller工具。这个算法适用于二倍体基因组的变异检测，它会识别出潜在变异区域，然后利用De Bruijn图重新组装并重新比对该区域，以识别出变异位点；同时，该算法会推断群体样本的单倍体型组合情况，然后反推单个样本的基因型组合，因而既适合于群体的变异检测，也能够依据群体信息更好地识别单个样本的变异位点。

```shell
gatk HaplotypeCaller \
	-R ${reference_fa} \
	-I sample.bqsr.bam \
	-O sample.hc.vcf.gz
```

对于多个样本，我们通常加上`-ERC GVCF`参数，先生产gVCF的中间文件，再利用CombineGVCFs和GenotypeGVCFs将各个样本数据整合，这样对于多样本、新增样本、重测样本的情况较为省时省力。

```shell
for sample in ${samples};
do gatk HaplotypeCaller \
	-R ${reference_fa} \
	-ERC GVCF \
	-I ${sample}.bqsr.bam \
	-O ${sample}.hc.g.vcf.gz
done;

sample_gvcfs=""
for sample in ${samples}; do sample_gvcfs=${sample_gvcfs}"-V ${sample}.hc.g.vcf.gz " done;

gatk CombineGVCFs \
	-R ${reference_fa} \
	${sample_gvcfs} \
	-O multi_samples.hc.g.vcf.gz 
gatk GenotypeGVCFs \
	-R ${reference_fa} \
	-V multi_samples.hc.g.vcf.gz \
	-O multi_samples.hc.vcf.gz 
```

#### 变异位点质控和过滤

在获得了原始的变异位点后，我们需要对其进一步地质控和过滤，以筛选出准确性、可靠性更高的位点。要注意的是，通常SNP和Indel部分参数的使用是不同的。另外，对于resource部分，

* known：表明该数据集可作为一个已知集，不过仅用于数据标注
* training：表明该数据集可作为VQSR机器学习模型的训练集
* truth：表明该数据集可作为验证模型预测结果的真集

```shell
gatk VariantRecalibrator \
	-R ${reference_fa} \
	--resource:hapmap,known=false,training=true,truth=true,prior=15.0 ${hapmap} \
	--resource:omni,known=false,training=true,truth=true,prior=12 ${omni} \
	--resource:1000G,known=false,training=true,truth=false,prior=10.0 ${G1000} \
	--resource:dbsnp,known=true,training=false,truth=false,prior=2.0 ${dbsnp}} \
	-an DP -an FS -an MQ -an QD -an SOR -an MQRankSum -an ReadPosRankSum \
	-mode SNP \
	-tranche 100.0 -tranche 99.9 -tranche 99.0 -tranche 90.0 \
	--max-gaussians 4 \
	-V sample.hc.vcf.gz \
	-O sample.hc.snps.recal.table \
	--tranches-file sample.hc.snps.recal.tranches \
	--rscript-file sample.hc.snps.recal.plots.R
gatk ApplyVQSR \
	-R ${reference_fa} \
	-V sample.hc.vcf.gz \
	--recal-file sample.hc.snps.recal.table \
	--tranches-file sample.hc.snps.recal.tranches \
	-ts-filter-level 99.0 \
	-mode SNP \
	-O sample.hc.snps.vqsr.vcf.gz \
	--create-output-variant-index true
```

