---
layout:     post
title:      "搭建基于Docker的Tensorflow+Cuda环境"
subtitle:   "Walks"
date:       2019-01-16
author:     "Wenlong Shen"
header-img: "img/bg/2019_1.jpg"
tags: 机器学习 数据分析 2019
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

机器学习已是大数据分析的必备手段，我们尝试在Linux（Ubuntu 18.04）下搭建一个基于Docker的Tensorflow+Cuda环境，以用于学习、试验等。

关于Docker的安装和使用请参考<a href="https://wenlongshen.github.io/2018/09/08/Pipelining-Solution-1/" target="_blank">我以前的博客内容</a>。我们这里使用的是Tensorflow 1.12，使用前还需查看其对GPU的<a href="https://tensorflow.google.cn/install/gpu/" target="_blank">相关要求</a>。

#### Nvidia驱动安装配置

可以`lspci | grep -i nvidia`查看自己机子的N卡，也可以`ubuntu-drivers devices`查看系统给出的配置及驱动推荐，这时，我们可以直接`ubuntu-drivers autoinstall`进行安装，完成并重启后，可通过`nvidia-smi`验证安装是否成功并查看N卡的状态。

#### 关于Cuda的Docker

可以参考<a href="https://github.com/NVIDIA/nvidia-docker/" target="_blank">官方github文档</a>以及相应的FAQ等。

	# apt自己升个级
	$ sudo apt-get update
	
	# 添加nvidia-docker的官方GPG key及distribution list
	$ curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
	$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
	$ curl -sL https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

	# 接下来apt再自己升个级
	$ sudo apt-get update
	
	# 安装nvidia-docker
	$ sudo apt-get install nvidia-docker2

测试一下吧。

	# cuda镜像可参见https://hub.docker.com/r/nvidia/cuda
	$ docker pull nvidia/cuda:9.0-cudnn7-devel
	
	# 测试
	$ docker run --runtime=nvidia --rm nvidia/cuda:9.0-cudnn7-devel nvidia-smi

#### 关于Tensorflow的Docker

可以参考<a href="https://tensorflow.google.cn/install/docker/" target="_blank">官方文档</a>以及相应的FAQ等。

	# tensorflow镜像可参见https://hub.docker.com/r/tensorflow/tensorflow/
	# 也可以去https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dockerfiles/自行build
	$ docker pull tensorflow/tensorflow:1.12.0-devel-gpu-py3
	
	# 测试
	$ docker run --runtime=nvidia -it --rm tensorflow/tensorflow:1.12.0-devel-gpu-py3 python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"

	# 喜欢jupyter的可以
	$ docker run --runtime=nvidia -d -v yourdir:/notebooks -p 8888:8888 tensorflow/tensorflow:1.12.0-devel-gpu-py3 jupyter notebook --notebook-dir=/notebooks --ip 0.0.0.0 --no-browser --allow-root

以上便设置好了一个基本的机器学习环境，后续我们也可以根据自己的需求DIY相应的Docker，再接再励吧。
