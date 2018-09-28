---
layout:     post
title:      "生物信息分析流程（1）"
subtitle:   "docker入门"
date:       2018-09-08
author:     "Wenlong Shen"
header-img: "img/bg/2018_5.jpg"
tags: 生物信息 数据分析 流程 2018
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

*Build, Manage and Secure Your Apps Anywhere. Your Way.*

流程化是工业进步的标志，生物学科尚处于基因组学大发现时代，面临着庞杂数据的处理，相应的分析流程必不可少（*我早应开发自己的流程工具集，错过第一波的最佳时机，sigh...*）。最近被人强行安利docker，作为一个开源的应用容器引擎，小巧、可移植、可虚拟化、沙箱式的使用机制使其成为流程应用开发的不二之选。

使用docker要理解image和container的概念，镜像包含了一个完整的操作系统及所需要的应用程序，容器则是从镜像创建的一个实例，同一个镜像可以对应多个容器，而容器彼此之间隔离。下面借用<a href="http://merrigrove.blogspot.com/2015/10/visualizing-docker-containers-and-images.html" target="_blank">Daniel Eklund</a>的图来说明：
![docker](/img/post/2018_09_08_pipelining_ImageContainer.png)
注意用层（layer）的概念来理解，简单来说，镜像构建过程可能是一层一层分而进行的，且容器可以看作是镜像加上可读写层。一个运行状态的容器就是一个可读写的文件系统加上隔离的进程空间，一方面保证彼此安全，一方面对文件的修改只作用于可读写层而不影响原镜像，这种技术使得Docker前途无量。

#### 配置安装

没啥好说的，直接<a href="https://docs.docker.com/install/linux/docker-ce/ubuntu/" target="_blank">官网文档</a>来一遍就好，我后面都以Ubuntu为例，基本流程就是首先在apt中配置好repository的相关信息，再apt-get直接安装。建议注册个<a href="https://hub.docker.com/" target="_blank">账号</a>，还可以跟自己的GitHub账号连接。

	# apt自己升个级
	$ sudo apt-get update
	
	# apt安装相关包
	$ sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
	
	# 添加Docker的官方GPG key
	$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
	
	# 验证密钥安装成功
	$ sudo apt-key fingerprint 0EBFCD88
	
	# 配置Docker官方repository
	$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

	# 接下来apt再自己升个级
	$ sudo apt-get update
	
	# 直接安装docker
	$ sudo apt-get install docker-ce

	# 创建docker用户组，并加入
	$ sudo groupadd docker
	$ sudo usermod -aG docker $USER

	# 注意：卸载命令
	$ sudo apt-get purge docker-ce
	$ sudo rm -rf /var/lib/docker

测试一下吧，`docker run hello-world`，会显示`Hello from Docker！... (blahblah)`，即表示安装配置成功了。

#### 基本规则

**docker create \<image-id\>**
为指定的某个镜像（image）添加一个可读写层，构成一个新的容器（container），注意此时该容器并未运行。
![docker](/img/post/2018_09_08_pipelining_create.png)

**docker start \<container-id\>**
为指定的某个容器创建进程空间，注意每一个容器只能够有一个进程空间，且容器之间彼此隔离。
![docker](/img/post/2018_09_08_pipelining_start.png)

**docker run \<image-id\>**
该命令实际为create和start的共同体，即先从指定镜像创建出容器，再运行该容器。

**docker ps**
列出所有运行中的容器，如果使用`-a`（如下），则列出所有容器。
![docker](/img/post/2018_09_08_pipelining_ps-a.png)

**docker images**
列出所有“顶层”镜像，即那些创建容器时使用的镜像，如果使用`-a`（如下），则列出镜像的所有只读层。
![docker](/img/post/2018_09_08_pipelining_images-a.png)

**docker stop \<container-id\>**
向运行中的容器发送SIGTERM信号，使其停止所有进程，如果使用`docker kill`则发送一个残忍的SIGKILL信号。
![docker](/img/post/2018_09_08_pipelining_stop.png)

**docker rm \<container-id\>**
移除一个容器。
![docker](/img/post/2018_09_08_pipelining_rm.png)

**docker rmi \<image-id\>**
移除一个镜像，如果使用`-f`则可以只删除中间的只读层。
![docker](/img/post/2018_09_08_pipelining_rmi.png)

**docker commit \<container-id\>**
该命令将一个容器的可读写层转换为只读层，这样即把容器变为了镜像。
![docker](/img/post/2018_09_08_pipelining_commit.png)

**docker build**
该命令会根据Dockerfile文件中的FROM指令获取相应镜像，然后重复执行：1）run（create和start）；2）修改、安装、配置程序等；3）commit。每步循环都会生成一个新的层，因而最终会有多个新层被创建。
![docker](/img/post/2018_09_08_pipelining_build.png)

**docker exec \<running-container-id\>**
在运行的容器中执行一个新进程
![docker](/img/post/2018_09_08_pipelining_exec.png)

**docker save \<image-id\>**
导出一个镜像的压缩文件，使其可以在其它主机的Docker上使用。和export命令不同，save为每一个层都保存了它们的元数据，且只能对镜像生效。相应地，可以用load命令导入镜像。
![docker](/img/post/2018_09_08_pipelining_save.png)

**docker export \<container-id\>**
导出一个容器的压缩文件，移除了元数据和不必要的层，将多个层整合成了一个层。expoxt后再利用import命令导入，只能看到一个镜像，而save后的能看到其历史镜像。
![docker](/img/post/2018_09_08_pipelining_export.png)

#### 应用实例

我们使用生信分析中常用的比对工具bowtie2作为应用实例。这里顺便一提<a href="https://biocontainers.pro/" target="_blank">BioContainers</a>，一个类似<a href="http://www.bioconductor.org/" target="_blank">Bioconductor</a>、<a href="http://bioconda.github.io/" target="_blank">Bioconda</a>的生物信息分析工具集合。

首先`docker pull biocontainers/bowtie2`，从BioContainers上抓取bowite2的镜像，成功后可通过`docker images`查看。

运行的相关命令可以`docker run --help`查看，BioContainers的细节也可以参照其官网文档，这里我们先准备好bowtie2比对时用到的数据，存放在本地`/home/wenlong/Data`下，索引文件hg19放入文件夹hg19index中，测序文件test.r1.gz和test.r2.gz。接下来执行：

	docker run --rm -v /home/wenlong/Data/:/data/ biocontainers/bowtie2 bowtie2 -x /data/hg19index/hg19 -1 /data/test.r1.gz -2 /data/test.r2.gz -S /data/test.sam

其中`--rm`使得该容器退出后自动删除，`-v`进行文件夹的挂载，bowtie2生成的比对文件将出现在本地`/home/wenlong/Data`下，注意biocontainers构建这个镜像时使用了`CMD ["bowtie2"]`，所以`-it`在这里没有意义。

#### Docker化

对于我们自己的程序工具，往往也希望做成Docker镜像，以供未来使用或分享给他人，这里以ChIP-seq等分析时常用的peak calling工具MACS2为例。

构建镜像的方式有好几种，我们主要利用Dockerfile，简要介绍一下相关指令（大小写不敏感，但惯例为均大写），具体的还应参考<a href="https://docs.docker.com/develop/develop-images/dockerfile_best-practices/" target="_blank">官方文档</a>：

* **FROM**：指定基础镜像，必须是第一条指令。
* **MAINTAINER**：指定镜像作者信息。
* **LABEL**：指定镜像信息。
* **ENV**：设置环境变量。
* **ADD & COPY**：将本地文件复制进镜像，注意COPY只能是本地文件，ADD可以是url，并且自动解压缩。
* **RUN**：运行指定的命令，每一个RUN指令都将在一个新的container里面运行，并提交为一个image作为下一个RUN的base，即层与层之间不会共用内存，所以如果要将多条命令联合起来执行则需加上&&。另外，在一个Dockerfile中可以包含多个RUN，按顺序执行。
* **CMD**：在容器启动时要运行的命令。

我们首先下载官方的Ubuntu镜像作为起始镜像`docker pull ubuntu`，然后建立一个文件夹用于存放制作镜像过程中所用到的文件，比如这里我建立文件夹`/home/wenlong/DockerMacs2`，下载MACS2的源码包并新建Dockerfile文件如下：

	# base image
	FROM ubuntu

	# maintainer
	MAINTAINER Wenlong Shen <shenwl1988@gmail.com>

	# metadata
	LABEL version="1.0" \
		software="macs2" \
		software.version="2.1.1" \ 
		about.summary="Model-based Analysis of ChIP-Seq on short reads sequencers" \
		about.home="http://github.com/taoliu/MACS/" \ 
		about.copyright="2013, Tao Liu lab at UB and Xiaole Shirley Liu lab at DFCI" \ 
		about.license="BSD-3-clause"

	# put source files into /usr/local/src and unpack
	ADD MACS2-2.1.1.20160309.tar.gz /usr/local/src

	# install prerequisites
	RUN apt-get update && \
		apt-get install -y --no-install-recommends \
		python python-dev python-setuptools python-numpy gcc && \
		apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/*

	# change dir to /usr/local/src/MACS2-2.1.1.20160309
	#WORKDIR /usr/local/src/MACS2-2.1.1.20160309

	# execute command to compile macs2
	RUN cd /usr/local/src/MACS2-2.1.1.20160309 && \
		python setup.py install && \
		cd .. && \
		rm -rf MACS2-2.1.1.20160309

接下来执行`docker build -t macs2:ubuntu.v1 .`，其中`:`前面为镜像名称，后面为TAG，`.`表示当前目录，也可以使用全路径，目的就是找到Dockerfile文件。由于需要网络环境，所以整体过程可能较慢。构建成功后，可docker images`查看。下面我们尝试使用该镜像：

	docker run --rm -v /home/wenlong/Data/:/data/ macs2:ubuntu.v1 macs2 callpeak -t /data/test.sam -f SAM -n test --outdir /data/macs2_result

我们还可以将该镜像上传到自己的docker hub，首先要把TAG规范化

	docker tag macs2:ubuntu.v1 wenlongshen/macs2:ubuntu.v1

然后登陆`docker login`，成功后推送`docker push wenlongshen/macs2:ubuntu.v1`，至此，就可以在自己的docker hub里查看并管理了，同样地，也可以`docker pull wenlongshen/macs2:ubuntu.v1`到本地使用。

