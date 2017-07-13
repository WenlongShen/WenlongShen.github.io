---
layout:     post
title:      "Linux Troubleshooting"
subtitle:   "Linux常见问题解决办法整理"
date:       2010-01-01
author:     "Wenlong Shen"
header-img: "img/bg/2017_3.jpg"
tags: 计算机
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

*OS：Ubuntu16.04（xenial，64bit），不定期更新...*

1. 
**Q**：编译时出现cannot find lz错误  
**A**：apt-get install zlib1g-dev  

2. 
**Q**：更改全局环境变量  
**A**：/etc/environment（CentOS下source /etc/environment或者/etc/profile莫名无效，可尝试/etc/bashrc）  

3. 
**Q**：CPAN中安装GD出现错误：No success on command[/usr/bin/perl Build.PL --installdirs site]  
**A**：手动下载编译GD包，或者直接安装Ubuntu的libgd-perl包  
