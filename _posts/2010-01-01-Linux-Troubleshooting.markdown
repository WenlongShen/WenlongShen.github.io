---
layout:     post
title:      "Linux Troubleshooting"
subtitle:   "Linux常见问题解决办法整理"
date:       2010-01-01
author:     "Wenlong Shen"
header-img: "img/bg/2017_3.jpg"
tags: 计算机
---

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=default"></script>

*OS：Ubuntu，不定期更新...*

1. 
**Q**：编译时出现cannot find lz错误  
**A**：apt-get install zlib1g-dev  

2. 
**Q**：Ubuntu16.04更改全局环境变量  
**A**：/etc/environment（CentOS下source /etc/environment或者/etc/profile莫名无效，可尝试/etc/bashrc）  

3. 
**Q**：CPAN中安装GD出现错误：No success on command[/usr/bin/perl Build.PL --installdirs site]  
**A**：手动下载编译GD包，或者直接安装Ubuntu的libgd-perl包  

4. 
**Q**：配置XX-Net  
**A**：从<a href="https://github.com/XX-net/XX-Net/" target="_blank">XX-Net官方github</a>下载最新版本，sudo运行./start；从<a href="https://www.switchyomega.com/" target="_blank">SwitchyOmega官网</a>下载并安装最新插件，然后使用XX-Net/SwitchyOmega/OmegaOptions.bak；命令行sudo apt-get install miredo；CA证书从<a href="http://127.0.0.1:8085" target="_blank">XX-Net配置界面</a>下载安装即可。注意XX-Net和miredo不会自启动，可参考下条  

5. 
**Q**：Ubuntu18.04配置开机自启动脚本  
**A**：这里只给出实用方法之一。Ubuntu18.04使用systemd管理系统，默认读取/etc/systemd/system下的配置文件，该目录下的文件会链接/lib/systemd/system/下的文件，更改该目录下rc.local.service文件，在末尾增加下面几行  
	`[Install]`  
	`WantedBy=multi-user.target`  
	`Alias=rc-local.service`  
然后手动创建/etc/rc.local文件  
	`#!/bin/sh -e`  
	`...`  
	`exit 0`  
把开机要执行的命令放到exit 0前面即可。最后执行  
`ln -s /lib/systemd/system/rc.local.service /etc/systemd/system/`  
在/etc/systemd/system目录下创建软链接即可  

6. 
**Q**：root下使用gedit出现Gtk-WARNING: cannot open display  
**A**：这是由于Xserver在默认情况下不允许其他用户的图形显示在当前屏幕上，因而需要在当前登陆用户身份下，命令行执行`xhost +`  

7. 
**Q**：无法启动，文件系统需修复，出现The root filesystem on /dev/sda1 requires a manual fsck  
**A**：`(initramfs) fsck /dev/sda1`随后一路回车，最后重启即可  

8. 
**Q**：Linux远程连接  
**A**：首推TeamViewer。另一种选择是VNC，建议在<a href="https://www.realvnc.com/" target="_blank">realvnc</a>注册并下载相应系统的server和viewer安装包。在Linux下安装server后，使用`vnclicensewiz`命令设置好账号和密码，然后`service vncserver-x11-serviced start`开启服务，建议将这一命令加入开机启动，参见上面第5条  

9. 
**Q**：pdf文件压缩  
**A**：第一种方法使用ghostscript命令，参考命令如`gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen -dNOPAUSE -dQUIET -dBATCH -dColorImageResolution=150 -sOutputFile=output.pdf input.pdf`；另一种方法利用convert命令将文件以图片形式压缩，参考命令如`convert -density 200x200 -quality 60 -compress jpeg input.pdf output.pdf`