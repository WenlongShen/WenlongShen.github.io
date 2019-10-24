---
layout:     post
title:      "Windows Troubleshooting"
subtitle:   "Windows常见问题解决办法整理"
date:       2010-01-01
author:     "Wenlong Shen"
header-img: "img/bg/2017_3.jpg"
tags: 计算机
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

*OS：Windows7（64bit），不定期更新...*

**1. 软件安装**  
**Q**：下载Chrome离线安装包  
**A**：http://www.google.cn/chrome/browser/desktop/index.html?standalone=1&platform=win64  
**Q**：Adobe Flash Player全版本通用安装  
**A**：https://get.adobe.com/cn/flashplayer/otherversions/  

**2. Sublime Text 3设置**  
**Q**：如何让Terminal插件启动CMD  
**A**：Terminal默认启动PowerShell，需要在Preferences->Package Settings->Terminal->Settings中更改设置`"terminal": "C:\\Windows\\System32\\cmd.exe"`  
**Q**：关闭自动打开上次文件  
**A**：Preferences->Settings中添加"hot_exit": false  

**3. Windows 10免费升级**  
**Q**：错过官方的免费升级  
**A**：看这里https://www.microsoft.com/zh-cn/accessibility/windows10upgrade  

**4. 虚拟机**  
**Q**：指定的文件不是虚拟磁盘，打不开磁盘或它所依赖的某个快照磁盘，模块”Disk“启动失败  
**A**：删除所有以.lck结尾的文件以及文件夹。虚拟磁盘（.vmdk）本身有一个磁盘保护机制：为了防止多台虚拟机同时访问同一个虚拟磁盘(.vmdk)带来的数据丢失和性能削减方面的隐患，每次启动虚拟机的时候虚拟机会使用扩展名为.lck（磁盘锁）文件对虚拟磁盘（vmdk）进行锁定保护，当虚拟机关闭时.lck（磁盘锁）文件自动删除。若虚拟机关闭时未能正确删除系统上的.lck（磁盘锁）文件，则下次启动时可能出现上述错误。

**5. Rstudio设置**  
**Q**：更改package默认安装路径  
**A**：更改R安装路径下etc文件夹中的Rprofile.site文件。可使用`myPaths <- .libPaths()`获取当前系统路径信息，通过更改myPaths，如`myPaths <- c(myPaths[2], myPaths[1])`来更改默认安装路径顺序，第一个为默认值，最后`.libPaths(myPaths)`完成修改。

**6. 其它问题**  
**Q**：系统卡在”配置Windows Update失败，还原更改“  
**A**：利用PE或者系统盘，删除文件Windows\winsxs\pending.xml，重启后短时等待即可。

