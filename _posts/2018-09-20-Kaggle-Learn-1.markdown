---
layout:     post
title:      "Kaggle Learn 学习笔记（1）"
subtitle:   "Python"
date:       2018-09-20
author:     "Wenlong Shen"
header-img: "img/bg/2018_6.jpg"
tags: 机器学习 读书笔记 Kaggle 2018
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

#### Hello, Python!

1. 变量赋值前不需要提前声明，也不需要指定类型。  
2. 用`#`表示注释信息。  
3. 通过4个空格或一个tab表示代码块层级，但不要混用。  
4. 运算顺序PEMDAS - Parentheses, Exponents, Multiplication/Division, Addition/Subtraction。

#### Functions and Getting Help

1. 用`def function(arguments):`定义一个函数。  
2. 使用`help()`得到函数的帮助信息，在自定义的函数内部可利用三重引号描述帮助信息`"""..."""`。  
3. lambda表达式可以方便地定义一个匿名函数`lambda parameter_list: expression`。  

#### Booleans and Conditionals

1. bool变量具有`True`和`False`两个值，常见操作有`and`，`or`和`not`，0和空串是False，其余数字和字符串为True。  
2. 判断语句常使用`if`，`elif`和`else`。另外可以使用单行的条件表达式进行简写`x if C else y`，即当C为真时，返回x的值，否则返回y的值。  

#### Lists and Tuples

1. 列表是用方括号包围、逗号分开的一组元素，可用索引访问、修改。  
2. 常见的函数如`len`求列表长度，`sum`数值求和，`sorted`排序等。  
3. 列表内置的函数还有`list.append()`在列表末端增加一个元素，`list.pop()`移除并返回列表最末端的元素，`list.index()`返回该元素的索引等。  
4. 元组与列表几乎相同，唯二不同，一是元组用圆括号定义，二是元组里的元素不能改变。  

#### Loops and List Comprehensions

1. 对于for循环，我们可以遍历元素`for foo in x`，可以遍历索引`for i in range(len(x))`，也可以两者同时`for i, num in enumerate(nums)`。  
2. while循环，一直执行到布尔表达式为False。  
3. 列表解析可以一行代码搞定赋值，如`variable = [statement for foo in x if boolean_statement]`。  

#### Strings and Dictionaries

1. 字符串赋值可以使用单引号或双引号，三重引号可以默认换行而不需要`\n`。  
2. 可通过索引访问字符串中的字符。  
3. 常用的内置函数有`string.index()`查找子串，`string.split()`按照某个子串分割，`string.join()`合并多个子串，`string.format()`整理字符串内容。  
4. 字典使用大括号包围，元素为`key:value`，访问时使用key代替列表中的索引。  
5. `dict.keys()`返回所有的键，`dict.values()`返回所有的值，`dict.items()`返回所有的键值对。  

#### Working with External Libraries

1. 通过`import`引入别的模块，`import ... as ...`将该模块重命名，用该类方法访问函数时需加上模块的名字。  
2. 通过`from ... import ...`引入别的模块中的函数，用该类方法访问函数时不需要加上模块的名字。  
3. 善于运用`type()`，`dir()`和`help()`将有助于理解别人写的模块。  
