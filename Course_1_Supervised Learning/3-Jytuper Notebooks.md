# Jupyter Notebook

本课程使用的编程环境

首先创建个虚拟环境（用VS建的，python环境 > 概览 > 在PowerShell中打开

输入

```shell
pip install notebook
```

等待下载安装完成即可

输入

```shell
jupyter notebook
```

创建一个新的notebook，此时C:/Users/Username/AppData/Roaming/jupyter下面会出现实时web页，运行结束后这个会自动删除，不用担心。

![image-20230512152159220](.\3-Jytuper Notebooks.assets\image-20230512152159220.png)

按`Ctrl + C`退出当前Notebook



# pip依赖

打开对应环境，运行

```shell
pip install -r .\requirements.txt
```



## 报错`python setup.py egg_info did not run successfully`

![image-20230531171212357](.\3-Jytuper Notebooks.assets\image-20230531171212357.png)

有一说一这个报错名称真的没参考性。看到挺多解答都说是pip和setup.py的问题，要更新pip和setup.py

```
pip install --upgrade pip
pip install --upgrade setuptools
```

但问题点不是这里，这个问题应该是pycurl自己对pip的支持问题，[issue#657]([Install pycurl on windows · Issue #657 · pycurl/pycurl · GitHub](https://github.com/pycurl/pycurl/issues/657))里面提到这个问题，有人提出用第三方的预编译轮子

```py
pip install https://download.lfd.uci.edu/pythonlibs/archived/pycurl-7.45.1-cp39-cp39-win_amd64.whl
```

![image-20230601105439095](.\3-Jytuper Notebooks.assets\image-20230601105439095.png)

这样姑且是装上了。

顺带微软还没修复安装python包的时候疯狂报`Error: missing params.textDocument.text`的问题，虽然工程师嘴硬说修了。

![image-20230601105908542](.\3-Jytuper Notebooks.assets\image-20230601105908542.png)

# 在对应文件夹运行

输入jupyter notebook的路径就是创建notebook的根路径，如果想打开本地文件，需要先移动到对应路径。

![image-20230512152614147](.\3-Jytuper Notebooks.assets\image-20230512152614147.png)

找到第一个作业文件，双击

![image-20230512152626349](.\3-Jytuper Notebooks.assets\image-20230512152626349.png)

成功打开

![image-20230512152650414](.\3-Jytuper Notebooks.assets\image-20230512152650414.png)



# 编辑Jupyter notebook

单元格分为两类：

- 文本单元格。markdown格式，单击选中，双击或在选中后按Enter进入编辑模式，Shift + Enter确认编辑

- 代码单元格。Shift + Enter运行。也可以点击菜单栏中的”运行“

  ![image-20230512155458951](.\3-Jytuper Notebooks.assets\image-20230512155458951.png)

当进入编辑模式的时候，右上角会有铅笔图标

![image-20230512153757879](.\3-Jytuper Notebooks.assets\image-20230512153757879.png)

退出编辑模式也可以使用Esc；铅笔图标的右边就是内核指示器，指示当前是否在计算





# 关闭notebook

运行结束，确定已经保存后，在Shell里面按Ctrl + C关闭notebook。