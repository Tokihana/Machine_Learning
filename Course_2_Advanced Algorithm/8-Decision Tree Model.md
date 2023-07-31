# Cat classification example

这一章节使用的样例为猫咪分类样例，如下图所示，给出猫咪的三个特征，要求判断是否为猫咪（二分类任务）

![image-20230731172759104](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230731172759104.png)



# Decision Tree

决策树算法的模型是一棵树。对每个输入样例，算法会从根节点开始，逐步前进到对应target的叶子。

![image-20230731173013662](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230731173013662.png)



对一个任务有非常多种可能的决策树，例如

![image-20230731173347880](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230731173347880.png)



## Learning Process

决策树模型的主要流程遵循**分而治之**：
$$
\underline{
\overline{
\begin{array}{l}
\bf {def}\ BuildTree:\\
1. 创建node \\
2. \bold {if}\ 当前node对应的样例都是同一类别，建立叶子并\bf {return} \\
3. \bold {if}\ 当前集合为空 \empty，或者不存在可以划分的属性，建立叶子并\bf {return}\\
\\
4.不满足上两个返回条件，则按*最优划分属性*进行划分:\\
\bf {for}\ 每个属性取值:\\
\quad 生成分支，递入下一层
\end{array}
}
}
$$
![image-20230731175147611](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230731175147611.png)