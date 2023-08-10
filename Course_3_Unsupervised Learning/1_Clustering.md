# Unsupervised learning

在无监督学习中，数据只有features，没有label，例如下图

![image-20230808170308746](D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230808170308746.png)

无监督学习的目标是通过学习找到数据的内在规律。最常见的任务类型是聚类（Clustering）。

# Clustering

聚类的目标是将样本划分为若干个不相交的子集。既可以作为单独过程，也可以作为其他学习任务的前驱过程。 



# K-means Clustering

## Intuition

k均值是一种**原型聚类（prototype-based clustering）**方法，算法假设聚类能够通过原型进行刻画，原型指样本空间中**具有代表性的点**。k均值视图找到一组**原型向量（prototype vector）**，通过最小化样本围绕这些向量的紧密程度，使得这些向量最具代表性。

1. 随机选择一组点作为最开始的**质心（centroid）向量**，centroid和feature的维数相同

   <img src="D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230809151719487.png" alt="image-20230809151719487" style="zoom:50%;" />

2. 遍历所有的样例，根据距离匹配质心

   <img src="D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230809151850077.png" alt="image-20230809151850077" style="zoom:50%;" />

3. 计算每组样例的平均值，将质心移动到该点。

   <img src="D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230809152006035.png" alt="image-20230809152006035" style="zoom:50%;" />

4. 重复"遍历-移动到均值"的过程。直到收敛（质心位置变化变得非常小）

   <img src="D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230809152201738.png" alt="image-20230809152201738" style="zoom:50%;" />

   <img src="D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230809152244432.png" alt="image-20230809152244432" style="zoom:50%;" />



## Algorithm

算法的伪码表示如下
$$
\over{
\underline{
\begin{array}{l}
输入:样本D;\\
\quad\quad 类别数k;\\
过程:\\
1.初始化k个质心{\mu_1, \cdots, \mu_k}, 一般是选k个样例\\
2.\bf{repeat}\\
3.\quad \mathbf{for} \ x_i \ \mathbf{in}\ D:\\
4.\quad\quad min_k||x_i - \mu_k||^2, 划分到距离最近的类别\\
5.\quad \mathbf{for}\ \mu_i\ \mathbf{in}\ \{\mu_1, \cdots, \mu_k\}:\\
6.\quad\quad 计算新的均值\mu_i^{’}\\
7.\quad\quad \mathbf{if}\ \mu_i^{'} \neq \mu_i:\\
8.\quad\quad\quad 更新\mu_i\\
9.\mathbf{until}\ 没有任何质心向量被更新\\
输出: 划分的聚类
\end{array}
}
}
$$

> 在划分类别的时候，可能出现某些类别样例数为0的情况，此时可以选择删除该类别，或者重新随机化初始化这个类别。
>
> 算法的过程使用了贪心思想：每轮迭代都选择当前聚类的最优解（均值）



## Optimization

k均值所做的优化相当于最小化平方误差
$$
E = \sum^k_{i = 1}\sum_{x\in C_i}||x - \mu_i||_2^2\\
\mu_i = \frac 1 {|C_i|}\sum_{x\in C_i}x
$$
E表现出样例围绕当前质心组的紧密程度，E值越小则类别内的相似度越高。$||\ ||^2_2$这个写法代表L2范数的平方。

若设$\mu_{C_i}$为第$i$个样例对应的质心，上式也可以表示为
$$
E = \frac 1 m \sum_{i = 1}^m ||x_i - \mu_{C_i}||^2_2
$$

> 上式也有人称distortion function



从直观上理解k均值所做的最小化：在每轮迭代的第一个循环中，算法通过为每个样例匹配最接近的$\mu_i$来最小化损失函数，此时相当于最优化样例匹配的精确度；而在第二个循环中，算法通过取$C_i$中所有样例的均值，使$\mu_i$移动到当前类别的质心，此时相当于最优化聚类的离散度。

> 一些证明：
>
> 1. $\mu = \frac 1 m \sum_{i = 1}^m x_i$时，$E=\sum_{i = 1}^{m}||x_i - \mu||_2^2$最小
>
>    对$\mu$求导，
>    $$
>    \frac {\partial \sum_{i = 1}^{m}||x_i - \mu||_2^2} {\partial \mu} = \sum_{i=1}^n -2(x_i - \mu)\\
>    = -2\sum_{i = 1}^n x_i + 2 n\mu
>    $$
>    可知$\mu = \frac 1 m \sum_{i = 1}^m x_i$为函数驻点。因为E是严格凸的（$f(\frac {x_1 + x_2)} {2}) \leq \frac {f(x_1) + f(x_2)} 2$），因此$\mu = \frac 1 m \sum_{i = 1}^m x_i$为当前类别cost的唯一极小值点。
>
>    
>
> 2. 算法收敛性
>
>    根据函数式可知，$E\geq0$，且在迭代过程中，数列${E_n}$单调递减；根据单调有界准则，单调递减有下界，该数列一定收敛，$\lim_{n\to \infin}E$存在。
>
>    因此，k-means是一定会收敛的，不过对不同的初始值，收敛的结果可能不一样，所以初始化也是k-means的超参数。



## Initializing

k-means中k一定严格小于样例数m，因为当$k > m$的时候，样例数都不够均分给每个类别。

初始质心的选择会影响最后收敛的结果，如图所示，第一种收敛效果明显更好一些

<img src="D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230809171811302.png" alt="image-20230809171811302" style="zoom:80%;" />

可以通过计算E来判断哪个收敛结果更好
$$
\over{
\underline{
\begin{array}{l}
\mathbf{for}\ i\ =\ 1\ \mathbf{to}\ randTimes: \\
\quad 随机初始化\mu\\
\quad 运行\ k-means, 计算\ E\\
\quad 选择E最小的
\end{array}
}}
$$

> randTimes通常为50~1000次，太多次数会比较费时间；太少可能找不到比较好的。



