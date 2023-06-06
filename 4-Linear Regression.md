# 线性回归（Linear Regression）

为数据拟合（fitting）一条直线。线性回归的很多思想在其他model中也有体现。



还是房价的那个例子，给出一系列房屋大小-房价的数据，绘图（plot）

![image-20230512160028918](C:\CS\Machine Learning\4-Linear Regression.assets\image-20230512160028918.png)



也可以绘制Data table

![image-20230512160301990](C:\CS\Machine Learning\4-Linear Regression.assets\image-20230512160301990.png)



# Terminology 术语

- Training set，训练集，用来训练的数据。
- feature，特征，也叫input，或者映射关系中的x
- target，目标，也叫output，映射关系中的y
- $(x, y)$代表训练集中的一个样例，$(x^{(i)}, y^{(i)})$代表第i个样例。





# Linear Regression with One Variable

![image-20230526193948648](C:\CS\Machine Learning\4-Linear Regression.assets\image-20230526193948648.png)

如图，在监督式学习中，学习算法使用训练集学习出一个Model（或者说是一种函数），这个model接受x输入，并预测y的值，记$\hat y$；



定义直线二维坐标下直线的公式
$$
f_{w, b}(x) = wx + b
$$
称有一个输入变量的线性回归公式（也叫Univariate linear egression）。



# Cost Function

用于衡量模型的好坏。构建成本函数可以自动化优化模型。

给定一组数据和一个模型，使用模型预测出y的预测值$\hat y$，将估计值与y的真值（true value）进行比对。从直观上，单组预测值与真值之间的关系可以描述为二者在空间上的距离关系，即$\sqrt{(y - \hat y)^2}$，也可以用$(y - \hat y)^2$来表示这种关系，称平方误差（square error）。将整组数组的误差求和，就可以得到衡量模型整体好坏的总平方误差（total squarre error）。

总平方误差会随着数据量的变化而变化，例如，假定单组数据的平方误差都为1，对于10组数据，总平方误差为10，对于100组数据，总平方误差为100；为了能够比较不同数据量训练出来的模型之间的好坏，还应该对总平方误差取均值，获得平均平方误差（average square error）。习惯上，我们还会多除一个2以简化计算（因为后面要求导，求导会出来一个2）
$$
Average\ Square\ Errror = \frac 1 {2m}\sum_{i = 1}^{m}(\hat y - y)^2
$$


# Cost Functionb Intuition

在这里建立下对成本函数的直观理解。

在回归任务中，我们的目标是让成本函数最小化，如下图

![image-20230601112942351](D:\CS\Machine Learning\4-Linear Regression.assets\image-20230601112942351.png)



考虑一个更为简化的模型，假定b恒=0，只考虑w的变化，则回归函数$f_w(x)$与损失函数$J(w)$分别为以x和以w为参数的函数。

对每个特定的w（fixed w），绘制$f_w(x)$并计算对应的$J(w)$值，可以绘制如下曲线

![image-20230601113447469](D:\CS\Machine Learning\4-Linear Regression.assets\image-20230601113447469.png)

以上是对一元损失函数的直观可视化



# More Visualize of the Cost Function

上面展示了损失函数随着w一个参数的变化，接下来，对w和b进行综合考虑。即J(w, b)，使用等高线图或者3维视图绘制如下

![image-20230602184014602](D:\CS\Machine Learning\4-Linear Regression.assets\image-20230602184014602.png)

![image-20230602184023657](D:\CS\Machine Learning\4-Linear Regression.assets\image-20230602184023657.png)

可以看到，函数表面呈“碗形”，存在最低点对应的(w, b)，使得损失最小。

在这个例子中，我们通过手动打点的方式绘制并找到了最低点，这种方法明显不适合更为复杂的机器学习模型；接下来将介绍机器学习中非常重要的概念——梯度下降（Gradient Descent），该算法用于自动化找到使成本最低的组合。



# Gradient Descent

![image-20230604074037460](D:\CS\Machine Learning\4-Linear Regression.assets\image-20230604074037460.png)

如图所示，从直观上理解梯度下降，就是在每一步选择当前下降最快的那个方向走一步，最终到达一个局部最低点的过程。



在具有两个参数的损失函数J(w, b)中，算法的数学表达式如下
$$
w = w - \alpha \frac {\part}{\part w} J(w, b)
\newline
b = b - \alpha \frac {\part}{\part b} J(w, b)
$$
其中，$\alpha$是学习率（Learning rate），该参数控制每次下降时跨出的一步有多长；过小的学习率会使训练缓慢，过大的学习率会让算法冲过最低点。

> 需要注意的是，w和b参数应该是**同步更新（Simultaneously update）**的，即在实现算法的时候，应当先使用变量暂存w和b的旧值，使用旧值计算出新值并赋值
> $$
> tmp\_w = w - \alpha \frac {\part}{\part w} J(w, b)
> \newline
> tmp\_b = b - \alpha \frac {\part}{\part b} J(w, b)
> \newline
> w = tmp\_w
> \newline
> b = tmp\_b
> $$



### 梯度下降有效性

进一步从图像角度上来解释梯度下降为什么有效，如图，在仅考虑w或仅考虑b的情况下，从某个初始值开始进行梯度下降，该点的导数为切线斜率，斜率可正可负，计算结果表现为向最低点移动。

![image-20230604081006192](D:\CS\Machine Learning\4-Linear Regression.assets\image-20230604081006192.png)



### 学习率（learning rate）

继续对学习率进行深入理解。不合适的学习率可能会导致无法梯度下降，如下图所示。当学习率过低的时候，导数乘一个非常小的学习率，因此每次下降的步长非常小，虽然可以获得最低成本，但耗时会非常的长；当学习率过高的时候，步长会过大，冲过最低点，甚至使成本更高，最终导致无法收敛（converge）。

![image-20230606092232087](D:\CS\Machine Learning\4-Linear Regression.assets\image-20230606092232087.png)



当到达最低点的时候，导数值为0，此时不会继续下降。



# Gradient Descent for Linear Regression

在这里实现线性回归的梯度下降。给出对w偏导公式如下
$$
\frac {\part}{\part w} J(w, b) = \frac {\part} {\part w} \frac {1} {2m} \sum_{i = 1}^{m} (f_{w, b}(x_i) - y_i)^2
$$
根据导数运算法则，和的导数等于导数的和，对w求偏导可得
$$
\frac {\part}{\part w} J(w, b) = \frac {1} {2m} \sum_{i = 1}^m \frac {\part}{\part w} (w x_i + b - y_i)^2
\newline 
= \frac {1} {2m} \sum_{i = 1}^m 2(w x_i + b - y) * x_i
\newline
= \frac {1} {m} \sum_{i = 1}^m (w x_i + b - y) * x_i
$$
同理，推导出对b的偏导
$$
\frac {\part} {\part b} J(w, b) = \frac {\part}{\part b} \frac 1 {2m} \sum_{i = 1}^m(f_{w, b}(x_i) - y_i)^2
\newline 
= \frac 1 {2m} \sum_{i = 1}^m \frac{\part}{\part b} (w x_i + b - y_i)^2
\newline
= \frac 1 m \sum_{i = 1}^m (w x_i + b - y_i)
$$
