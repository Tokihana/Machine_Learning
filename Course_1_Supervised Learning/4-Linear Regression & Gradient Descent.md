

# 写在前面

我想了想，还是应该把这一章总结为**线性回归与梯度下降**，因为梯度下降不止用于线性回归，应该将这两个话题划分开来。

# 线性回归（Linear Regression）

为数据拟合（fitting）一条直线。线性回归的很多思想在其他model中也有体现。

严格定义的话，线性模型一般可以写成：
$$
f(x) = w_1x_1 + w_2x_2 + ... + w_dx_d + b
$$
或者使用向量形式
$$
f(x) = w^Tx + b
$$


还是房价的那个例子，给出一系列房屋大小-房价的数据，绘图（plot）

![image-20230512160028918](.\4-Linear Regression.assets\image-20230512160028918.png)



也可以绘制Data table

![image-20230512160301990](.\4-Linear Regression.assets\image-20230512160301990.png)



# Terminology 术语

- Training set，训练集，用来训练的数据。
- feature，特征，也叫input，或者映射关系中的x
- target，目标，也叫output，映射关系中的y
- $(x, y)$代表训练集中的一个样例，$(x^{(i)}, y^{(i)})$代表第i个样例。





# Linear Regression with One Variable

![image-20230526193948648](.\4-Linear Regression.assets\image-20230526193948648.png)

如图，在监督式学习中，学习算法使用训练集学习出一个Model（或者说是一种函数），这个model接受x输入，并预测y的值，记$\hat y$；



定义直线二维坐标下直线的公式
$$
f_{w, b}(x_i) = wx_i + b, \ 使得f_{w, b}(x) \simeq y_i
$$
称有一个输入变量的线性回归公式（也叫Univariate linear egression）。



# Cost Function

用于衡量模型的好坏。构建成本函数可以自动化优化模型。

给定一组数据和一个模型，使用模型预测出y的预测值$\hat y$，将估计值与y的真值（true value）进行比对。从直观上，单组预测值与真值之间的关系可以描述为二者在空间上的距离关系，即$\sqrt{(y - \hat y)^2}$，也可以用$(y - \hat y)^2$来表示这种关系，称平方误差（square error）。将整组数组的误差求和，就可以得到衡量模型整体好坏的总平方误差（total squarre error）。

总平方误差会随着数据量的变化而变化，例如，假定单组数据的平方误差都为1，对于10组数据，总平方误差为10，对于100组数据，总平方误差为100；为了能够比较不同数据量训练出来的模型之间的好坏，还应该对总平方误差取均值，获得平均平方误差（average square error，或称**均方误差**，**平方损失**）。习惯上，我们还会多除一个2以简化计算（因为后面要求导，求导会出来一个2）
$$
Average\ Square\ Errror = \frac 1 {2m}\sum_{i = 1}^{m}(\hat y - y)^2
$$

> 均方误差是回归任务中最常用的性能度量，对应“**欧氏距离（Euclidean distance）**”，通过最小化均方误差来求解模型的方法称为“**最小二乘法（least square method）**”

# Cost Functionb Intuition

在这里建立下对成本函数的直观理解。

在回归任务中，我们的目标是让成本函数最小化，如下图

![image-20230601112942351](.\4-Linear Regression.assets\image-20230601112942351.png)



考虑一个更为简化的模型，假定b恒=0，只考虑w的变化，则回归函数$f_w(x)$与损失函数$J(w)$分别为以x和以w为参数的函数。

对每个特定的w（fixed w），绘制$f_w(x)$并计算对应的$J(w)$值，可以绘制如下曲线

![image-20230601113447469](.\4-Linear Regression.assets\image-20230601113447469.png)

以上是对一元损失函数的直观可视化



# More Visualize of the Cost Function

上面展示了损失函数随着w一个参数的变化，接下来，对w和b进行综合考虑。即J(w, b)，使用等高线图或者3维视图绘制如下

![image-20230602184014602](.\4-Linear Regression.assets\image-20230602184014602.png)

![image-20230602184023657](.\4-Linear Regression.assets\image-20230602184023657.png)

可以看到，函数表面呈“碗形”，存在最低点对应的(w, b)，使得损失最小。

在这个例子中，我们通过手动打点的方式绘制并找到了最低点，这种方法明显不适合更为复杂的机器学习模型；接下来将介绍机器学习中非常重要的概念——梯度下降（Gradient Descent），该算法用于自动化找到使成本最低的组合。



# Gradient Descent

![image-20230604074037460](.\4-Linear Regression.assets\image-20230604074037460.png)

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

![image-20230604081006192](.\4-Linear Regression.assets\image-20230604081006192.png)



### 学习率（learning rate）

继续对学习率进行深入理解。不合适的学习率可能会导致无法梯度下降，如下图所示。当学习率过低的时候，导数乘一个非常小的学习率，因此每次下降的步长非常小，虽然可以获得最低成本，但耗时会非常的长；当学习率过高的时候，步长会过大，冲过最低点，甚至使成本更高，最终导致无法收敛（converge）。

![image-20230606092232087](.\4-Linear Regression.assets\image-20230606092232087.png)



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


对于线性回归，不会存在多个局部最小值，而是存在一个全局最小值；这类函数被称为凸函数（Convex Function）。



# Batch & Subset

Batch指每步训练都使用所有的训练数据进行训练。

Batch之外还有其他的算法，每次使用训练集的一个子集（Subset）进行训练。



# 导数可视化

对单个参数，可以使用函数曲线进行可视化

![image-20230607111006004](.\4-Linear Regression.assets\image-20230607111006004.png)

如果想要同时表示两个参数，可以使用“箭头图”。箭头的大小反映导数大小；箭头的方向表示了该点两个导数的比值。

![image-20230607111042328](.\4-Linear Regression.assets\image-20230607111042328.png)



# 运行梯度下降

![image-20230607112055005](.\4-Linear Regression.assets\image-20230607112055005.png)



从输出中可以看出，成本在逐渐下降。导数最开始很大，然后逐渐变小。下降速度也随着变慢。在固定学习率的情况下，下降速度仍随着导数变慢。



# 成本下降可视化

因为最开始下降快，后面下降慢，所以分两段绘制

成本应该一致下降，并且先快后慢。

![image-20230607113403789](.\4-Linear Regression.assets\image-20230607113403789.png)



绘制等高线图来查看随着w和b以及成本的变化。

![image-20230607114209249](.\4-Linear Regression.assets\image-20230607114209249.png)



# NumPy报错：Python int too large to convert to C long

这个问题是因为numpy数组默认使用int，需要在赋值语句后面加上`.astype('float')`或者`.astype('int64')`来规定类型为float或int64

另外，在修改外部代码文件后，需要重启jupyter notebook内核，并重新运行代码才能读到修改后的代码。



# 学习率过大的可视化

如果学习率过大，会使梯度下降冲过最低点，且导致不能收敛的情况。

![image-20230607120205086](.\4-Linear Regression.assets\image-20230607120205086.png)





# 西瓜书的相关推导

## 均方误差（mean squared error）

在回归任务中最常使用的性能度量就是均方误差。
$$
E(f; D) = \frac 1 m \sum_{i = 1}^m(f(x_i) - y_i)^2
$$
更一般的情况，对数据分布D和概率密度函数$p(\cdot)$，可描述为
$$
E(f; D) = \int_{x \sim d}(f(x) - y)^2p(x)dx
$$



## 凸函数定义

对区间[a,b]上定义的函数，若对区间内任意两点，均有$f(\frac {x_1 + x_2} 2) \le \frac{f(x_1) + f(x_2)} {2} $，则称该函数为区间[a, b]上的凸函数。

换个直观点的说法就是，区间内任意两点，其割线在函数曲线的上方，类似下图

![image-20230607144758127](.\4-Linear Regression & Gradient Descent.assets\image-20230607144758127.png)



对于实数集上的函数，可以通过二阶导数判别是否为凸函数，若二阶导非负，则为凸函数，若二阶导恒大于0，则称为严格凸函数。

> 这里的凸函数定义是最优化里的定义，与高等数学中的凸函数定义不同。



## 一元线性回归闭式解推导

一元线性回归的最优化过程，就是求解w和b，使$E_{w, b} = \sum_{i = 1}^m(wx_i + b - y_i)^2$最小

> 严格上说，西瓜书这里用的不是均方误差，因为没有求均值

分别对w和b求偏导
$$
\frac {\part E_{w, b}}{\part w} = \sum_{i = 1}^m \frac {\part} {\part w}(wx_i + b - y_i)^2
\newline 
= 2\sum_{i=1}^m (wx_i + b - y_i) x_i
\newline 
= 2 (w\sum_{i=1}^m x_i^2 - \sum_{i=1}^m(y_i - b)x_i)
\newline
\newline
\frac {\part E_{w, b}}{\part b} = \sum_{i = 1}^m \frac {\part} {\part b}(wx_i + b - y_i)^2
\newline 
= 2 \sum_{i=1}^m (wx_i + b - y_i)
\newline
= 2 (mb - \sum_{i=1}^m (y_i - wx_i))
$$


进一步，得到w和b的闭式解（closed-formed solution，也叫解析解analytical solution），等价于吴恩达老师给出的w和b的计算式（这里我从吴恩达老师的公式开始，推导西瓜书的表达式）

> 闭式解指可以通过具体的表达式求出待解参数。例如可以直接根据上面的表达式求得w，机器学习算法很少有闭式解，线性回归是特例。

先推导b，因为后面推w会用到
$$
b = b - \alpha \frac {\part}{\part b} J(w, b)
\newline 
= b - \frac 1 m \sum_{i = 1}^m (w x_i + b - y_i)
\newline\rightarrow
\sum_{i = 1}^m (w x_i + b - y_i) = 0
\newline\rightarrow
mb = \sum_{i = 1}^m(y_i - wx_i)
\newline\rightarrow
b = \frac 1 m \sum_{i = 1}^m(y_i - wx_i)
\newline\rightarrow
b = \bar y - w \bar x
$$
接着推导w
$$
w = w - \alpha \frac {\part}{\part w} J(w, b)
\newline
= w - \frac {1} {m} \sum_{i = 1}^m (w x_i + b - y) * x_i
\newline\rightarrow
(w\sum_{i=1}^m x_i^2 - \sum_{i=1}^m(y_i - b)x_i) = 0
\newline带入b\rightarrow
w\sum_{i=1}^m x_i^2 = \sum_{i=1}^m y_ix_i - \sum_{i = 1}^m (\bar y - w \bar x)x_i
\newline\rightarrow
w \sum_{i=1}^m x_i^2 = \sum_{i=1}^m y_ix_i - \bar y  \sum_{i = 1}^m x_i + w \bar x\sum_{i = 1}^mx_i
\newline\rightarrow
w (\sum_{i=1}^m x_i^2 - \bar x\sum_{i = 1}^mx_i) = \sum_{i=1}^m y_ix_i - \bar y  \sum_{i = 1}^m x_i
\newline\rightarrow
w = \frac {\sum_{i=1}^m y_ix_i - \bar y  \sum_{i = 1}^m x_i}{\sum_{i=1}^m x_i^2 - \bar x\sum_{i = 1}^mx_i}
\newline
令\bar y  \sum_{i = 1}^m x_i = \frac 1 m \sum_{i = 1}^m y_i \sum_{i = 1}^m x_i = \bar x \sum_{i = 1}^m y_i
\newline
\bar x\sum_{i = 1}^mx_i = \frac 1 m (\sum_{i = 1}^m x_i)^2
\newline
得w = \frac {\sum_{i = 1}^m y_i (x_i - \bar x)}{\sum_{i=1}^m x_i^2 - \frac 1 m (\sum_{i = 1}^mx_i)^2}
\newline

\newline
$$
综上，w和b最优解的闭式解为
$$
w = \frac {\sum_{i = 1}^m y_i (x_i - \bar x)}{\sum_{i=1}^m x_i^2 - \frac 1 m (\sum_{i = 1}^mx_i)^2}
\newline
b = \frac 1 m \sum_{i = 1}^m(y_i - wx_i)
$$





## 补充：最小二乘估计与极大似然估计

基于均方误差最小化来进行模型求解的方法称为”最小二乘法“，前一小节已经讨论过很多了。
$$
\arg\min_{(w,b)}E_{(w, b)} = \sum_{i = 1}^m(y_i - f(x_i))^2
\newline
= \sum_{i = 1}^m(y_i - (wx_i + b))^2
$$

> $\arg\min_{(w,b)}$意思是使表达式值最小的参数w和b



与最小二乘估计殊途同归的是最大似然估计。似然可以理解为**某个参数取特定值的可能性**，极大似然估计则是通过寻找可能性的最大值点，找到最有可能的参数值。两者为何殊途同归可以参考[南瓜书的视频P2](https://www.bilibili.com/video/BV1Mh411e7VU/?p=2&vd_source=fbee134f28f10053951823e9c44f4191)



## 一元线性回归闭式解的向量化

参考南瓜书，利用NumPy等线性代数库加速，对之前推导的一元线性回归闭式解进行向量化

此前推导的一元线性回归闭式解
$$
w = \frac {\sum_{i = 1}^m y_i (x_i - \bar x)}{\sum_{i=1}^m x_i^2 - \frac 1 m (\sum_{i = 1}^mx_i)^2}
\newline
b = \frac 1 m \sum_{i = 1}^m(y_i - wx_i)
$$
这个解中的求和运算只能使用循环实现（事实上吴恩达老师上一小节的notebook里就是用循环实现的），我们需要进一步向向量化靠拢。

已知形如$\sum x_iy_i$的求和可以被表示为向量$x$和$y$的点乘，或者表示为类似$x^Ty$的形式。进行凑配
$$
w = \frac {\sum_{i = 1}^m y_i (x_i - \bar x)}{\sum_{i=1}^m x_i^2 - \frac 1 m (\sum_{i = 1}^mx_i)^2}
\newline
= \frac {\sum_{i = 1}^m y_i (x_i - \bar x)}{\sum_{i=1}^m x_i^2 - \bar x \sum_{i = 1}^mx_i}
\newline
= \frac {\sum_{i = 1}^m y_i (x_i - \bar x)}{\sum_{i=1}^m (x_i^2 - \bar x x_i)}
\newline
\because 
\left.\begin{array}{}
\sum_{i=1}^m \bar x x_i = \sum_{i=1}^m x_i \bar x = m\bar x^2 = \sum_{i=1}^m \bar x^2\\
\sum_{i = 1}^m y_i \bar x = \sum_{i = 1}^m x_i \bar y = m\bar x \bar y =\sum_{i = 1}^m \bar x \bar y
\end{array}\right.
\newline
\therefore
w = \frac {\sum_{i = 1}^m (x_iy_i - \bar x y_i - x_i\bar y + \bar x \bar y)}{\sum_{i=1}^m (x_i^2 - \bar x x_i - x_i\bar x + \bar x^2)}
\newline
=\frac {\sum_{i = 1}^m (x_i - \bar x)(y_i - \bar y)}{\sum_{i=1}^m (x_i - \bar x)^2}
\newline
令\ x_d = (x_1 - \bar x, x_2 - \bar x, \dots, x_n - \bar x), y_d = (y_1 - \bar y, y_2 - \bar y, \dots, y_n - \bar y)
\newline
w = \frac{x_d^T y_d}{x_d^T x_d}
$$


下面尝试编程实现，测试计算$\bar a$与$a_d$

```py
>>> import numpy as np
>>> a = np.arange(10); print(a)
[0 1 2 3 4 5 6 7 8 9]
>>> a_bar = a.mean(); print(a_bar)
4.5
>>> a_d = a - a_bar; print(a_d)
[-4.5 -3.5 -2.5 -1.5 -0.5  0.5  1.5  2.5  3.5  4.5]
```

根据闭式解直接求得结果

```py
>>> # init parameters
... b = 0
>>> w = 0
>>> # set data
... x = np.arange(10); print(f'x = {x}')
x = [0 1 2 3 4 5 6 7 8 9]
>>> y = np.arange(5, 15); print(f'y = {y}')
y = [ 5  6  7  8  9 10 11 12 13 14]
>>> x_bar = x.mean()
>>> y_bar = y.mean()
>>> x_d = x - x_bar
>>> y_d = y - y_bar
>>> w =  np.dot(x_d, y_d)/ np.dot(x_d, x_d)
>>> b = (y - w*x).mean()
>>> print(w, b)
1.0 5.0
```

> 在这里强调一下，闭式解是可以直接根据表达式求得参数的，例如这里直接求得了w和b，**没有进行梯度下降**等方法去拟合一个模型来预测。正如南瓜书所说，机器学习算法很少有闭式解，线性回归是个特例。
