# Motivation

线性回归（Linear Regression）不能很好地解决分类问题，例如检测垃圾邮件，或者识别恶性肿瘤等。

以识别恶性肿瘤为例，$y$只会取“是”或“否”两个值，属于**二分类问题（binary classification)**，通常对两个分类取值0和1，或者true和false。如图所示，使用线性回归，会使得分界线（在这个例子中，分界线为$y = 0.5$），或者称决策边界（decision boundary）发生偏移。

<img src="D:\CS\Machine Learning\7-Classification.assets\image-20230620120000511.png" alt="image-20230620120000511" style="zoom:67%;" />

<img src="D:\CS\Machine Learning\7-Classification.assets\image-20230620115930554.png" alt="image-20230620115930554" style="zoom: 67%;" />



# Logistic Regression

## From unit step to Logistic

对于二分类任务，应将线性模型产生的实值$z = \vec w^T \vec x + b$，转换为0/1值，最理想的函数是**单位阶跃函数（Unit Step Function）**
$$
H(x) = 
\left\{
\begin{array}{}
1, x > 0 \\
0, x < 0
\end{array}
\right.
$$
![](D:\CS\Machine Learning\7-Classification.assets\unit step function.png)

但这个函数不连续，所以只能考虑寻找一个近似的、单调可微的替代函数（surrogate function）。替代函数应当能够拟合一条S形曲线（S-shaped curve），如图

![image-20230620173827287](D:\CS\Machine Learning\7-Classification.assets\image-20230620173827287.png)



这种曲线为S形的函数，中文直接称S型函数或者乙状函数（Sigmoid function）；对数几率函数（Logistic function）是一种常见的S型函数，其表达式为
$$
g(z) = \frac 1 {1 + e^{-z}}, 0 < g(z) < 1
$$


## Logistic Regression

通过上面的讨论，以及[多元线性回归](5-Multiple Features.md)中对特征设计与广义线性模型的讨论，我们可以定义出使用对数几率回归：
$$
f_{\vec w, b}(\vec x) = g(\vec w \vec x + b) = \frac 1 {1 + e^{-(\vec w \vec x + b)}},
$$
我们可以将该函数的输出理解为，当给定输入$\vec x$时， $y = 1$的概率（probability）。对应的，$y = 0$的概率为$1 - f_{\vec w, b}(\vec x)$。对率回归虽然名字中带有“回归”，但实际上是一种分类算法。

> 有些时候能够看到形如$f_{\vec w, b}(\vec x) = P(y = 1| \vec x; \vec w, b)$的表述方式，该表述方式的意思是，已知$\vec x$发生、参数$\vec w, b$的条件下，$y = 1$的后验概率



# Logistic的由来

logistic这个名称具有很大的迷惑性，西瓜书将其翻译为“对数几率函数”，也有人觉得这个词来源于logic，因此翻译为“逻辑函数”。从函数定义上来看
$$
g(z) = \frac {1} {1 + e^{-z}}
$$
该函数似乎和对数、逻辑没有明显的关联。



根据参考文献[1]("D:\CS\Machine Learning\阅读材料\MathAlive.pdf")，我们可以找到一些线索，显示出logistic这个名字，很可能来取自“log-like"，而在当时那个时期，所谓的”对数曲线（logarithm curve）“，其实是现在通称的指数曲线；即，提出该函数的作者，可能是想表达该函数在一定区间内，具有”类似指数函数“的性质，因此命名为”logistic function"。如图所示。

<img src="D:\CS\Machine Learning\7-Classification.assets\Courbe_logistique,_Verhulst,_1845.png" alt="Courbe_logistique,_Verhulst,_1845" style="zoom:50%;" />