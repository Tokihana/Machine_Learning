# Motivation

线性回归（Linear Regression）不能很好地解决分类问题，例如检测垃圾邮件，或者识别恶性肿瘤等。

以识别恶性肿瘤为例，$y$只会取“是”或“否”两个值，属于**二分类问题（binary classification)**，通常对两个分类取值0和1，或者true和false。如图所示，使用线性回归，会使得分界线（在这个例子中，分界线为$y = 0.5$），或者称决策边界（decision boundary）发生偏移。

<img src=".\7-Classification.assets\image-20230620120000511.png" alt="image-20230620120000511" style="zoom:67%;" />

<img src=".\7-Classification.assets\image-20230620115930554.png" alt="image-20230620115930554" style="zoom: 67%;" />



## Multiple features

分类问题的特征也可以是多元的，不同的输出值通常用不同图标表示，例如

```py
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])
```

![multiple_features](.\7-Classification.assets\multiple_features.png)



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
![](.\7-Classification.assets\unit step function.png)

但这个函数不连续，所以只能考虑寻找一个近似的、单调可微的替代函数（surrogate function）。替代函数应当能够拟合一条S形曲线（S-shaped curve），如图

![image-20230620173827287](.\7-Classification.assets\image-20230620173827287.png)



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



## Implement Logistic Function

使用[`np.exp()`](https://numpy.org/doc/stable/reference/generated/numpy.exp.html)来实现对率函数

```py
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
    
    z = np.clip( z, -500, 500 )           # protect against overflow
    g = 1/(1+np.exp(-z))
   
    return g
```

> `np.exp()`既可以接收单个值，也可以接收一个向量，并对向量中的每个元素求指数。





## Why "Logistic"?

> 这里讨论名称由来是为了强化记忆，和主要内容关联性不大，可以跳过

logistic这个名称具有很大的迷惑性，西瓜书将其翻译为“对数几率函数”，也有人觉得这个词来源于logic，因此翻译为“逻辑函数”。从函数定义上来看
$$
g(z) = \frac {1} {1 + e^{-z}}
$$
该函数似乎和对数、逻辑没有明显的关联。



根据参考文献[1](".\阅读材料\MathAlive.pdf")，我们可以找到一些线索，显示出logistic这个名字，很可能来取自“log-like"，而在当时那个时期，所谓的”对数曲线（logarithm curve）“，其实是现在通称的指数曲线；即，提出该函数的作者，可能是想表达该函数在一定区间内，具有”类似指数函数“的性质，因此命名为”logistic function"。如图所示。

<img src=".\7-Classification.assets\Courbe_logistique,_Verhulst,_1845.png" alt="Courbe_logistique,_Verhulst,_1845" style="zoom: 33%;" />



而西瓜书翻译为对数几率，则是根据
$$
y = \frac {1} {1 + e^{-(\vec w^T \vec x + b)}}
\newline
\rightarrow \ln \frac {y} {1-y} = \vec w^T \vec x + b
$$
设$y$为二分类问题中，样本取正例的可能性（probability），$\frac y {1-y}$即为正例与反例可能性的比值，称为**几率（odds）**，再加一个$\ln$，称**对数几率（log odds, 也做logit）**。



# Decision boundary

在对率回归模型中，通常假定$g(z) = 0.5$为模型的决策边界。

![](.\7-Classification.assets\sigmoid_funciton.png)

如图所示，$g(z) = 0.5$，代表$z = 0$，而$z$对应线性模型$\vec w \vec x + b$，因此有
$$
\vec w \vec x + b \geq 0, y = 1
\newline
\vec w \vec x + b < 0, y = 0
$$


此时，$z = \vec w \vec x + b = 0$构成一个边界（在二元的情况下，这个边界是一条曲线），分割了两种不同的类别。

以一个简单的数据集为例

```py
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 
```

![](.\7-Classification.assets\simple_data.png)



假设通过经过学习后，我们获得了参数 $b = -3, w_0 = 1, w_1 = 1$，则对率回归模型为
$$
f(x) = g(x_0+x_1-3)
$$
模型的边界为$x_0 + x_1 - 3 = 0$，整理后进行绘图

![](.\7-Classification.assets\decision_boundary.png)

如图，位于蓝色区域的点，将被预测为$y = 0$，而位于白色区域的点，则被预测为$y = 1$，蓝色直线就是模型的决策边界



决策边界同样可以是非线性的(non-linear)，利用我们此前学到的特征设计和多项式回归，我们可以绘制出更加复杂的决策边界，例如

![image-20230621164547290](.\7-Classification.assets\image-20230621164547290.png)



# Loss Function

## Definitions

首先明确下Loss和Cost的定义，吴恩达老师在课程中对这两个词的定义为：

- Loss：单个样例的预测值与结果只之间的差异
- Cost：整个训练集的Loss

后面为了消歧义，会采用英文表述



## Squared Error Loss

在此前的线性回归中，我们使用平方误差（Squared error）作为Loss函数，表达式为
$$
Loss(f_{\vec w, b}(\vec x_i), y_i) = (f_{\vec w, b}(\vec x_i) - y_i)^2
\newline
其中，f_{\vec w, b}(\vec x_i) = \vec w^T\vec x + b
$$
Cost函数写作
$$
J(\vec w, b) = \frac 1 {2m} \sum_{i = 0}^{m - 1} (f_{\vec w, b}(\vec x_i) - y_i)^2
$$
在线性回归中，该函数是凸函数，我们对Cost求偏导，调整参数移动到局部最低点。



对于对率回归任务，我们很自然地想尝试用同样的Cost函数。为了确定平方误差Cost是否适用于对率回归，我们可以绘图观察
$$
J(\vec w, b) = \frac 1 {2m} \sum_{i = 0}^{m - 1} (f_{\vec w, b}(\vec x_i) - y_i)^2
\newline
f_{\vec w, b}(\vec x_i) = logistic(\vec w^T\vec x + b)
$$
![](.\7-Classification.assets\logistic_squared_error.png)

很明显，这个函数不是个凸（Convex）函数，存在非常多的局部最低点，不适合梯度下降。

> 实际上，Logistic regression是可以使用最小二乘的，不过不能用这种经典的最小二乘（或者叫普通最小二乘，Ordinary least squares），而应该使用加权最小二乘（weighted least squares）。详见[wiki](hhttps://en.wikipedia.org/wiki/Logistic_regression#Model_fitting)。不过一般还是用**极大似然估计（Maximum Likelihood Estimation）**。



## Likelihood Loss

定义似然函数（Likelihood function）为
$$
Loss(f_{\vec w, b}(\vec x_i), y_i) = 
\left\{
\begin{array}{}
-\log(f_{\vec w, b}(\vec x_i)), y_i = 1\\
-\log(1 - f_{\vec w, b}(\vec x_i)), y_i = 0
\end{array}
\right.
$$
> 这里用的极大似然（MLE），大致参考西瓜书做些说明：
>
> 对数据集$D$，$D_c$表示其中类别为$c$的样本集合，在本课程的二分类任务中，这个集合被划分为$y = 1$与$y = 0$两类。概率分别为
> $$
> p(y = 1|\vec x) = f_{\vec w, b}(\vec x) =\frac 1 {1 + e^{-(\vec w^T \vec x + b)}} = \frac {e^{\vec w^T \vec x + b}} {1 + e^{\vec w^T \vec x + b}}
> \newline
> p(y = 0|\vec x) = 1 - p(y = 1|\vec x) = \frac 1 {1 + e^{\vec w^T \vec x + b}}
> $$
> 假设样本独立同分布（iid），得联合分布（同时发生的概率）
> $$
> P(D_{y=1}|\vec w, b) = \prod_{\vec x_i \in D_{y = 1}}f_{\vec w, b}(\vec x_i)
> \newline
> P(D_{y=0}|\vec w, b) = \prod_{\vec x_i \in D_{y = 1}}(1 - f_{\vec w, b}(\vec x_i))
> $$
> 称似然函数（likelihood function），可以想到，当似然函数最大化的时候，样本为真实标记的概率最大，联合分布最符合数据集的情况，使得似然函数最大的$\vec w, b$就是最优参数解，即
> $$
> \hat{\vec w}, \hat b = \arg \max_{\vec w, b} P(D_c|\vec w, b)
> $$
> 由于概率取值范围为[0, 1]，上式中的连乘操作易造成浮点数下溢，因此通常对似然取对数，即对数自然（log-likelihood），提取单个样例的Loss为
> $$
> Loss(f_{\vec w, b}(\vec x_i), y_i) = 
> \left\{
> \begin{array}{}
> -\log(f_{\vec w, b}(\vec x_i)), y_i = 1\\
> -\log(1 - f_{\vec w, b}(\vec x_i)), y_i = 0
> \end{array}
> \right.
> $$
> 这里加符号是为了后面梯度下降做极小化。



绘制其曲线

![](.\7-Classification.assets\two_catagiries_logistic.png)

可以看到，该函数能够满足Loss函数的行为：当预测值接近目标值的时候，逼近0，当预测值远离目标值的时候，快速增大。从而在远处快速下降，逼近时下降减慢。

# Cost function

为了方便实现，可以将上式重写为
$$
Loss(f_{\vec w, b}(\vec x_i), y_i) = -y_i \log(f_{\vec w, b}(\vec x_i)) - (1 - y_i) \log(1 - f_{\vec w, b}(\vec x_i))
$$
因为$y_i$只会取0或1，所以该式与上式等价；从而得到Cost表达式
$$
J(\vec w, b) = \frac 1 m \sum_{i = 1}^m Loss(f_{\vec w, b}(\vec x_i), y_i)
\newline
= -\frac 1 m \sum_{i = 1}^m y_i \log(f_{\vec w, b}(\vec x_i)) + (1 - y_i) \log(1 - f_{\vec w, b}(\vec x_i))
$$
计算Cost并绘制图像

![](.\7-Classification.assets\cost.png)

左图是cost，右图是取log之后的cost。图像非常平滑且是个凸曲面，可以通过梯度下降拟合参数。

> 前面取log是为了连乘变连加，方便计算的同时防止因多个(0, 1)内的概率相乘导致浮点下溢。这里取log是为了将数据压缩到较小的区间，从而在可视化的时候可以更好地观察差异。
>
> 之所以能够取对数是因为单调性相同，能够取到同一个最大似然点



## Implement

实现`compute_cost_logistic`函数

```py
import numpy as np
def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost
```



# Gradient Descent

## Partial derivatives

已知Cost
$$
J(\vec w, b)
= -\frac 1 m \sum_{i = 1}^m y_i \log(f_{\vec w, b}(\vec x_i)) + (1 - y_i) \log(1 - f_{\vec w, b}(\vec x_i))
$$
求解偏导，首先计算$g(z) = \frac 1 {1 + e^{-z}}$的导数
$$
g'(z) = \frac 1 {(1 + e^{-z})^2}e^{-z} = g(z)(1 - g(z))
$$
求取$\vec w$偏导
$$
\frac \partial {\partial \vec w_j}J(\vec w, b) = \frac \partial {\partial \vec w_j} -\frac 1 m \sum_{i = 1}^m y_i \log(f_{\vec w, b}(\vec x_i)) + (1 - y_i) \log(1 - f_{\vec w, b}(\vec x_i))
$$
分别推导两部分的导数
$$
\frac \partial {\partial \vec w_j} y \log(f_{\vec w, b}(\vec x)) 
= y * \frac 1 {f_{\vec w, b}(\vec x)}*f'_{\vec w, b}(\vec x) \frac \partial {\partial \vec w_j}(\vec w^T\vec x + b)
\newline
= y * \frac 1 {f_{\vec w, b}(\vec x)} * f_{\vec w, b}(\vec x)(1-f_{\vec w, b}(\vec x)) * \vec x_j
\newline
= y*(1-f_{\vec w, b}(\vec x))\vec x_j
\newline
\frac \partial {\partial \vec w_j}(1 - y)log(1 - f_{\vec w, b}(\vec x))
= (1 - y) * \frac 1 {1 - f_{\vec w, b}(\vec x)} * (-f'_{\vec w, b}(\vec x))\frac \partial {\partial \vec w_j}(\vec w^T \vec x + b)
\newline
= (1 - y) * \frac 1 {1 - f_{\vec w, b}(\vec x)} * -(f_{\vec w, b}(\vec x)(1-f_{\vec w, b}(\vec x)))  * \vec x_j
\newline 
= (y - 1) f_{\vec w, b}(\vec x) \vec x_j
$$
整理得
$$
\frac \partial {\partial \vec w_j}J(\vec w, b)
= \frac 1 m \sum_{i = 1}^m(f_{\vec w, b}(\vec x_i) - y_i) \vec x_{i, j}
$$


同理，求$b$偏导（去掉$\vec x_{i, j}$即可）
$$
\frac \partial {\partial b}J(\vec w, b) = \frac 1 m \sum_{i = 1}^m(f_{\vec w, b}(\vec x_i) - y_i)
$$


从而得出梯度下降迭代式
$$
w_j = w_j - \alpha[\frac 1 m \sum_{i = 1}^m(f_{\vec w, b}(\vec x_i) - y_i) \vec x_{i, j}]
\newline
b = b - \alpha[\frac 1 m \sum_{i = 1}^m(f_{\vec w, b}(\vec x_i) - y_i)]
$$
与线性回归中应用梯度下降一样，参数需要同步更新（synchronous update），需要关注模型是否收敛，可以使用向量化加速计算，使用特征缩放来调整特征。



## Implement - compute gradient

首先计算梯度，伪码逻辑如下

```py
for each example i:
	error = f_wb(X[i]) - y_i   
    for each feature j:
        dj_dw[j] += error * X[i][j]
    dj_db += error
    
dj_dw /= m
dj_db /= m
```

具体实现

```py
def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar
        
    return dj_db, dj_dw 
```



## Implement - Gradient descent

```py
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
        
    return w, b        #return final w,b and J history for graphing

```



下图为在一元二分类任务中运行梯度下降的结果

![image-20230627093950895](.\7-Classification.assets\image-20230627093950895.png)



# Predict的一些注意事项

实现对率回归的predict函数，可以直接用sigmoid求出概率后舍入。但是这里有个小细节需要注意，numpy的`round`函数采用的舍入规则是**银行舍入**，x.5会被舍入到距离最近的偶数上。且numpy没有提供方法来选择舍入规则。对某些情况，例如医学检测上，我们通常期望0.5被舍入到1，宁可错检，也不应放过可能的隐患，此时调整为`np.ceil(logistics - 0.5)`更好一些。



# 参考

- 吴恩达2022《机器学习》
- 《机器学习》周志华
- 南瓜书
- 《数理统计学导论》Robert V.Hogg, Joseph W.McKean
