# Underfitting & Overfitting

在训练模型的过程中，我们通常希望模型能够很好地反映训练集的特征的同时，也能够泛化（generalization）到新的数据上（在新的数据集上也能表现得很好）。

在训练集上表现不好的模型，称为欠拟合（underfit），也称模型与训练集有很高的偏差（bias）；在训练集上表现得很好，在新数据上表现不好的模型，称为过拟合（overfit），也称有高方差（high variance）。如图所示，在房价预测模型中设计不同阶的特征，左边的模型欠拟合，右边的模型过拟合。

![image-20230628171136482](.\9-Overfitting & Regularization.assets\image-20230628171136482.png)



对于分类任务，同样需要关注欠拟合与过拟合问题，如图所示，过拟合和欠拟合的模型都不能称得上是良好的模型。

![image-20230628171826833](.\9-Overfitting & Regularization.assets\image-20230628171826833.png)



# Addressing Overfitting

> 后面会讲如何debug和检测过拟合，这里先讲如何处理过拟合

- 获取更多的训练数据（复杂的特征需要更多数据的支持）。有的时候很难实现。

  ![image-20230628172058432](.\9-Overfitting & Regularization.assets\image-20230628172058432.png)

- 使用更少的特征，尤其是高阶多项式特征，并尝试通过特征设计选择合适的特征（feature selection），但注意，排除掉部分特征可能导致信息丢失。可以通过自动化手段来选择这些特征。

- 正则化（Regularization）。保留所有的特征，减少过大特征的影响（缩小对应特征的数量级）；通常只需要正则化$w_j$，不需要正则化$b$





# Regularization

前面提到，使用正则化的目的是使$w_j$尽可能的小，从而减小每个特征的影响，降低过拟合的可能性；$w_j$尽可能的小，也即$w_j$尽可能趋向0，我们可以使用最小二乘来建模这一关系
$$
\arg \min_{\vec w, b} \sum_{j = 1}^n (w_j-0)^2
$$
这个式子将被带入cost函数中，作为一个惩罚项存在，以线性回归的最小二乘cost为例
$$
\arg \min_{\vec w, b} [\frac 1 {2m} \sum_{i = 1}^m (f_{\vec w, b}(\vec x_i) - y_i)^2 + \frac \lambda {2m} \sum_{j = 1}^n w_j^2]
$$
其中，新引入的式子称为正则项，$\lambda > 0$为正则参数；在上式中，均方误差项最小化使模型拟合（fit data），正则项最小化保证$w_j$不会过高，两者之间的平衡通过$\lambda$调整。过小的$\lambda$不能有效抑制过拟合；而过大的$\lambda$会将特征值压得过小，导致模型趋近$y = b$，表现为欠拟合。

> 正则项除2m，吴恩达老师的解释是在实践上，这样更有利于选择稳定的$\lambda$。
>
> 更标准的说法是$L_p$范数，$L_0, L_1$范数使分量稀疏，非零分量个数尽可能少，因为$L_0$不连续、非凸、不可微，所以用$L_1$代替；$L_2$范数（类似上面取平方的形式）使分量稠密，且更加均衡。
>
> Scikit-Learn的`Ridge`线性模型就使用了$L_2$范数





## Regularized Linear Regression

### Equations

已知引入正则化之后的cost
$$
J(\vec w, b) =\frac 1 {2m} \sum_{i = 1}^m (f_{\vec w, b}(\vec x_i) - y_i)^2 + \frac \lambda {2m} \sum_{j = 1}^n w_j^2
$$
使用梯度下降最优化参数，求导如下
$$
\frac {\partial} {\partial w_j} J(\vec w, b) = \frac 1 m \sum_{i = 1}^m(f_{\vec w, b}(\vec x_i) - y_i)x_{i, j} + \frac \lambda m w_j
\newline
\frac {\partial} {\partial b} J(\vec w, b) = \frac 1 m \sum_{i = 1}^m(f_{\vec w, b}(\vec x_i) - y_i)
$$
因为没有对b正则化，因此对b的偏导没有变化。



可以通过重新整理表达式，来更直观观察正则化对参数更新的影响
$$
w_j = w_j - \alpha \frac {\partial} {\partial w_j} J(\vec w, b) 
\newline 
= w_j - \alpha [\frac 1 m \sum_{i = 1}^m(f_{\vec w, b}(\vec x_i) - y_i)x_{i, j} + \frac \lambda m w_j]
\newline
= (1 - \alpha \frac \lambda m)w_j - \alpha rac 1 m \sum_{i = 1}^m(f_{\vec w, b}(\vec x_i) - y_i)x_{i, j}
$$
假定$\alpha = 0.01, \lambda = 1, m = 50$，可知$w_j$的系数为`0.9998`，即正则化在每轮迭代将参数缩小一点点。



### Implements

首先实现Cost的计算

```py
def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
	'''
	Computes total cost of all examples
	Args:
		X (ndarray (m, n)): Data, m examples, n features
		y (ndarray (m, )): target values
		w (ndarray (n, ))： parameters
		b (scalar) : interpret parameter
		lambda_ (scalar) : the amout of regularization, add "_" to distinguish from key word "lambda"
		
    Returns:
    	total_cost (scalar): the total cost of all examples
    '''
    m, n = X.shape
    errors = np.dot(X, w) + b - y
    cost = (1/2/m) * np.dot(errors, errors) + (lambda_/2/m) * np.dot(w, w)
    return cost
```

> `np.dot(a,b)`算法支持多种支持多种运算规则，在这里`np.dot(X, w)`相当于对w做一次线性变换。需要注意的是，这种计算方式是会进行广播（broadcast）的。当`a, b`都是1-D array的时候，计算向量内积。
>
> `np.matmul(x1,x2)`，或者说运算符`@`提供矩阵乘法，要求shape满足*(n,k),(k,m)->(n,m)*。`x1, x2`不能是scalar或array；虽然用`np.dot`也可实现这一运算，但推荐使用`@`。
>
> `np.multiply(a, b)`或者说运算符`*`提供数乘操作，要求任意一个输入为scalar。同样可以用`np.dot`实现，但推荐使用`*`。



梯度下降函数不会因为引入正则项发生变化，但计算梯度的函数需要做出调整

```py
def compute_gradient_linear_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)

    errors = np.dot(X, w) + b - y
    dj_db = errors.mean()
    dj_dw = np.dot(errors.reshape(1, -1), X).reshape(-1, )/m + (lambda_/m) * w
    # 或者这样写也可以，但是逻辑上会变成(m, ) dot (m, n)，
    # 虽然numpy会自动修正过来，但最好先reshape
    # dj_dw = np.dot(errors, X)/m + (lambda_/m) * w
    
    return dj_db, dj_dw
```



## Regularized Logistic Regression

### Equations

同理，在对率回归中引入$L_2$范数项
$$
J(\vec w, b) = -\frac 1 m \sum_{i = 1}^m y_i [\log(f_{\vec w, b}(\vec x_i)) + (1 - y_i) \log(1 - f_{\vec w, b}(\vec x_i))] +\frac \lambda m w_j^2
$$


引入正则项不会对似然项的求导造成影响，因此求得导数
$$
\frac \partial {\partial \vec w_j}J(\vec w, b)
= \frac 1 m \sum_{i = 1}^m(f_{\vec w, b}(\vec x_i) - y_i) \vec x_{i, j} + \frac \lambda m w_j
\newline
\frac \partial {\partial b}J(\vec w, b) = \frac 1 m \sum_{i = 1}^m(f_{\vec w, b}(\vec x_i) - y_i)
$$
代入梯度下降表达式中，即可求得参数迭代式

> 在Scikit-Learn中，`LogisticRegression`模型支持通过`penalty`参数来选用正则化
>
> | penalty | r(w)                                   |
> | ------- | -------------------------------------- |
> | None    | 0                                      |
> | $l_1$   | $||w||_1$                              |
> | $l_2$   | $\frac 1 2 ||w||^2_2 = \frac 1 2 w^Tw$ |



### Implements

计算cost

```py
def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m,n  = X.shape
    reg_cost = (lambda_/2/m) * np.dot(w, w)
    f_wbs = sigmoid(np.dot(X, w) + b)
    cost = -(np.dot(y, np.log(f_wbs)) + np.dot(1-y, np.log(1-f_wbs)))/m                                                 #scalar

    total_cost = cost + reg_cost
    return total_cost
```

> sigmoid函数在此前实现过，核心代码为
>
> ```py
> g = 1.0/(1.0+np.exp(-z))
> ```



计算梯度

```py
def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape

    errors = sigmoid(np.dot(X, w) + b) - y
    dj_db = errors.mean()
    dj_dw = np.dot(errors.reshape(1, -1), X).reshape(-1, )/m + (lambda_/m) * w

    return dj_db, dj_dw  
```

