# Multiple Linear Regression

在前一部分，我们研究了单个特征的线性回归，进一步，我们想要研究具有多个特征的线性回归模型。



## Terminology

$x_j$：第j个特征

n：特征的数量

$\vec x^{(i)}$：第i个训练数据的特征集

$\vec x_j^{(i)}$：第i个数据的第j个特征



## Model

对于多个特征的模型，其线性回归模型的表达式为
$$
f_{w, b} = w_1x_1 + w_2x_2 + ... + w_nx_n + b = \sum_{j = 1}^n w_jx_j + b
$$
用向量的方式表示
$$
\vec w = [w_1, w_2, ..., w_n]
\newline
\vec x = [x_1, x_2, ..., x_n]
\newline
f_{\vec w, b} = \vec w \cdot \vec x + b = \vec w \vec x^T + b
$$
这种技巧称为矢量化（vectorization）。



## 离散特征的量化

类似“房屋面积：房屋价格”这种连续的特征关系，量化较为容易；而实际场景中，有很多特征都是离散的，在此讨论下如何量化离散特征

- 对只有“是”或“否”的离散特征，可以**使用0,1**来表示，称为二值离散特征
- 对于有多个离散值，且离散值之间存在顺序关系的，例如饭量“大”，“中”，“小”，可以**使用有序的离散值**来表示，称为有序的多值离散特征（“大”：3，“中”：2，“小”：1）
- 对有多个离散值，离散值之间不具有顺序关系，例如肤色“黄”，”白“，”黑“，不能直接分配有序的离散值，而是应该**拆成多个二值离散特征**进行表示（黄：[1,0,0]，白：[0,1,0]，黑：[0,0,1]）

# Vectorization

## Code Implement

设有如下参数

![image-20230610172121447](D:\CS\Machine Learning\5-Multiple Features.assets\image-20230610172121447.png)

在Numpy中，可以这样表示

```py
import numpy as np
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10, 20, 30])
```

通过向量化的，$f_{\vec w, b}$可以实现为

```
f_wb = np.dot(w, x) + b
```



使用向量化运算有两个好处：

- 代码更加整洁可读
- Numpy底层可以并行处理，效率更快



## How could vectorization run much more faster than simply loop

当采用loop实现方法的时候，程序会顺序执行每个`f += w[i] * x[i]`

![image-20230610173009248](D:\CS\Machine Learning\5-Multiple Features.assets\image-20230610173009248.png)

而在向量化实现中，计算机可以同时并行计算多个`w[i] * x[i]`

![image-20230610173051049](D:\CS\Machine Learning\5-Multiple Features.assets\image-20230610173051049.png)

以上是使用线性代数库计算性能的直观表示



## Gradient Descent

首先关注多个feature，假定b=0，此时对多个$w_j$，存在同样数量的导数$d_j$，同样考虑一个向量化的实现方式。

```python
w = np.array([...])
d = np.array([...])
alpha = 0.1 # 学习率
# loop实现
for j in range(n):
    w[j] = w[j] - alpha * d[j]
# 向量化实现
w = w - alpha * d
```



# NumPy和Python补充

## 创建array

### 转换Python序列**（List, Tuple）**为NumPy Array

```py
import numpy as np
a1 = np.array([1, 2, 3, 4]) # 创建1D array
a2 = np.array([1, 2], [3, 4]) # 创建2D array
```

> List或Tuple的嵌套会创建高维的数组。所有的array对象在NumPy中都称为`ndarray`。即`n-dimensional array`

你可以显式指定数组元素的数据类型，使用`dtype`参数；指定参数类型要注意别越界，例如

```py
a = np.array([127， 128， 129], dtype = np.int8)
'''
array([ 127, -128, -127], dtype=int8)
'''
```

当使用两个不同dtype的array进行计算时，结果矩阵的数据类型会匹配为能够兼容两种dtype的类型。



### 内置的矩阵创建函数

NumPy包含40多个内置的函数可以创建矩阵（[Array creation routines](https://numpy.org/doc/stable/reference/routines.array-creation.html#routines-array-creation)）。这些函数可以根据数组的维数，大致分为3类：

- 1D arrays
- 2D arrays
- ndarrays

> 这里先列出吴恩达老师提到的课程要用的那些

#### 1D array

```py
# np.arange((start = 0, )end(, step = 1)) # 返回从start开始，end结束，步长为step的分布数组, start和step可缺省
>>> np.arange(10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.arange(2, 10, dtype = float)
array([2., 3., 4., 5., 6., 7., 8., 9.])
>>> np.arange(2, 3, 0.1)
array([2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])

# np.linspace(start, end, num) # 返回从start开始，end结束，长度为num的数组，step呈均匀分布
>>> np.linspace(1., 4., 6)
array([1. , 1.6, 2.2, 2.8, 3.4, 4. ])
```



#### nD array

```py
# np.zeros(shape: Tuple) # 生成多维array，参数shape是一个元组，指定array的形状，填充0
>>> np.zeros((2,))
array([0., 0.])
>>> np.zeros((2, 3, 2))
array([[[0., 0.],
        [0., 0.],
        [0., 0.]],

       [[0., 0.],
        [0., 0.],
        [0., 0.]]])
# np.ones(shape: Tuple) # 生成多维array, shape是元组，指定array的形状，填充1
>>> np.ones((2, 3, 2))
array([[[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]]])
# np.random.random_sample(shape: Tuple) # 生成多维array，填充[0, 1]的浮点
>>> np.random.random_sample((2, 3, 2))
array([[[0.58886443, 0.00125135],
        [0.55973185, 0.25874808],
        [0.03675558, 0.7065201 ]],

       [[0.72083918, 0.13066142],
        [0.50732886, 0.37403505],
        [0.16216774, 0.07918865]]])
```



#### copy

当需要将一个array中的数据赋给一个新的变量的时候，需要显式调用numpy.copy，执行深拷贝，否则新变量指向的是原始array的数据。

给出一个**错误的例子**

```py
>>> a = np.array([1, 2, 3, 4, 5, 6])
>>> b = a[:2]
>>> b += 1
>>> print('a =', a, '; b =', b)
a = [2 3 3 4 5 6] ; b = [2 3]
```

在这个例子中，没有执行深拷贝，因此b指向a的原始数据，当执行b+=1的时候，会修改a中的值。或者说，这里创建的b是a的一个视图(view)，视图相当于对原始数据的一个子集的索引。

**正确的写法**

```py
>>> a = np.array([1, 2, 3, 4])
>>> b = a[:2].copy()
>>> b += 1
>>> print('a = ', a, 'b = ', b)
a =  [1 2 3 4] b =  [2 3]
```



## operation

### Index 索引

使用`[]`索引array中的元素，索引从0到n-1，索引不可以越界

```py
>>> a = np.arange(10)
>>> print(a[2])
2
```



### Slicing 切片

使用`(start = 0: stop = n-1: step = 1)`获取数组的一个子集，注意`[start, stop)`左包含右不包含

```py
>>> a = np.arange(10)
>>> c = a[2: 7: 2]; print(c)
[2 4 6]
>>> c = a[3:]; print(c)
[3 4 5 6 7 8 9]
>>> c = a[:-1]; print(c)
[0 1 2 3 4 5 6 7 8]
>>> c = a[:]; print(c)
[0 1 2 3 4 5 6 7 8 9]
```

同样可以对n维数组进行切片，多个`(start = 0: stop = n-1: step = 1)`之间用`,`隔开。

```py
>>> a = np.arange(20).reshape(-1, 10)
>>> print(a)
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]]
>>> #access 5 consecutive elements (start:stop:step)
... print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")
a[0, 2:7:1] =  [2 3 4 5 6] ,  a[0, 2:7:1].shape = (5,) a 1-D array
>>> #access 5 consecutive elements (start:stop:step) in two rows
... print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")
a[:, 2:7:1] = 
 [[ 2  3  4  5  6]
 [12 13 14 15 16]] ,  a[:, 2:7:1].shape = (2, 5) a 2-D array
>>> # access all elements
... print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)
a[:,:] = 
 [[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]] ,  a[:,:].shape = (2, 10)
>>> # access all elements in one row (very common usage)
... print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
a[1,:] =  [10 11 12 13 14 15 16 17 18 19] ,  a[1,:].shape = (10,) a 1-D array
>>> # same as
... print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
a[1]   =  [10 11 12 13 14 15 16 17 18 19] ,  a[1].shape   = (10,) a 1-D array
```



### Single vector

NumPy支持对向量执行sum()，mean()等操作，包括对向量中的元素执行逐元素的操作（例如倍乘，或者说数乘向量）

```py
>>> a = np.arange(4)
>>> print(-a)
[ 0 -1 -2 -3]
>>> print(np.sum(a))
6
>>> print(np.mean(a))
1.5
>>> print(a**2)
[0 1 4 9]
>>> print(a * 5)
[ 0  5 10 15]
```



### 向量和差

NumPy支持两个等shape向量求和差，如果不同shape的话，会尝试进行向量**广播（Boardcast）**，广播可能导致不可预知的后果。

```py
>>> a = np.arange(4)
>>> b = np.arange(5, 9)
>>> print(a, b, a + b)
[0 1 2 3] [5 6 7 8] [ 5  7  9 11]
```

```py
>>> a = np.arange(4)
>>> b = np.ones((2, 4))
>>> print(a); print(b)
[0 1 2 3]
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]]
>>> print(a + b)
[[1. 2. 3. 4.]
 [1. 2. 3. 4.]]
```



### dot

NumPy的`.dot()`在底层会通过并行计算加速运算，所以不要自己写个loop代替。

```py
>>> a = np.array([1, 2, 3, 4])
>>> b = np.array([-1, 4, 3, 2])
>>> c = np.dot(a, b); print(c)
24
>>> c = np.dot(b, a); print(c)
24
```

dot的结果是一个标量（数），同时，正如你在线性代数课程上所学到的，dot满足交换律，具体可以参考3b1b《线性代数的本质》



### reshape

将ndarray的shape修改，不修改内部元素。

```py
numpy.reshape(a, newshape, order={'C', 'F', 'A'})
'''
	parameters: 
		a: array_like # Array to be reshaped
		newshape: int or tuple of ints # new shape should be compatible with the original shape. 
									# one shape dimension can be -1, so the value will be inferred
		order: {'C','F','A'} # different index order
						   # 'C' means using C-like order
						   # 'F' means using Fortran-like order
						   # 'A' means auto choice
	returns
		reshaped_array: ndarray
'''
```

```py
>>> a = np.arange(6).reshape((-1,2))
>>> print(f'{a.shape}, \n{a}')
(3, 2), 
[[0 1]
 [2 3]
 [4 5]]
```





# Gradient Descent for Multiple Regression

类似单元线性回归，多元线性回归需要迭代更新$w_i$和$b$，从而最小化损失函数

![image-20230613112418935](D:\CS\Machine Learning\5-Multiple Features.assets\image-20230613112418935.png)



## Gradient & Hessian Matrix

梯度的严格定义是多元函数的一阶导数。设$n$元函数$f(x)$对自变量$x = (x_1, x_2, ..., x_n)^T$各分量$x_i$的偏导数$\frac {\part f(x)}{\part x_i}$都存在，称函数$f(x)$对$x$一阶可导，向量
$$
\nabla f(x) = \left[\begin{array}{}
\frac {\part f(x)}{\part x_1}\\
\frac {\part f(x)}{\part x_2}\\
\vdots\\
\frac {\part f(x)}{\part x_n}
\end{array}\right]
$$
为$f(x)$对x的一阶导数或梯度。



多元函数的二阶导数称为Hessian矩阵。设$n$元函数$f(x)$对自变量$x = (x_1, x_2, ..., x_n)^T$各分量$x_i$的二阶偏导数$\frac {\part^2 f(x)}{\part x_i \part x_j}$都存在，称函数$f(x)$对$x$二阶可导，矩阵
$$
\nabla f(x) = \left[\begin{array}{}
\frac {\part^2 f(x)}{\part x_1^2} & \frac {\part^2 f(x)}{\part x_1 \part x_2} & \dots & \frac {\part^2 f(x)}{\part x_1 \part x_n} \\
\frac {\part^2 f(x)}{\part x_2 \part x_1} & \frac {\part^2 f(x)}{\part x_2^2} & \dots & \frac {\part^2 f(x)}{\part x_2 \part x_n} \\
\vdots & \vdots & \ddots & \vdots\\
\frac {\part^2 f(x)}{\part x_n \part x_1} & \frac {\part^2 f(x)}{\part x_n \part x_2} & \dots & \frac {\part^2 f(x)}{\part x_n^2} 
\end{array}\right]
$$
为$f(x)$对x的二阶导数或Hessian矩阵。



