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

