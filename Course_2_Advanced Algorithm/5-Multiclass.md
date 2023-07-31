# Multi Class Classification

在二分类任务中，y只可能取两个值，通常表示为0/1。

而在多分类任务中，y可能取不止两个离散值，例如预测所有的手写数字，则y可以取`0123456789`，就是一个多分类任务。一个多分类任务的数据图示可能类似于下图

![image-20230717164857697](D:\CS\Machine Learning\Course_2_Advanced Algorithm\5-Multiclass.assets\image-20230717164857697.png)





# Softmax

## Equation

Softmax算法是Logistic算法的推广（generalization），在logistic regression中，我们使用表达式
$$
Sigmoid(z) = \frac 1 {1 + e^{-z}}
$$


而对于softmax regression，其表达式为
$$
Softmax(\vec z)_j = \frac {e^{z_j}}{\sum_{k = 1}^N e^{z_k}}
$$


softmax实际所做的事情是将包含k个实数的向量，转换为同样包含k个实数，但实数**和为1**的向量。

> 与sigmoid不同，softmax计算的对象是向量（vector），而sigmoid计算的对象是标量（scalar）
>
> 当只有两个分类器的时候，设输入向量为$[z, 0]$，则有
> $$
> Softmax([z, 0])_1 = \frac {e^z} {e^z + e^0} =  \frac {e^z} {e^z + 1}
> $$
> 上下同除$e^z$，得
> $$
> Softmax([z, 0])_1 = \frac 1 {1 + e^{-z}}
> $$
> 

输入可以是+/-/0，**输出范围[0, 1]**。当输入很小或者为负的时候，输出逼近0；当输入很大的时候，输出逼近1；综合这两个特性，可以将输出解释为概率。

使用Softmax作为分类器的一个前提是类别之间互斥（mutually exclusive）。如果存在某些样例可以同时是多个类别的成员，则不可以使用Softmax，而应该使用多个logistic（拆分成多个二分类）。



使用NumPy表示：

```py
def softmax(z):
	'''
	convert vector values to probability
	Args:
		z (ndarray (N,)) : inputs, N features from linear function
	Returns:
		a (ndarray (N,)) : softmax of z
	'''
    ez = np.exp(z)
    a = ez / ez.sum()
    return a
```



在softmax中，指数会对最大值产生放大效应，这也是为什么它叫“softmax"，我们可以从一下几个示例中观察到这个性质

<img src="D:\CS\Machine Learning\Course_2_Advanced Algorithm\5-Multiclass.assets\softmax (1).png" style="zoom:50%;" />

<img src="D:\CS\Machine Learning\Course_2_Advanced Algorithm\5-Multiclass.assets\softmax (2).png" style="zoom:50%;" />

<img src="D:\CS\Machine Learning\Course_2_Advanced Algorithm\5-Multiclass.assets\softmax (3).png" style="zoom:50%;" />



## Loss  & Cost

在前一篇我们讨论了交叉熵法。复习一下，对于对率回归，其loss为
$$
loss = -y\log f(x) - (1-y) \log (1 - f(x))
$$
cost取所有loss的均值，等价于求交叉熵
$$
cost = -P(y = 1) \log f(x) - [1-P(y = 1)\log[1-f(x)]
$$


同理可知，Softmax函数的cost为
$$
cost = \mathrm{E}_{x \sim P}(-\log f(x))
$$
其中，P是观测数据的分布，f是模型函数。



可推得loss
$$
a_j = \frac {e^{z_j}}{\sum_{k = 1}^N e^{z_k}} = P(y = j)
\newline
loss = -\log a_j
$$
![image-20230718165417888](D:\CS\Machine Learning\Course_2_Advanced Algorithm\5-Multiclass.assets\image-20230718165417888.png)

从图像可以看出，当$a_j$接近1的时候，loss下降减慢，当$a_j$逼近0的时候，loss下降加快。



## Neural Network with Softmax output

还是识别手写体的例子，假设要识别手写体0~9，则输出层应该有10个units，并选用Softmax作为激活函数。

> 因为手写体的数字都是互斥的，所以可以用Softmax

![image-20230718170010986](D:\CS\Machine Learning\Course_2_Advanced Algorithm\5-Multiclass.assets\image-20230718170010986.png)

> Softmax的输入是整个向量，计算单个结果需要依赖所有的向量元素。



代码

```py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.deras.layers import Dense
# 设置模型
model = Sequential([
    Dense(units = 25, activation = 'relu'),
    Dense(units = 15, activation = 'relu'),
    Dense(units = 10, activation = 'softmax')
])
# 指定loss
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(loss = SparseCategoricalCrossentropy)
# 训练
model.fit(X, Y, epochs = 100)
```

> `SparseCategoricalCrossentropy`是label和prediction之间的交叉熵，也就是前面的公式

这部分代码可以运行，但不是效率最好的，只是为了整理前面所学的知识，后面会给出更推荐的写法。



# Improved Implements

## Round out Errors

设以下两个等价的运算
$$
x = \frac 2 {10000}
\newline
x = (1 + \frac 1 {10000}) - (1 - \frac 1 {10000})
$$


如果在计算机中运行的话，可能出现舍入错误

```py
>>> x = 2.0 / 10000.0
>>> print(f'{x:.18f}')
0.000200000000000000
>>> x = 1 + 1.0/10000.0 - (1 - 1.0/10000.0)
>>> print(f'{x:.18f}')
0.000199999999999978
```



在logistic和softmax算法中，同样存在这种舍入问题。我们需要选择更为合理的计算方法，来减少舍入误差。



## More numerically accurate implement of logistic loss

首先讨论如何更精确计算logistic loss，只需要将输出层的激活函数改为linear，并设置loss的from_logit属性为True即可

```py
model = Sequential([
    Dense(units = 25, activation = 'relu'),
    Dense(units = 15, activation = 'relu'),
    # Dense(units = 10, activation = 'sigmoid')
    Dense(units = 10, activation = 'linear') # 这里改用linear
])

# model.compile(loss = BinaryCrossEntropy())
model.compile(loss = BinaryCrossEntropy(from_logits = True))
```



logit是logistic函数的反函数，因此数学定义为
$$
\mathrm{logit}\ p = \ln \frac p {1 - p}
$$
因此也被称为对数几率（log-odds）。应用`from_logit`意味着传递logit而不是probability，在logistic regression中，这个值就是$z$。TensorFlow源码中也推荐使用`from_logits = True`。

当设置`from_logits = false`的时候，首先会计算`f = \frac 1 {1 + e{-x*w + b}}`，再将这个值带入cost表达式；而当设置`from_logits = True`的时候，会直接将f先带入cost表达式，再进行计算，这样就能减少舍入，并且让tensorflow重新排列各项，提高运算的精度。

这种运算方式在logistic regression中提升不明显，但在softmax regression中提升非常显著



## More numerically accurate implement of softmax

softmax的loss表达式为
$$
loss = -\log \frac {e^{z_j}}{\sum_{k = 1}^N e^{z_k}}
$$
直觉上，可能存在某些极端大或极端小的$e^{z_j}$，使得计算的精度变差；而通过设置`from_logits = True`，TensorFlow可以重新排列部分表达式，避免极端值的出现。



代码

```py
model = Sequential([
    Dense(units = 25, activation = 'relu'),
    Dense(units = 15, activation = 'relu'),
    Dense(units = 10, activation = 'linear')
])

model.compile(loss = SparseCategoricalCrossEntropy(from_logits = True))
```



当然，因为输出层变成了`linear`，predict的输出会变成`z`而不是概率，需要再用对应的函数求一次概率

```py
# in logistic regression
predict = tf.nn.sigmoid(model(X))

# in softmax
predict = tf.nn.softmax(model(X))
```



# Multi Label Classification 

Multi Class 和 Multi Label是经常容易混淆的两个概念。在前面我们讨论了Multi Class Classification，输出label可能是多个label中的一个（**多个中的一个**）；而对于Multi Label Classification问题，同一个输入可能**对应着多个label**。

例如，在道路目标检测的问题中，同一张图像中可能既有`car`，也有`person`，也有`bus`。如下面一组图所示。

![image-20230718212640606](D:\CS\Machine Learning\Course_2_Advanced Algorithm\5-Multiclass.assets\image-20230718212640606.png)



解决这类问题，需要输出层输出多个输出结果，例如可以用一个0/1数组，其中的每一项都表示场景中是否存在对应的label

![image-20230718213334842](D:\CS\Machine Learning\Course_2_Advanced Algorithm\5-Multiclass.assets\image-20230718213334842.png)



# Adaptive Moment estimation

简称Adam算法，该算法能够自动调整学习率$\alpha$的大小，并作出调整

- 当学习率较低，持续向同一个方向下降时，算法会增大学习率
- 当学习率过高，发生振荡的时候，算法会减小学习率

![image-20230719143013799](D:\CS\Machine Learning\Course_2_Advanced Algorithm\5-Multiclass.assets\image-20230719143013799.png)



使用示例：在MNIST模型上应用

```py
# model
model = Sequential([
    tf.keras.layers.Dense(25, activation = 'sigmoid'),
    tf.keras.layers.Dense(15, activation = 'sigmoid'),
    tf.keras.layers.Dense(10, activation = 'linear'),
])
# compile
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
             loss = tf.keras.SparseCategoricalCrossentropy(from_logits = True))
# fit
model.fit(X, Y, epochs = 100)
```

> Adam需要提供一个初始学习率，可以像之前梯度下降中所做的那样，尝试不同数量级的学习率，选择训练速度最快的那个。
>
> 为了细微调节，不同层的学习率会有不同。
>
> Adam通常是比较推荐的优化算法。



# Additional Layer Types

到现在为止，我们使用的都是`Dense`类型的层，在这里讨论其他的常用层类型。



## Convolutional Layer

卷积层的每个神经元只能看到前一个层输出数据的一部分，这样做有两个优势

- 计算更快
- 不易过拟合，需要的训练数据更少



使用至少一层卷积层的神经网络称为Convolutional Neural Network（CNN，卷积神经网络），以心电图信号分类为例子进行说明，如图所示，该神经网络处理心电图的时间序列，包含两个卷积层。

![image-20230719152053949](D:\CS\Machine Learning\Course_2_Advanced Algorithm\5-Multiclass.assets\image-20230719152053949.png)

机器学习这门课不会过多深入CNN，知道layer不只有`Dense`就可以了。



# 参考

- 吴恩达《机器学习2022》
- 西瓜书
- [Softmax Function](https://deepai.org/machine-learning-glossary-and-terms/softmax-layer)
