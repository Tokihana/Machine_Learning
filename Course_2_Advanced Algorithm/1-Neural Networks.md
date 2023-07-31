# Motivation

早期的神经网络的研究动机是为了模仿生物大脑的行为（mimic the brain）。不过当下的神经网络和大脑的学习方式的关联很小。

如下图所示，神经网络的数学模型是建立在当下人们对神经元的理解上的，我们可以将其抽象为一系列获取输入和输出的圆圈

![image-20230711142554230](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230711142554230.png)

> 当然，正如吴恩达老师所讲，脑科学的研究任重而道远，这种模型不一定能够构建出真实的原始智能。不过，这种模型已经可以创建出非常强大的算法，且由数学证明支撑。在学习深度学习的角度上，我们可以更多关注工程上的优化问题，考虑“人工神经网络（Artificial Neural Network）。

# Why now?

![image-20230711143817808](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230711143817808.png)

如上图所示，随着数据大小的不断增长，更大的神经网络往往能够获得更好地模型表现，这需要更快的计算机性能支持。



# Example: Demand Prediction & Terminology

我们可以通过一个预测畅销品（需求）的例子，来进一步深入。例如，判断一款衬衫是否为畅销品，输出为yes/no，这是一个典型的二分类任务，可以用logistic regression来解决；

![image-20230711144614703](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230711144614703.png)

如图，设输入$x = price$，输出$f(x) = \frac 1 {1 + e^{-(wx + b)}}$在神经网络中通常表述为$a$，即activation。这个模型可以被看作是一个神经元，输入衬衫的价格，输出该衬衫为畅销品的可能性。

> activation（激活）借用了神经学上的概念，当一个神经元激活、或者说电位超过某个threshold（阈值）的时候，该神经元就会向轴突相连的神经元传递物质（向下一层发送数据）；神经元处理输入并产生输出的函数也被称为activation function（激活函数）



接着，考虑一个更加复杂的例子，假设对于一款衬衫，有四项参考特征（feature）：price, shipping cost, marketing, material，分别对应衬衫的价格、运输费用、营销和材质。同时假设，衬衫的畅销程度与下面几个因素（factor）相关：

- affordability
- awareness
- perceived quality

分别代表在顾客角度，产品的性价比、关注度和感知质量（顾客对产品质量的主观认知）。我们可以试着构建一个神经网络：

![image-20230711150417623](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230711150417623.png)

在这个神经网络中，蓝色的三个神经元（neurons）组成一层（layer），品红色的神经元组成一层；层由具有相似输入特征的神经元构成，可以是多个或单个神经元。

> 也可以将神经网络理解为一系列嵌套的函数，每层函数接收上一层的输出作为本层的输入，并将输出传递给下一层。



在施加实现中，通常不会如上图所示的那样，手动指定每个神经元的连接关系，而是让神经元链接所有的上一层输入，并自适应地拟合这些feature（特征）的weight（权重），设$\vec x = [price, cost, marketing, material]$，则上面的神经网络可以简化
$$
\vec x(input\ layer ) \overset {hidden\ layer} \to \vec {a1} \overset {outpu\ layer} \to a2(Probability)
$$
中间的一层（上图中的蓝色圆圈层）之所以被称为hidden layer（隐层），是因为这一层不在训练集中，训练数据中不存在该层的真值。input layer（输入层）只接收输入，不对输入进行处理，隐层与输出层包含处理函数。上面这个神经网络只有两个包含处理函数的功能层，其中一个是隐层，因此也被称为**两层网络**或**单隐层网络**。也有文献将神经网络表述为perceptron（感知机），看到了要认得。

> 整理一下，神经网络的学习，就是根据训练数据，调整神经元之间的**连接权重**，以及调整神经元内部处理函数的**阈值**。
>
> 同时还需要注意的是，尽管在这个例子中，我们使用affordability, awareness和perceived quality描述了隐层信号，通常来说，神经网络应在训练时自行选择要使用的特征和处理后的信号。
>
> 网络层数以及每层应该包含多少个神经元，也是影响性能的重要因素。



# Example: Computer Vision

在这里讨论下神经网络在计算机视觉中如何发挥作用，着重建立直觉。

在计算机视觉中，输入通常为图像中的像素点展开成的向量，如下图所示，这是个面部识别的例子。

![image-20230712133705863](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230712133705863.png)

假设使用一个三隐层网络，并提取每个隐层中不同神经元的结果，可以得到类似下图的结果

![image-20230712134124359](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230712134124359.png)

在第一层中，神经网络试图寻找小的边缘；这些边缘在第二层被组合成脸部的不同部分；在第三层组合成完整的面部，并在输出层对比组合结果符合哪个面部。隐层中的特征都是神经网路自行学习的，不需要手动编码。

> 课件里的参考图是从[Convolutional Deep Belief Networks for Scalable Unsupervised Learning of Hierarchical Representations](../阅读材料/Convolutional Deep Belief Networks for Scalable Unsupervised Learning of Hierarchical Representations.pdf)摘出来的，参照其中Figure 2的说明，前一层中的每个basis（隐层中的权重）都会在后一层经过weighted linear combination（加权线性组合）。
>
> ![image-20230712140813275](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230712140813275.png)





# Activation Function

在简单的神经元模型中，神经元接收n个其他神经元传递的**加权**信号，并处理这些信号，通过激活函数（activation function）产生输出。

理想情况下，神经元应当输出0或1，分别对应神经元抑制和激活。则理想的激活函数应当为阶跃函数，但正如在对率回归中所讨论的那样，阶跃函数不连续不光滑，通常用Sigmoid函数代替，典型的代表是Logistic function。

![image-20230712142742322](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230712142742322.png)

将多个包含Sigmoid函数的这样的神经元组合起来，就组成了神经网络。



# Neural Network Layer

**层**是现代神经网络的基本构建单元。

仍然从需求预测的例子入手，观察每层的构建

![image-20230712151128651](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230712151128651.png)

如图所示，layer 1包含三个神经元，每个神经元都实现一个对率函数，处理$\vec x$，并将输出结果$\vec a^{[1]}$传递给layer 2



![image-20230712151418950](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230712151418950.png)

layer 2包含一个神经元，接收$\vec a^{[1]}$作为输入，获得输出结果$a^{[2]}$，该结果代表衬衫为畅销品的概率。也可以将这个概率舍入，得到0/1值。



# Complex Neural Network

讨论一个更加复杂的神经网络，尝试得到每层的activation value（激活值）的通式。

![image-20230712152804605](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230712152804605.png)

如图所示，以layer 3的计算为例，同时令$\vec x = \vec a^{[0]}$，可以推得通式
$$
a^{[i]}_j = g(\vec w^{[i]}_j \cdot \vec a^{[i -1]} + b^{[i]}_j)
$$


# Inference: making predictions using forward propagation

讨论使用神经网络进行推断的过程，设需要预测8x8格矩阵内的0/1，这是一个二分类问题

![image-20230712154202351](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230712154202351.png)



使用一个三层网络来完成这个任务，layer 1有25个神经元，layer 2有15个神经元

![image-20230712154258462](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230712154258462.png)

其中，
$$
\vec a^{[1]} = \left[
\begin{array}{}
g(\vec w^{[1]}_1 \cdot \vec a^{[0]} + \vec b^{[1]}_1) \\
\vdots \\
g(\vec w^{[1]}_{25} \cdot \vec a^{[0]} + \vec b^{[1]}_{25}) \\
\end{array}
\right]
\newline
\vec a^{[2]} = \left[
\begin{array}{}
g(\vec w^{[2]}_1 \cdot \vec a^{[1]} + \vec b^{[2]}_1) \\
\vdots \\
g(\vec w^{[2]}_{15} \cdot \vec a^{[1]} + \vec b^{[2]}_{15}) \\
\end{array}
\right]
\newline
a^{[3]} = g(\vec w^{[3]}_1 \cdot \vec a^{[2]} + \vec b^{[3]}_1)
$$
这个推理过程称为forward propagation（前向传播）。相对应的，神经网络的学习算法使用backward propagation，也就是后向传播。



# Inference in Code - With TensorFlow

同样的神经网络算法可以被用于多种应用上，再举一个烤咖啡豆的例子，假设可以调整烤豆的温度（Temperature）和时间（Duration）。如图所示，只有当温度和时长在合理的范围内的时候，才能得到好的咖啡豆。

![image-20230712160241284](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230712160241284.png)



假设使用下图的神经网络，给出新的数据[200.0, 17.0]（200度烤17分钟），进行推断

![image-20230712171624305](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230712171624305.png)



代码实现

```py
x = np.array([[200.0, 17.0]])
layer_1 = Dense(units = 3, activation = 'sigmoid')
a1 = layer_1(x)
layer_2 = Dense(units = 1, activation = 'sigmoid')
a2 = layer_2(a1)
yhat = np.round(a2) # 可选步骤，对概率进行舍入
```

> Dense是layer的一种，代表regular densely-connected NN layer，activation指定激活函数为sigmoid。tf还提供了很多其他选择，可以进一步参考手册。这两个API都是`tf.keras`提供的，Keras是TensorFlow中用于构建和训练模型的高级API，更多内容可以查看[指南](https://www.tensorflow.org/guide/keras)。
>
> 这部分的代码只提供了如何用模型进行预测，导入tf和加载权重在lab里会讲。



# Data in TensorFlow

TensorFlow和NumPy的数据格式可能有些地方不太一样，两者之间传递数据的时候记得做点转换。

假设存在一个2行3列（$2 \times3$）的矩阵
$$
\left[
\begin{array}{}
1 & 2 & 3 \\
4 & 5 & 6
\end{array}
\right]
$$
在NumPy中，下面这个的矩阵可以表示为

```py
X = np.array([[1, 2, 3],
			 [4, 5, 6]])
```



比较容易混淆的是$1 \times n，n \times 1$的矩阵和向量

```py
>>> import numpy as np
>>> X1 = np.array([[200, 70]]) # 1 x 2 矩阵
>>> X2 = np.array([[200],
...               [70]]) # 2 x 1 矩阵
>>> x = np.array([200, 70]) # 含有2个元素的向量
>>> print(X1.shape, X2.shape, x.shape)
(1, 2) (2, 1) (2,)
```



在TensorFlow中，通常使用**矩阵**作为数据的表示形式，因为tf的设计目标是处理大规模数据。

```py
>>> layer_1 = tf.keras.layers.Dense(units = 2, activation = 'sigmoid')
>>> a1 = layer_1(X1)
>>> print(a1)
tf.Tensor([[1. 1.]], shape=(1, 2), dtype=float32)
>>> a1.numpy()
array([[1., 1.]], dtype=float32)
```

`tf.Tensor`是TensorFlow的数据类型，是TensorFlow程序主要的操作和传递对象，包含两个属性：

- 数据类型dtype。Tensor中所有元素的数据类型相同
- shape。数组的形状。

`.numpy()`方法提供了将Tensor数组转换为NumPy数组的方法，很常用。



# Bulid Neural Network

在之前我们实现forward propagation的方法是

```py
x = np.array([[200.0, 17.0]])
layer_1 = Dense(units = 3, activation = 'sigmoid')
a1 = layer_1(x)
layer_2 = Dense(units = 1, activation = 'sigmoid')
a2 = layer_2(a1)
```



该方法需要手动在层间传递数据（即通过变量赋值的方式传递数据），TensorFlow提供了方法`Sequential([])`，可以不用手动在层间传递数据，该方法将提供的一系列层进行线性串联，并提供模型的训练和推理功能。

```py
layer_1 = Dense(units = 3, activation = 'sigmoid')
layer_2 = Dense(units = 1, activation = 'sigmoid')
model = Sequential([layer_1, layer_2])

'''或者写为
model = Sequential([Dense(units = 3, activation = 'sigmoid'), 
				  Dense(units = 1, activation = 'sigmoid')])
'''
# .add()方法用于添加新的层
model.add(layer_1)

model.compile(...) # 配置模型属性，还有个类似的方法叫.complie_from_config()
				 # compile()的细节后面讲。
model.fit(X, y) # 拟合模型
model.predict(x_new) # 预测新的值
```



# Forward Prop in single layer

通过自己实现前向传播，深入理解TensorFlow或者PyTorch框架中的代码逻辑，首先来实现烤咖啡豆例子中的网络

![image-20230713162057450](D:\CS\Machine Learning\Course_2_Advanced Algorithm\1-Neural Networks.assets\image-20230713162057450.png)

给出输入

```python
x = np.array([200, 17])
```

x是1D array



在layer 1中，计算$\vec a$，记`wi_j`表示$w^{[i]}_j$，则`a1_1`的计算可以表示为

```py
w1_1 = np.array([1, 2])
b1_1 = np.array([-1])
z1_1 = np.dot(w1_1, x) + b
a1_1 = sigmoid(z1_1)
```

同理可计算`a1_2, a1_3`，改写成线性代数计算

```py
w1 = np.array([1, -3, 5],
             [2, 4, -6])
b1 = np.array([-1, 1, 2])

a1 = sigmoid(np.dot(w1, x) + b1)
```



再计算a2

```
w2 = np.array([-7, 8])
b2 = np.array([3])

a2 = sigmoid(np.dot(w2, a1) + b2)
```

> 实际应用中根据w和a的组织形式，`np.dot`的顺序可能会发生变化，例如，设
>
> ```py
> w = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
> b = np.array( [-9.82, -9.28,  0.96] )
> a1 = np.array([
>     [200,13.9],
>     [200,17]]
> )
> ```
>
> w是(2, 3)，a1是(2, 2)，样例数`m = 2`，特征数`n = 2`，神经元数`j = 3`，最终结果a2应该为(2, 3)，即2个样例，每个样例的a2输出包含3个元素，此时矩阵是按行标准组织的，因此点乘要倒过来
>
> ```
> a2 = sigmoid(np.dot(a1, w) + b)
> ```

# More General Implement of Forward Prop

定义`dense()`函数

```py
def dense(a_in, W, b, g):
    '''
    Args:
    	a_in (ndarray (m, )): input from previous layer
    	W (ndarray (m, n)): para matrix in this layer
    	b (ndarray (n, )): translate para
    	g (function):
    Returns:
    	a_out (ndarray (n, )): output array to next layer
    '''
    return g(np.dot(a_in, W) + b)
```

则`sequential()`函数可以类似

```py
def sequential(x):
    a1 = dense(x, W1, b1，sigmoid)
    a2 = dense(a1, W2, b2, sigmoid)
    return dense(a2, W3, b3, sigmoid)
```



# General Vectorization

更标准些的向量化代码，这里将`X, W, B`都改写成矩阵形式

```py
X = np.array([[200, 17]]) # 1 x 2
W = np.array([[1, -3, 5],
             [-2, 4, -6]]) # 2 x 3
B = np.array([[-1, 1, 2]]) # 1 x 3

def dense(A_in, W, B, g):
    A_out = g(np.matmul(A_in, W) + B)
    return A_out
```

> 复习一下，`np.matmul()`或者运算符`@`是`np.dot()`的一种特殊形式，专门处理2D array乘法。
>
> 如果你之前看过3b1b，那么这里A_in在前，W在后的情况可能会让你有些迷惑，因为这里的向量和矩阵都是行标准，每个单独的向量`w, x, b`在这里都是一行；而3b1b讲课用的列标准。
>
> 如果还是觉得迷糊的话，可以从头开始顺一遍：在TensorFlow中，惯例上认为`X`每个样例中的每个样例`x`都是按行标准排列的，设`X`有m个样例，n个特征，则`X`为(m, n)矩阵；即
> $$
> X = \left[\begin{array}{}
> x_{11} & \cdots & x_{1n} \\
> \vdots & & \vdots \\
> x_{m1} & \cdots & x_{mn}
> \end{array}\right]
> $$
> 设处理`X`的层中有`j`个神经元，则该层的矩阵`W`可以表示为，该层接收`m`个输入，返回`j`个输出，则`W`应该是(n, j)矩阵
> $$
> W = \left[\begin{array}{}
> w_{11} & \cdots & w_{1j} \\
> \vdots & & \vdots \\
> w_{n1} & \cdots & w_{nj}
> \end{array}\right]
> $$
> 根据矩阵运算规则，应使用`X @ W`，结果为(m, j)矩阵，即`m`个样例，`j`个特征。



# NumPy BroadCast

说明一下广播的机制，广播在概念上相当于对需要广播的维度进行复制操作

在代码运行前，NumPy会自右向左检查两个数组的维度是否匹配：

- 维度相等
- 或任意一边维度为1

当维度不匹配的时候，报错`ValueError: operands could not be broadcast together`



而当满足条件的时候，结果的维度会是`max(array1.axis(i), array2.axis(2))`，相等的情况直接去，任意一边为1的情况取不为1的那一个。
