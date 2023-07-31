# TensorFlow implements

使用此前讨论过的预测手写体0/1的例子，首先给出训练网络的代码，然后分别分析每个部分地作用

![image-20230717084748341](D:\CS\Machine Learning\Course_2_Advanced Algorithm\4-Training a Neural Network in TensorFlow.assets\image-20230717084748341.png)

```py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# 1: sequence model
model = Sequential([
	Dense(units = 25, activation = 'sigmoid'),
	Dense(units = 15, activation = 'sigmoid'),
	Dense(units = 1, activation = 'sigmoid')
])
# 2: compile model
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss = BinaryCrossentropy())
# 3: training the model
model.fit(X, Y, epochs = 100)
```



回忆一下此前训练模型的步骤，可以划分为下面几步：

1. 确定如何根据输入`x`计算输出。在神经网络中，这一步对应inference。
2. 确定loss和cost，loss是单个样例的差异，cost是整个样本的差异。上面的代码使用了binary cross entropy作为loss；如果需要训练linear regression，可以使用`MeanSquareError()`
3. 训练模型，最小化cost。我们曾在线性回归和对率回归讨论了如何使用梯度下降做到这一点。TensorFlow在调用`.fit`的时候会进行反向传播（back propagation）计算每部分的导数项。



# Entropy & Binary Cross Entropy

在此前关于logistic regression的讨论中，我们使用了下面的loss函数：
$$
Loss(f_{\vec w, b}(\vec x_i), y_i) = -y_i \log(f_{\vec w, b}(\vec x_i)) - (1 - y_i) \log(1 - f_{\vec w, b}(\vec x_i))
$$
在统计学上，这个函数被称为binary cross entropy，二分类交叉熵；TensorFlow同样采用了这个术语，也即代码中的`BinaryCrossentropy()`。

> 讨论logistic regression的时候，这个loss是用MLE（极大似然法）推导的，这里讨论下交叉熵法（Cross entropy, CE）。



## Entropy

在信息论中，熵（entropy）的含义是**无损编码事件的最小平均编码长度**，我们曾经在数据结构中学习过霍夫曼编码，通过对出现频率高的字符使用短编码，对出现频率低的字符使用长编码，霍夫曼编码可以设计出长度最短的二进制前缀码，$\frac {字符串码长}{字符数}$就是最小平均编码长度。我们可以假定字符（事件）的出现服从某种概率分布，在某一样本中，事件出现的概率可以表述为：
$$
P = \frac {该事件发生的次数} {总试验数}
$$


首先讨论一种简单的情况，设有N种等可能的事件，则每种事件发生的概率为$P = \frac 1 N$，此时编码该信息的最小长度可以表述为
$$
\log_2N = -\log_2\frac 1 N = -\log_2P = -\sum P\log_2P
$$

> 底数取2是因为要进行二进制编码



推广到可能性不均一的情况，以及连续变量的熵（单位为bits）
$$
Entropy = -\sum_i P(i) \log_2P(i)
\newline
Entropy = -\int P(x) \log_2P(x) dx
$$

> 从直观上理解，熵反映了信息中可能状态的多少。若熵比较大，则随机变量的不确定性较强，难以对新信息进行预测。
>
> 以二项分布为例，设$P(x = 1) = p,\ P(x = 0) = 1-p$，则熵为
> $$
> H(p) = -p \log_2 p - (1-p)\log_2(1-p)
> $$
> 其曲线为
>
> ![image-20230717103405092](D:\CS\Machine Learning\Course_2_Advanced Algorithm\4-Training a Neural Network in TensorFlow.assets\image-20230717103405092.png)
>
> 可以观察到，当熵最大的时候，p = 0.5，随机变量的不确定性最大。



熵公式同样可以理解为，对一个随机变量，计算其$-\log P(x)$的期望，因此熵公式又可以简写为：
$$
H(P) = Entropy = \mathrm{E}_{x\sim P}(-\log P(x))
$$
$x\sim P$代表使用P计算期望，从式中可知，熵只依赖于分布，而与x的取值无关。



## Cross-Entropy

前面讨论到，如果我们已知某事件服从概率分布P，就可以计算该事件的熵。然而，在实际场景中，通常无法得知真实分布P，此时就需要对熵做一个估计。

首先对熵公式中的不同部分进行划分，在公式
$$
Entropy = \mathrm{E}_{x\sim P}(-\log P(x))
$$
中，$-\log P(x)$指定了编码的方式，而$E_{x \sim }$则指定了被编码的变量的分布。



设在观测之前，预估的概率分布为Q，观测后得到的概率分布为P，则交叉熵
$$
Cross\ Entropy = \mathrm{E}_{x \sim P}(-\log Q(x))
$$
记$H(P, Q)$，表示**使用基于Q的编码，对来自P的变量编码所需要的字节数**。该值可以用于衡量预估分布Q是否能够反应真实的分布P。

根据熵和交叉熵的定义，易知$H(P, Q) \ge H(P)$，因为$H(P)$是理论最小值，所以$H(P, Q)$只可能大于等于$H(P)$。而交叉熵越接近理论最小值，预估分布Q就越接近真实分布P，这也是为什么可以使用交叉熵作为损失函数的原因。



在二分类任务中（或者说P服从二项分布的情况下），有
$$
H(P, Q) = -P \log_2 Q - (1-P)\log_2(1-Q)
$$
称为binary cross entropy。

> 我们此前讨论过，logistic regression的loss可以写为
> $$
> loss = -y\log f(x) - (1-y) \log (1 - f(x))
> $$
> 而cost则是取所有loss的均值，等价于这里的交叉熵。

# Alternatives to the sigmoid activation

在此前的神经网络分类任务中，我们使用的激活函数都是`sigmoid`，该函数假定输出结果都是二元的。例如在预测产品是否会畅销的例子中，我们假设隐层属性为`[affordability, awareness, perceived quality]`，则该层的输出为消费者是否有能力购买、是否已经被消费者了解、以及消费者是否认为产品质量好；这些都是二元的属性。

而在实际的应用场景中，这些属性可能并非二元的，取值范围也可能并非`[0,1]`，选择不同的激活函数，能够改善模型的性能。



Sigmoid之外，一种常见的激活函数是ReLU（Rectified linear unit），该函数的数学表示为$g(z) = max(0,z)$，图像为

![image-20230717114522336](D:\CS\Machine Learning\Course_2_Advanced Algorithm\4-Training a Neural Network in TensorFlow.assets\image-20230717114522336.png)



在前面的讨论中，也使用过linear，有时会会有人说“没有使用激活函数”，也是指的用linear

![image-20230717114900790](D:\CS\Machine Learning\Course_2_Advanced Algorithm\4-Training a Neural Network in TensorFlow.assets\image-20230717114900790.png)



# Choosing activation functions

## Output layer

输出层的激活函数可以根据输出的target决定。例如

- 二分类问题，$y = 0/1$，使用sigmoid
- 回归问题，y可以是正值或负值，使用linear
- 回归问题，y必须是正值（例如房价），使用ReLU



## Hidden layer

**ReLU**是当下的常见选择，最早期很多人用sigmoid。

原因有很多，首先ReLU的数学计算很快，$g(z) = max(0, z)$比sigmoid的$\frac 1 {1 + e^{-z}}$快得多；其次，从ReLU的函数图像来看，ReLU只会在左半的部分扁平，这使得导数值更大（因为导数反应斜率），梯度下降速度更快。

> ReLU还有个优势是稀疏激活（sparsely activated），因为对所有的负值结果，输出都为0。从而可以避免计算无关部分。



有时文献里还能看到其他的激活函数，例如`tanh`, `LeckyReLU`等，在某些情况下，这些激活函数效果更好，不过对大多数神经网络来说，使用ReLU就够用了。

> LeckyReLU和ReLU不同的地方在于负值区，ReLU的负值区直接归0了，而LeckyReLU还保持一个非常小的斜率，这样可以避免某些情况西ReLU归0导致神经元”死亡“。



# Why do we need activation functions

假设不使用激活函数（全用linear），神经网络的结果和线性回归不会有什么不同。

用一个简单的模型来说明，在下面这个模型中，所有节点的激活函数都是linear

![image-20230717163544135](D:\CS\Machine Learning\Course_2_Advanced Algorithm\4-Training a Neural Network in TensorFlow.assets\image-20230717163544135.png)

此时有
$$
a^{[1]} = w_1^{[1]}x + b_1^{[1]}
\newline
a^{[2]} = w_1^{[2]} a^{[1]} + b_1^{[2]}
$$
将$a^{[1]}$带入第二个式子，则
$$
a^{[2]} = w_1^{[2]} (w_1^{[1]}x + b_1^{[1]}) + b_1^{[2]} 
\newline
= w_1^{[2]}w_1^{[1]}x + w_1^{[2]}b_1^{[1]} + b_1^{[2]} 
$$
实质上等价于一个新的线性函数。从线性代数的角度上来看，相当于对线性变换进行组合，结果还是一个线性变换。

因此，对于多层的神经网络，隐层不要使用linear激活函数，这样做只是对结果做了个线性变换，等价于什么都没做。



# Back Propagation

Back Propagation（反向传播，误差逆传播，简写BP）是最常用的神经网络学习算法。这里讨论下如何直观理解反向传播。



## Derivative

首先举一个非常简单的例子来直观理解导数，设cost函数
$$
J(w) = w^2
$$
若$w = 3$，则$J(w) = 9$，假定$w$变化一个极小值，$\varepsilon = 0.001/0.002$，则
$$
\Delta w = 0.001/0.002\newline
\Delta J(w) = 0.006001/0.012004
$$
可以观察到$\Delta J(w) \approx 6 \Delta w$，随着$\varepsilon$的减小，这个结果会更加精确。

根据已有微积分知识也可以推得，$\frac d {dw} w^2 = 2w$，因为$w = 3$，所以导数为6。导数反映了$J(w)$随$w$的变化的比率。



在进行梯度下降的时候，我们使用下面的方法更新$w_j$
$$
w_j = w_j - \alpha \frac {\partial} {\partial w_j}J(\vec w, b)
$$
当导数很大的时候，意味着$w$的微小变化都会对$J$的产生很大的影响，这种时候就需要调整$w$，使其不会对$J$产生很大影响；反之，若导数很小，则$w$对$J$影响不大，也不需要对$w$做出太多的改变。



从函数图像上来看，导数反映了曲线的斜率

![image-20230719160408259](D:\CS\Machine Learning\Course_2_Advanced Algorithm\4-Training a Neural Network in TensorFlow.assets\image-20230719160408259.png)



## SymPy

SymPy是一个符号数学库（symbolic mathematical library），符号计算的意思是，在默认情况下不会计算非精确数值，而是会传递symbol对象，这样一方面可以防止精度损失，另一方面可以简化某些运算。

举例来说，计算$\sqrt 8$

```py
>>> import math
>>> import sympy
>>> math.sqrt(8)
2.82842712475
>>> sympy.sqrt(8)
2*sqrt(2)
```



使用SymPy计算导数

```py
>>> import sympy
# 设置symbol变量
>>> J, w = sympy.symbols('J, w')
>>> J = w**2
>>> J
𝑤2
# 求导数
>>> dJ_dw = sympy.diff(J, w)
>>> dJ_dw
2𝑤
# 带入值计算导数
>>> dJ_dw.subs([(w, 3)])
6
```



## Computation Graph & Prop.

神经网络是一种特殊形式的计算图。计算图是有向图，其节点对应**变量（variables）**或者**操作（operations）**，变量的值被传入操作，操作的输出值可以传入新的操作。通过这种方式，计算图可以表示特定的运算，例如下面的计算图就表示了仿射变换（affine transformation）

![image-20230719165812988](D:\CS\Machine Learning\Course_2_Advanced Algorithm\4-Training a Neural Network in TensorFlow.assets\image-20230719165812988.png)



首先从一个简单的线性回归例子开始，学习如何将运算展开为计算图，如下图所示，对单个线性神经元的神经网络，其cost函数为$J(w, b) = \frac 1 2 (wx + b - y)^2$，将每一步运算展开

![image-20230719170815116](D:\CS\Machine Learning\Course_2_Advanced Algorithm\4-Training a Neural Network in TensorFlow.assets\image-20230719170815116.png)

![image-20230719171109111](D:\CS\Machine Learning\Course_2_Advanced Algorithm\4-Training a Neural Network in TensorFlow.assets\image-20230719171109111.png)

这张计算图对应着神经网络的前向传播（forward prop.），进一步，我们可以利用这张计算图，通过反向传播（back prop.）推导导数



首先计算$d$与$J$的关系，因为
$$
\frac {\partial J}{\partial d} = d
$$
因此，如果$d = d + \varepsilon$，则$J = J + d * \varepsilon$。在上图中，因为$d = 2$，因此导数值为2



进一步推算$\frac {\partial J}{\partial a}$
$$
\begin{array}{l}
\frac {\partial d}{\partial a} = 1
\newline\to
\frac {\partial J}{\partial a} 
= \frac {\partial J}{\partial d} \frac {\partial d}{\partial a}
= d = 2
\end{array}
$$


同理，推导$\frac {\partial J}{\partial b}, \frac {\partial J}{\partial c}$
$$
\begin{array}{l}
\frac {\partial a}{\partial b} = 1
\newline
\to \frac {\partial J}{\partial b} 
= \frac {\partial J}{\partial d} \frac {\partial d}{\partial a} \frac {\partial a}{\partial b} 
= b = 2
\newline
\frac {\partial a}{\partial c} = 1
\newline
\to \frac {\partial J}{\partial c} 
= \frac {\partial J}{\partial d} \frac {\partial d}{\partial a} \frac {\partial a}{\partial c}
= b = 2
\end{array}
$$
最后，推导$\frac {\partial J}{\partial w}$
$$
\begin{array}{l}
\frac {\partial c}{\partial w} = x
\newline\to
\frac {\partial J}{\partial a} 
= \frac {\partial J}{\partial d} \frac {\partial d}{\partial a} \frac {\partial a}{\partial c} \frac {\partial c}{\partial w}
= d * x = -4
\end{array}
$$


![image-20230719175805121](D:\CS\Machine Learning\Course_2_Advanced Algorithm\4-Training a Neural Network in TensorFlow.assets\image-20230719175805121.png)



使用反向传播计算导数最大的优势在于能够重复利用已经计算过的导数值（例如上面的$\frac {\partial J}{\partial a}$），使得计算更加快速。

对于具有N个节点，P个参数的计算图，前向计算导数的时间复杂度为`O(N * P)`，而反向计算导数的时间复杂度为`O(N + P)`。



## Larger Network

给出一个两层，每层一个神经元的神经网络，进一步深入理解计算图和反向传播

首先根据前向传播绘制出计算图

![image-20230719182254952](D:\CS\Machine Learning\Course_2_Advanced Algorithm\4-Training a Neural Network in TensorFlow.assets\image-20230719182254952.png)

写出每部分的导数
$$
\begin{array}{c}
\begin{array}{l}
\frac {\partial z_1}{\partial b_1} = 1\\
\frac {\partial z_1}{\partial w_1} = x
\end{array} &
\begin{array}{l}
\frac {\partial z_2}{\partial b_2} = 1\\
\frac {\partial z_2}{\partial w_2} = z_1\\
\frac {\partial z_2}{\partial z_1} = w_2
\end{array} &
\begin{array}{l}
\frac {\partial J}{\partial z_2} = z_2 - y
\end{array}
\end{array}
$$
链式求导
$$
\begin{array}{l}
\frac {\partial J}{\partial b_2} = z_2 - y\\
\frac {\partial J}{\partial w_2} = z_1(z_2 - y)\\
\frac {\partial J}{\partial b_1} = w_2(z_2 - y)\\
\frac {\partial J}{\partial w_1} = w_2(z_2 - y)x
\end{array}
$$



## Accumulated error BP

上述反向传播的方法可以用下面的伪码表示
$$
\underline{
\over {
\begin{array}{l}
输入:训练集D = \{(x_i, y_i)\}_{i = 1}^m;\\
\quad\quad\quad学习率\alpha.\\
过程:\\
1. 随机初始化权重，值域(0, 1)\\
2. \mathbf{repeat}\\
\quad\quad \mathbf{for\ all\ (x_k, y_k)\ \in D\ do}\\
\quad\quad\quad 计算梯度;\\
\quad\quad\quad 更新权重;\\
\quad\quad \mathbf{end\ for}\\
3. \mathbf{until}\ 停止条件\\
输出: 确定权重的神经网络
\end{array}
}
}
$$


该算法为”标准BP算法"，每次只针对一个样例更新权重，而BP算法的目标是**最小化**整个训练集$D$上的**累计误差**：
$$
E = \frac 1 m \sum_{i = 1}^m E_i
$$


因此可以在读取整个训练集$D$之后，才对参数进行更新，这种方法称为Accumulated BP（累计误差逆传播）。相比于标准BP在读取每个样例之后都会更新权重，累计BP更新频率更低，且因为直接更新累计误差，通常比标准BP需要的迭代次数更少；然而累计BP也存在误差下降到一定程度后，下降变缓的问题，因此在$D$非常大的时候，选择标准BP能够更快获得较好的解。



# Overfitting

由于神经网络的表现能力非常强，很容易出现过拟合，解决神经网络过拟合通常有两种策略：

- early stopping。将数据划分为训练集和验证集，使用验证集在训练的时候估计误差，若训练集误差降低，验证集误差升高，则停止训练，验证集误差最小的权重
- regularization。在cost中引入正则项，正如前面笔记中讨论的一样。



# Local & Global Minimum

cost函数可能存在多个局部最小值，我们通常希望最优化算法能够寻找到全局的最小值。然而基于梯度的优化方法，在cost函数存在多个局部最小的时候，无法保证一定能找到全局最小。通常有以下策略用来解决这种问题：

- 用多组初始化参数训练多个网络，选择最优解
- 模拟退火（simulated annealing），以一定概率接受更差的结果，”跳出“局部最小；接受”次优解“的概率随迭代数升高而降低
- 随机梯度下降（SGD）。
- 遗传算法（genetic algorithms）。

不过这些算法都是启发式的，理论上缺乏保障。


# 参考

- 吴恩达《机器学习2022》
- 西瓜书
- [Deep Learning From Scratch: Theory and Implementation](https://www.codingame.com/playgrounds/9487/deep-learning-from-scratch---theory-and-implementation/computational-graphs)
- [Rectified Linear Units](https://deepai.org/machine-learning-glossary-and-terms/rectified-linear-units)
- [一文搞懂熵(Entropy),交叉熵(Cross-Entropy)](https://zhuanlan.zhihu.com/p/149186719)
