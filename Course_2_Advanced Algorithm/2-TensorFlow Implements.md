# Implement Linear Regression & Logistic Regression with TensorFlow

在前面我们讨论过，神经网络可以被视为许多个函数的嵌套集合，每个神经元（neuron）都包含一个特定的激活函数。这意味着，我们可以将简单的线性回归和对率回归模型，当作是只有一个神经元的神经网络。



## Linear Activation

创建线性层，只需要设置`activation`为`linear`即可。

```py
linear_layer = tf.keras.layers.Dense(units = 1, activation = 'linear'， )
```

但需要注意的是，此时则权重模型权重还未初始化（没有build），**需要使用`build(input_shape)`或者跑一轮数据确定`input_shape`，触发权重的初始化（Trigger the instantiation of the weights）**

```py
>>> X = np.array([[1, -3, 5],
...               [2, 4, -6]])
>>> linear_layer(X)
<tf.Tensor: shape=(2, 1), dtype=float32, numpy=
array([[-2.6016414],
       [ 4.7400627]], dtype=float32)>
```

通过build，指定了`output_shape = (2, 1)`，并分配默认值给`w`和`b`。

使用`get_weights()`来获取参数。注意只有build后才能调用，默认`w`设定为一个随机极小值，而`b`设为0。

```py
>>> linear_layer.get_weights()
[array([[ 0.61818826],
       [-0.9002726 ],
       [-1.1841295 ]], dtype=float32), array([0.], dtype=float32)]
```

可以观察到，w是2-D array，b是1-D array，使用`set_weights()`来设置参数，。TensorFlow对数据类型要求比较严格，务必遵守数据的格式。

```py
>>> linear_layer.set_weights([np.array([1, 2, 3]).reshape(-1,1) ,np.array([6])])
>>> linear_layer.get_weights()
[array([[1.],
       [2.],
       [3.]], dtype=float32), array([6.], dtype=float32)]
```

> 这里注意下reshape只能拿来把向量转成2D，如果想转置2D array，用`.T`



也可以选择在创建层的时候就指定input_shape，这里顺带提一下使用Sequential()建立model，并用`model.summary()`获取模型信息。注意这里指定input_shape的写法只有在Sequential内有效，看了下源码，好像是因为Sequential内部会对每层进行初始化。

```py
>>> model = tf.keras.Sequential([
...     tf.keras.layers.Dense(1, activation = 'linear', input_shape = (2, 1))])
>>> model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 2, 1)              2         
                                                                 
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
_________________________________________________________________
>>> model.get_weights()
[array([[-0.7633475]], dtype=float32), array([0.], dtype=float32)]
```

`model.summary()`也需要layer经过build后才可以调用，能够显示每层的统计信息。



对于线性回归，因为只有一个神经元，所以既可以直接使用linear_layer求取结果，也可以利用model来预测

```py
# 创建模型/层
>>> model = tf.keras.Sequential([
...     tf.keras.layers.Dense(1, activation = 'linear', input_shape = (2, 1))])
>>> linear_layer = tf.keras.layers.Dense(1, activation = 'linear', )
>>> linear_weights.build((2, 1))
>>> linear_layer.build((2, 1))
# 检查权重
>>> model.get_weights()
[array([[0.31324422]], dtype=float32), array([0.], dtype=float32)]
>>> linear_layer.get_weights()
[array([[1.473396]], dtype=float32), array([0.], dtype=float32)]
# 设置权重
>>> w = np.array([[1]]); b = np.array([2])
>>> model.set_weights([w, b]); linear_layer.set_weights([w, b])
>>> model.get_weights(); linear_layer.get_weights()
[array([[1.]], dtype=float32), array([2.], dtype=float32)]
[array([[1.]], dtype=float32), array([2.], dtype=float32)]
# 进行预测
>>> x = np.array([[-4]])
>>> linear_layer(x)
<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[-2.]], dtype=float32)>
>>> model.predict(x)
1/1 [==============================] - ETA: 0s
1/1 [==============================] - 0s 12ms/step
array([[[-2.]]], dtype=float32)
```

> 对于不确定X的情况，可以使用`.build(X.shape)`



## Sigmoid Activation

在创建layer的时候设置`activation`为`sigmoid`。

```py
model = Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
    ])
```

查看和设置参数

```py
>>> model.get_weights()
[array([[-0.83704394]], dtype=float32), array([0.], dtype=float32)]
>>> w = np.array([[2]]); b = np.array([3])
>>> model.set_weights([w, b])
```

推理

```py
>>> x = np.array([-1, 2])
>>> model.predict(x)

1/1 [==============================] - ETA: 0s
1/1 [==============================] - 5s 5s/step
array([[0.73105854],
       [0.999089  ]], dtype=float32)
```



# Implement Simple Neural Network Using TensorFlow

用烘烤咖啡豆的例子，首先绘制图像观察数据

![](D:\CS\Machine Learning\Course_2_Advanced Algorithm\2-TensorFlow Implements.assets\cofee.png)



对数据进行标准化

```PY
norm_1 = tf.keras.layers.Normalization(axis = -1)
norm_1.adapt(X)
Xn = norm_1(X)
```

> `Normalization`是一个预置层，可以用于将连续数据标准化。`.adapt()`方法可以计算数据集的mean和variance（均值和方差）。



实现如下神经网络

![](D:\CS\Machine Learning\Course_2_Advanced Algorithm\2-TensorFlow Implements.assets\Network.png)

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(3, activation = 'sigmoid', name = 'layer1'),
    Dense(1, activation = 'sigmoid', name = 'layer2')
])
```

> `tf.keras.Input(shape(2, ))`指定了输入的shape，使得tf可以在这一步设置权重和参数。这一步是可以省略的，tf会在调用`model.fit`的时候设置权重和参数。
>
> 这里最后一层使用sigmoid函数在实践中不是最好的方法。替代方法后面会说



测试权重的实例化，`W`的shape应该符合`(输入特征的数量，层中的神经元个数)`，在layer 1中，为(2, 3)，layer 2中为(3, 1)；`b.shape`应该符合神经元的个数，在layer 1中为3，layer 2中为1

```py
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)
'''
W1(2, 3):
 [[ 0.08 -0.3   0.18]
 [-0.56 -0.15  0.89]] 
b1(3,): [0. 0. 0.]
W2(3, 1):
 [[-0.43]
 [-0.88]
 [ 0.36]] 
b2(1,): [0.]
'''
```



`.compile()`和`.fit()`分别用于编译和拟合模型，具体的实现后面会整理

```py
model.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
)

model.fit(
	Xt, Yt,
    epochs = 10,
)
```



检查训练后的权重

```py
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)
'''
W1:
 [[-3.60e-03 -1.06e+01 -1.72e+01]
 [-8.37e+00 -2.29e-01 -1.43e+01]] 
b1: [-10.51 -11.54  -2.49]
W2:
 [[-39.3 ]
 [-43.93]
 [ 31.35]] 
b2: [-8.83]
'''
```



创建测试数据并进行预测，因为训练的时候对X进行过标准化，测试数据也需要进行标准化

```py
X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)
'''
predictions = 
 [[9.63e-01]
 [3.03e-08]]
 '''
```



设置阈值，输出0/1结果

```
yhat = (predictions >= 0.5).astype(int)
```



# Epochs and batches

`epochs`指定整个数据集被训练的轮数，在训练过程中会显示类似下面的标记

![image-20230714153451551](D:\CS\Machine Learning\Course_2_Advanced Algorithm\2-TensorFlow Implements.assets\image-20230714153451551.png)

`Epoch`指示当前进行的轮次。训练集通常会被分解为许多个`batch`，默认的batch大小为32，上图中batch的数量为6250.



# Layer Functions

检查下layer中的函数，看看这些函数是如何在确定咖啡豆最佳烘培时间中起作用的。绘制`W1, b1`对应的三条曲线

![](D:\CS\Machine Learning\Course_2_Advanced Algorithm\2-TensorFlow Implements.assets\logistics.png)



如图所示，每个神经元对应一个阈值，三个区域共同划分出好豆和坏豆的区域。且神经网络使用梯度下降自行学习出的这些函数，非常接近现实中人们的选择。



layer 2中的函数不是很好绘制，因为特征是3元的，尝试绘制一个3D视图，并用颜色标记好坏豆

![](D:\CS\Machine Learning\Course_2_Advanced Algorithm\2-TensorFlow Implements.assets\layer 2.png)

