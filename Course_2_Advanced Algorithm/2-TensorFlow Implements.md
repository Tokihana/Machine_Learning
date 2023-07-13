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
>>> linear_layer.set_weights([np.array([1, 2, 3]).reshape(-1,1),np.array([6])])
>>> linear_layer.get_weights()
[array([[1.],
       [2.],
       [3.]], dtype=float32), array([6.], dtype=float32)]
```



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

