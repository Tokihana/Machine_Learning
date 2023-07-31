# 预处理数据

需要删除掉`Name`, `Ticket`和`PassengerId`，，因为这些和是否生还基本没关系，且属于隐私数据

`Pclass`反应客舱的等级，这个需要展成向量

`Sibsp`, `Parch`分别反应是否有兄弟或配偶，父母或子女在船上

`Fare`是路费，

`Cabin`是客舱，虽然我感觉客舱会有影响，不过如果客舱数太多的话，可能会影响性能？

> 经过观察，训练集与测试集的客舱不一致，还是删掉比较好

`Embarked`是上船的港口。



load

```py
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
```



去除没法处理的列

```py
y = train_data.pop('Survived')
y.head()
X = pd.get_dummies(train_data.drop(['Name','Ticket','PassengerId','Cabin'], axis = 1))
# X.head()
passengerId = test_data.PassengerId
# passengerId.head()
test = pd.get_dummies(test_data.drop(['Name', 'Ticket', 'PassengerId','Cabin'], axis = 1))
# test.head()
print(f"{X.shape}\n{test.shape}")
```



tensorflow模型

```py
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid, linear, relu
```

```py
model = Sequential([
    tf.keras.layers.Input(shape = 10),
    Dense(10, activation = 'relu'),
    Dense(10, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)
```



拟合

```py
model.fit(X.values, y.values, epochs = 15)
```

这里出了问题，不知道为什么loss直接变成了nan，即使调整学习率也没用

![image-20230724153341493](D:\CS\Machine Learning\Projects\Titanic\Neural Network.assets\image-20230724153341493.png)

考虑到这个问题我目前搞不懂怎么解决，只能先作罢



