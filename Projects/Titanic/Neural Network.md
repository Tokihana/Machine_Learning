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



去除没法处理的列，整理训练集和测试集

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



检查dtypes，确认一致性

```py
print(X.dtypes)
print(test.dtypes)
```



## 填充nan

train和test中都存在很多nan，需要填充一下。`.fillna(.mean())`用来填充均值，

```py
X = X.fillna(X.mean())
y = y.fillna(y.mean())
```



# 神经网络

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
```



检查下是否存在nan

```py
assert not np.any(np.isnan(X.values))
```



拟合

```py
model.fit(X.values, y.values, epochs = 15)
model.summary()
```



输出预测结果

```
test.fillna(test.mean())
predictions = model.predict(test).round()
```

```
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions[:, 0]})
output.to_csv('submission.csv', index=False)
```



# 提交

![image-20230803163723275](D:\CS\Machine Learning\Projects\Titanic\Neural Network.assets\image-20230803163723275.png)

score 0 可还行，我估计哪里还有问题，得继续改改



# 修改

检查了一下，发现有两个问题，

首先test没赋值

```py
test = test.fillna(test.mean())
assert not np.any(np.isnan(X.values))
```


其次是没有转int

```py
predictions = model.predict(test).round().astype(int)
```

![image-20230803165216781](D:\CS\Machine Learning\Projects\Titanic\Neural Network.assets\image-20230803165216781.png)

OK了，非常好。
