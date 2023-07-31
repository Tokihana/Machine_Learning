# 入门

## 参加竞赛

在每个竞赛的首页（the competition page），点击`Join competition`，如果你发现显示的不是这个，而是`Submit Predictions`按钮，那么你已经加入这个竞赛中了

参加竞赛需要同意一些规则，例如每天最多提交几次，团队人数限制，以及某些竞赛自定的规则。点击`I understand and Accept`表示你将遵守这些规则。



## Titanic competition

这个竞赛的目标很简单：使用泰坦尼克号乘客的数据，尝试预测谁更可能在这场灾难中存活下来。

![image-20230722160833834](D:\CS\Machine Learning\Projects\Titanic\Tutorial.assets\image-20230722160833834.png)



## Data

点击Data栏，查看数据集的相关信息，从Data Explorer中可以看出，数据集包含三个文件：`gender_submission.csv`, `test.csv`, `train.csv`



### train.csv

`train.csv`包含一部分乘客的详细信息（891名乘客，每名乘客一行）。点击文件名就可以检查特定的文件

![image-20230722161307193](D:\CS\Machine Learning\Projects\Titanic\Tutorial.assets\image-20230722161307193.png)

`Survived`标记了该乘客是否幸存。



### test.csv

使用`train.csv`训练的模型，用来预测`test.csv`中的418名乘客是否存活

查看`test.csv`的信息，可以观察到这个数据集没有`Survived`列



### gender_submission.csv

这个文件是提交结果的示例，你应该按照该文件的格式整理预测结果并提交。

提交应该有以下两列：

- `PassengerId`，测试集中的乘客编号
- `Survived`，该乘客是否存活。



## Coding Environment

### Kaggle Notebook

点击`Code`栏，再点击`New Notebook`

创建新的Notebook后，可以进行下重命名



默认会给出一个代码栏，里面有一些注意事项，并提供了数据I/O。

```py
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```



### Load the data

你可以使用pandas的`.read_csv`方法加载`.csv`数据。数据的路径可以参考右边的Input栏

![image-20230722162951649](D:\CS\Machine Learning\Projects\Titanic\Tutorial.assets\image-20230722162951649.png)

例如，想要加载训练数据，可以

```py
train_data = pd.pandas.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
```

![image-20230722163044758](D:\CS\Machine Learning\Projects\Titanic\Tutorial.assets\image-20230722163044758.png)



同理将测试数据也加载出来

```py
test_data = pd.pandas.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
```



### First submission

可以先不考虑模型的问题，就像`gender_submission`文件中表示的那样，假定女性全部存活

首先来检查一下男性和女性的存活率是否有差异。提取`Survived`列，按性别划分

```python
women = train_data[train_data.Sex == 'female']['Survived']
man = train_data[train_data.Sex == 'male']['Survived']
```

因为`Survived`列是0/1值，因此可以简单计算出存活率

```py
women_rate = sum(women)/len(women)
men_rate = sum(men)/len(men)
print("Survival Rate:")
print(f"Women: {women_rate}")
print(f"Men: {men_rate}")
'''
Survival Rate:
Women: 0.7420382165605095
Men: 0.18890814558058924
'''
```



可以看出来这样也有一定的可行性，不过既然已经学过一些ML的知识了，不如尝试下写个更复杂的模型



### Logistic Regression

这里用几个数值特征做例子，离散特征需要处理为向量，这个方法被称为[One Hot](https://www.kaggle.com/code/dansbecker/using-categorical-data-with-one-hot-encoding/notebook)

Pandas提供了`.get_dummies()`来编码One Hot，直接用就行



提取数据

```py
y = train_data.Survived

features = ["Pclass", "Sex", "SibSp", "Parch"]
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
```



训练

```py
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y)
```



预测并写入结果

```py
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
```



### Save Version & submit

写入结果后，可以看到`Save Version`按钮，点击进去就可以保存一个提交

![image-20230722171446261](D:\CS\Machine Learning\Projects\Titanic\Tutorial.assets\image-20230722171446261.png)



选择`Save & Run ALL`，点击`Save`，这时候就会运行一遍（左下角弹窗）

![image-20230722171550999](D:\CS\Machine Learning\Projects\Titanic\Tutorial.assets\image-20230722171550999.png)



运行结束后，`Save Version`右边数字显示了当前有的版本数量

![image-20230722171717035](D:\CS\Machine Learning\Projects\Titanic\Tutorial.assets\image-20230722171717035.png)

点击这个数字查看当前的版本

![image-20230722171748735](D:\CS\Machine Learning\Projects\Titanic\Tutorial.assets\image-20230722171748735-1690017469085-1.png)



点击右边三点$\cdots$，选择`Open in Viewer`，这样会打开一个新窗口，显示这个版本的信息，选择Output栏

![image-20230722172019061](D:\CS\Machine Learning\Projects\Titanic\Tutorial.assets\image-20230722172019061.png)



可以看到有个Submit，点一下就能提交了。

![image-20230722172153046](D:\CS\Machine Learning\Projects\Titanic\Tutorial.assets\image-20230722172153046.png)



