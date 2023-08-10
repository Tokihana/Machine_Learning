# K-fold

k-fold是一种交叉验证方法，该方法将样例划分为k个子集，称fold。

若k = n，则每个子集包含1个样例。下面的代码显示了2-fold

```py
import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print(train,test)
'''
[2 3] [0 1]
[0 1] [2 3]
'''
```

```py
>>> kf = KFold(n_splits=3)
>>> for train, test in kf.split(X):
...     print(train, test)
... 
[2 3] [0 1]
[0 1 3] [2]
[0 1 2] [3]
```

```py
>>> kf = KFold(n_splits=10)
>>> for train, test in kf.split(X):
...     print(train, test)
... 
ValueError: Cannot have number of splits n_splits=10 greater than the number of samples: n_samples=4.
```

> 从上面几个测试可以看出，KFold实际上是给出多种数据集的划分方式，`n_splits`不能超过数据集的最大可划分数



每个fold包含两个array：第一个代表train，第二个代表test。array的值为index值。因此可以用下面的方法来划分数据

```py
for train, test in kf.split(X):
    print(train,test)
	X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
```



# RepeatedKFold

Repeated k-fold重复k-fold多次。每次生成不同的划分序列。可以通过radom_state保证每次生成的结果一致。

```py
>>> from sklearn.model_selection import RepeatedKFold
>>> rkf = RepeatedKFold(n_splits = 2, n_repeats=2, random_state=1)
>>> for train, test in rkf.split(X):
...     print(train, test)
... 
[0 1] [2 3]
[2 3] [0 1]
[1 3] [0 2]
[0 2] [1 3]
```

```py
>>> rkf = RepeatedKFold(n_splits = 3, n_repeats=2, random_state=1)
>>> for train, test in rkf.split(X):
...     print(train, test)
... 
[0 1] [2 3]
[1 2 3] [0]
[0 2 3] [1]
[1 3] [0 2]
[0 2 3] [1]
[0 1 2] [3]
```



如何理解`n_splits`和`n_repeats`？`n_splits`决定了数据集划分的比例，而`n_repeats`会变动分配的索引。



# Stratification k-fold

使用分层抽样（stratification）的k-fold，分层抽样用来处理分布不均的样本（例如正例数远大于反例数），对这种样本，分层抽样可以保证样本的分布频率。

```py
>>> from sklearn.model_selection import StratifiedKFold, KFold
>>> import numpy as np
>>> X, y = np.ones((50, 1)), np.hstack(([0] * 45, [1] * 5))
# skf results
>>> skf = StratifiedKFold(n_splits=3)
>>> for train, test in skf.split(X, y):
...     print('train -  {}   |   test -  {}'.format(
...         np.bincount(y[train]), np.bincount(y[test])))
... 
train -  [30  3]   |   test -  [15  2]
train -  [30  3]   |   test -  [15  2]
train -  [30  4]   |   test -  [15  1]
# kf results
>>> kf = KFold(n_splits=3)
>>> for train, test in kf.split(X, y):
...     print('train -  {}   |   test -  {}'.format(
...         np.bincount(y[train]), np.bincount(y[test])))
... 
train -  [28  5]   |   test -  [17]
train -  [28  5]   |   test -  [17]
train -  [34]   |   test -  [11  5]
```

类似`RepeatedKFold`，`RepeatedStratificationKFold`用来生成多组分层k-fold。



# 用于构建树集合

```py
# Repeated Stratified k-fold
from sklearn.model_selection import RepeatedStratifiedKFold
rskf = RepeatedStratifiedKFold(n_splits = 8, n_repeats = 10, random_state = 2000)
```

```py
# ensemble trees
import copy
def ensemble(rskf, X, y):
    X=X.to_numpy()
    y=y.to_numpy()
    models = []
    models_list = []
    for train_idx, test_idx in rskf.split(X, y):
        X_val, y_val = X[test_idx], y[test_idx]
        X_train, y_train = X[train_idx], y[train_idx]
        model = XGBClassifier(max_depth = 6, learning_rate = 0.01, n_estimators = 100)
        model.fit(X_train, y_train)
        models.append(model)
        models_list.append(copy.deepcopy(models))
    return models_list, models
models_list, last_models= ensemble(rskf, X_train, y_train)
```

