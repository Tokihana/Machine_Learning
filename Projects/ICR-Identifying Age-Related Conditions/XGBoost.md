# 设置离线运行

这个比赛要求必须离线运行代码，所以需要调整notebook options > internet

![image-20230803184240236](D:\CS\Machine Learning\Projects\ICR-Identifying Age-Related Conditions\XGBoost.assets\image-20230803184240236.png)

# 训练数据分析

## train & train

train 58列，其中Class是target。

test 57列

除了列EJ之外，其他列都是数字，且范围差距挺大。





# 数据预处理

## 导入

```py
train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
```



## 删除空target行

```py
train.dropna(axis = 0, subset = 'Class', inplace = True)
```



## 填充缺失值，整理数据

```py
y_train = train['Class']
X_train = pd.get_dummies(train.drop(['Id', 'Class'], axis = 1))
Id = test['Id']
X_test = pd.get_dummies(test.drop('Id', axis = 1))

X_train.fillna(X_train.mean(), inplace = True)
X_test.fillna(X_test.mean(), inplace = True)

assert not np.any(np.isnan(X_train.values))
assert not np.any(np.isnan(X_test.values))
assert not np.any(np.isnan(y_train.values))
```



检查是否存在One-hot编码缺失

```py
if 'EJ_B' not in X_test.columns:
    X_test['EJ_B'] = 0
if 'EJ_A' not in X_test.columns:
    X_test['EJ_A'] = 0
```



标准化数据

```py
X_train_norm = (X_train - X_train.mean()) / X_train.std()
X_test_norm = (X_test - X_test.mean()) / X_test.std()
```



# 训练 & 预测

```py
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train_norm, y_train)
```

```py
predictions = model.predict_proba(X_test_norm)
```

```py
output = pd.DataFrame({
    'Id':Id
})
output[['class_0','class_1']] = predictions
output.to_csv('/kaggle/working/submission.csv', index=False)
```



# 23/8/4 整理预处理流程

引入数据标准化，使用`sklearn.preprocessing.scale()`，同时对EJ做更好的处理

```py
from sklearn import preprocessing
```

```py
# train
train_EJ = train['EJ']
X_train = train.drop(['Id', 'Class','EJ'], axis = 1)
X_train.fillna(X_train.mean(), inplace = True)

X_train = pd.DataFrame(preprocessing.scale(X_train))
X_train['EJ_A'] = (train_EJ == "A").astype(int)
X_train['EJ_B'] = (train_EJ == 'B').astype(int)

assert not np.any(np.isnan(X_train.values))
print(len(X_train.columns))
X_train.head()
```

```py
# test
Id = test['Id']
test_EJ = test['EJ']
X_test = test.drop(['Id','EJ'], axis = 1)
X_test.fillna(X_test.mean(), inplace = True)

X_test = pd.DataFrame(preprocessing.scale(X_test))
X_test['EJ_A'] = (test_EJ == "A").astype(int)
X_test['EJ_B'] = (test_EJ == "B").astype(int)

assert not np.any(np.isnan(X_test.values))
print(len(X_test.columns))
X_test.head()
```



# 23/8/4 根据Loss建立评估机制，从train划分一部分val用于评估

使用`sklearn.model_selection.train_test_split()`

```py
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)
```



Evaluation 使用**balanced logarithmic loss**
$$
\mathrm{Log\ Loss} = \frac {-\frac 1 {N_0} \sum^{N_0}_{i = 1} y_{0i} \log p_{0i} - \frac 1 {N_1} \sum^{N_1}_{i = 1} y_{1i} \log p_{1i} } 2
$$
$N_c$是y_target中取值为c的样例数，$y_{ci}$代表target，$p_{ci}$为预测值

```py
def balanced_logarithmic_loss(y, yhat):
    '''
    Arg:
    y: 1D labels
    yhat: 2D labels with 2 classes
    
    Return:
    loss: balanced_logarithmic_loss
    '''
    loss_0 = (y * np.log(yhat[:, 0])).sum() / (y == 0).sum()
    loss_1 = (y * np.log(yhat[:, 1])).sum() / (y == 1).sum()
    loss = -(loss_0 + loss_1) / 2
    return loss
```

```py
yhat = model.predict_proba(X_val)
balanced_logarithmic_loss(y_val, yhat)
```



# 23/8/4 选择学习率，树深度和树规模

```py
min_loss = 1000000
best_alpha = 1
for alpha in [0.01, 0.001, 0.003, 0.005, 0.007, 0.009]:
    model = XGBClassifier(learning_rate = alpha)
    model.fit(X_train, y_train)
    yhat = model.predict_proba(X_val)
    loss = balanced_logarithmic_loss(y_val, yhat)
    if loss < min_loss:
        min_loss = loss
        best_alpha = alpha
best_alpha
```

```py
min_loss = 1000000
best_depth = 1
for depth in [5, 6, 7, 8, 9, 10]:
    model = XGBClassifier(learning_rate = 0.005, max_depth = depth)
    model.fit(X_train, y_train)
    yhat = model.predict_proba(X_val)
    loss = balanced_logarithmic_loss(y_val, yhat)
    print(f'Depth: {depth}, Loss: {loss}')
    if loss < min_loss:
        min_loss = loss
        best_depth = depth
best_depth
```

```py
min_loss = 1000000
best_n = 1
for n in [50, 100, 150, 200, 500, 1000, 5000]:
    model = XGBClassifier(learning_rate = 0.005, max_depth = 8, n_estimators = n)
    model.fit(X_train, y_train)
    yhat = model.predict_proba(X_val)
    loss = balanced_logarithmic_loss(y_val, yhat)
    print(f'N: {n}, Loss: {loss}')
    if loss < min_loss:
        min_loss = loss
        best_n = n
best_n
```

