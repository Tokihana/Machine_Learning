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

