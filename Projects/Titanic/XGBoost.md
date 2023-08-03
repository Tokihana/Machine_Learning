# 预处理数据

## 导入

```py
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
```



## 预处理

```py
y_train = train.pop('Survived')
assert not np.any(np.isnan(y_train.values))
y_train.shape
```

```py
X_train = pd.get_dummies(train.drop(['PassengerId', 'Name','Ticket','Cabin'], axis = 1))
X_train.fillna(X_train.mean(), inplace = True)
assert not np.any(np.isnan(X_train.values))

print(X_train.shape)
X_train.head()
```

```py
PassengerId = test.pop('PassengerId')
PassengerId.shape
```

```py
X_test = pd.get_dummies(test.drop(['Name', 'Ticket', 'Cabin'], axis = 1))
X_test.fillna(X_test.mean(), inplace=True)
assert not np.any(np.isnan(X_test.values))

print(X_test.shape)
X_test.head()
```



# 训练&预测

```py
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
```

```
predictions = model.predict(X_test)
print(predictions)
```



# 导出

```py
output = pd.DataFrame({
    'PassengerId':PassengerId, 
    'Survived':predictions
})
output.to_csv('submission.csv', index=False)
```



# 提交

![image-20230803173509164](D:\CS\Machine Learning\Projects\Titanic\XGBoost.assets\image-20230803173509164.png)