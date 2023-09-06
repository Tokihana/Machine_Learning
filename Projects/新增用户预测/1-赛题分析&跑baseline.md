# 环境配置

DataWhale的baseline，可以直接fork：https://aistudio.baidu.com/projectdetail/6618108?contributionType=1&sUid=1020699&shared=1&ts=1691406191660

在个人中心 > 项目中可以找到fork的项目。如果没有GPU，可以试着申请一下免费试用。



# 背景&任务

分析用户是否为新增用户，有助于产品迭代升级，属于二分类问题



# 数据

## 数据说明

数据包含13个字段（特征），62万条训练集，20万条测试集。

![image-20230816115058435](D:\CS\Machine Learning\Projects\新增用户预测\1-赛题分析&跑baseline.assets\image-20230816115058435.png)

`uuid`为样本的唯一标识，`eid`为访问行为ID，`udmap`为行为属性，包含`key1 ~ key9`，其他的字段是用户相关的属性，匿名处理过。

`target`为预测结果字段。



## 评估指标

评估标准为$F_1$

```py
from sklearn.metrics import f1_score

score = f1_score(y_true, y_pred, average = 'macro')
```



## 提交要求

csv格式，编码为utf-8，第一行为表头。



# 处理思路

![](D:\CS\Machine Learning\Projects\新增用户预测\1-赛题分析&跑baseline.assets\whiteboard_exported_image.png)

任务属于二分类任务，根据输入的数据预测用户是否属于新增用户。

这里用机器学习方法，深度学习方法一定程度上可以自动学习特征，但对特定问题，手动设计特征可能更加有效。

逻辑回归和决策树两者的选择：决策树能够处理非线性关系，并且可以捕获特征之间的交互作用，能够生成可解释的规则，便于理解模型的决策依据，帮助设计特征。



# 跑baseline

## 观察数据dtype

```py
train_data.dtypes
'''
uuid                        int64
eid                         int64
udmap                      object
common_ts          datetime64[ns]
x1                          int64
x2                          int64
x3                          int64
x4                          int64
x5                          int64
x6                          int64
x7                          int64
x8                          int64
target                      int64
key1                      float64
key2                      float64
key3                      float64
key4                      float64
key5                      float64
key6                      float64
key7                      float64
key8                      float64
key9                      float64
eid_freq                    int64
eid_mean                  float64
udmap_isunknown             int64
common_ts_hour              int64
'''
```

可以看到`udmap`和`common_ts`不是数值型，需要处理一下



## 导入数据

```py
import pandas as pd
import numpy as np

train_data = pd.read_csv('用户新增预测挑战赛公开数据/train.csv')
test_data = pd.read_csv('用户新增预测挑战赛公开数据/test.csv')

train_data['common_ts'] = pd.to_datetime(train_data['common_ts'], unit='ms')
test_data['common_ts'] = pd.to_datetime(test_data['common_ts'], unit='ms')

train_data.head()
```



## 数据处理

```py
def udmap_onethot(d):
    v = np.zeros(9)
    if d == 'unknown':
        return v
    
    d = eval(d)
    for i in range(1, 10):
        if 'key' + str(i) in d:
            v[i-1] = d['key' + str(i)]
            
    return v

train_udmap_df = pd.DataFrame(np.vstack(train_data['udmap'].apply(udmap_onethot)))
test_udmap_df = pd.DataFrame(np.vstack(test_data['udmap'].apply(udmap_onethot)))

train_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
test_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
```



`udmap_onehot`函数用于将`udmap`特征映射为向量，`udmap`可能是一个字典或`unknown`，当值为`unknown`的时候，映射为空向量；当为字典时，对应`key1 ~ 9`映射。

`np.vstack`方法用于将array呈竖列堆叠，堆叠`train_data['udmap'].apply(udmap_onehot)`后的每一行都是一个array，包含`key 1~9`的对应值。

`train_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]`为处理后的udmap分配列名称。

将处理后的`udmap`连接到训练集中

```py
train_data = pd.concat([train_data, train_udmap_df], axis=1)
test_data = pd.concat([test_data, test_udmap_df], axis=1)
```



## 设计新的特征

提取`eid`的频次

```py
train_data['eid_freq'] = train_data['eid'].map(train_data['eid'].value_counts())
test_data['eid_freq'] = test_data['eid'].map(train_data['eid'].value_counts())
```

`DataFrame.map`方法会对每个元素应用括号中的函数，并返回一个标量值。

在这里，`train_data['eid'].value_counts()`返回`eid : freq`的映射关系，则对于每个`eid`，map会返回对应的频次结果

```py
train_data['eid'].value_counts()
26    174811
35     82643
...
22         1
24         1
Name: eid, dtype: int64
```

```py
train_data['eid'].value_counts()[26]
174811
```



同理，设计`eid`的`target`均值，相当于对应用户行为的新增率

```py
train_data['eid_mean'] = train_data['eid'].map(train_data.groupby('eid')['target'].mean())
test_data['eid_mean'] = test_data['eid'].map(train_data.groupby('eid')['target'].mean())
```

`groupby('eid')`方法将整个数据集按`eid`分组，`train_data.groupby('eid')['target'].mean()`返回`eid : prob`的映射关系，对每个eid返回其用户新增率。



## 处理时间戳

`common_ts`是`datatime`类型的数据，可以通过`.dt.XX`的方式提取时间，例如提取`hour`

```py
train_data['common_ts_hour'] = train_data['common_ts'].dt.hour
test_data['common_ts_hour'] = test_data['common_ts'].dt.hour
```



## 训练决策树

```py
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier() # 0.62
clf.fit(
    train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1),
    train_data['target']
)
```



## 预测结果

```py
pd.DataFrame({
    'uuid': test_data['uuid'],
    'target': clf.predict(test_data.drop(['udmap', 'common_ts', 'uuid'], axis=1))
}).to_csv('submit.csv', index=None)
```



# 提交

下载`submit.txt`文件，在赛事主页 > 提交结果，选择该文件并提交

![image-20230816151853960](D:\CS\Machine Learning\Projects\新增用户预测\1-赛题分析&跑baseline.assets\image-20230816151853960.png)

# 参考

- [科大讯飞 用户新增预测挑战赛](https://challenge.xfyun.cn/topic/info?type=subscriber-addition-prediction&option=ssgy)
- [AI夏令营第三期 - 用户新增预测挑战赛教程](https://datawhaler.feishu.cn/docx/HBIHd7ugzoOsMqx0LEncR1lJnCf)
- [机器学习实践：用户新增预测挑战赛 baseline](https://aistudio.baidu.com/projectdetail/6618108?contributionType=1&sUid=1020699&shared=1&ts=1691406191660)

