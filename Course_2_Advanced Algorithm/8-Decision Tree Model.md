# Cat classification example

这一章节使用的样例为猫咪分类样例，如下图所示，给出猫咪的三个特征，要求判断是否为猫咪（二分类任务）

![image-20230731172759104](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230731172759104.png)



# Decision Tree

决策树算法的模型是一棵树。对每个输入样例，算法会从根节点开始，逐步前进到对应target的叶子。

![image-20230731173013662](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230731173013662.png)



对一个任务有非常多种可能的决策树，例如

![image-20230731173347880](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230731173347880.png)



## Learning Process

决策树模型的主要流程遵循**分而治之**：
$$
\underline{
\overline{
\begin{array}{l}
\bf {def}\ BuildTree:\\
1. 创建node \\
2. \bold {if}\ 当前node对应的样例都是同一类别，建立叶子并\bf {return} \\
3. \bold {if}\ 当前集合为空 \empty，或者不存在可以划分的属性，建立叶子并\bf {return}\\
\\
4.不满足上两个返回条件，则按*最优划分属性*进行划分:\\
\bf {for}\ 每个属性取值:\\
\quad 生成分支，递入下一层
\end{array}
}
}
$$
![image-20230731175147611](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230731175147611.png)



几个重要的决策点：

- 如何选择每个node的属性？目的是让子节点足够纯净（pure）
- 何时停止split，即确定叶子
  * 都为同一类
  * 或者超过预定的深度（限制深度保证运行效率和防止过拟合)
  * 或者收益太小（容易过拟合）
  * 或者划分出来的子集太小



## Measuring purity - entropy

我们使用熵（entropy）来描述数据的混乱程度，回顾前面的知识，对特定分布的数据，其熵可以表示为
$$
H(p) = \mathrm {E}_{x\sim P}(-\log_2 P(x)) = -\sum P(x)\log_2 P(x)
$$
对样例中的二分类任务，可以写成$-p_1\log_2 (p_1) - (1 - p_1)\log_2 (1 - p_1)$，其曲线为

![image-20230801144737002](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801144737002.png)

如果数据更偏向某一类别（更纯净），熵值较低；如果一半一半（最混乱），熵值最高。



二分类熵值计算实现

```py
def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    # You need to return the following variables correctly
    entropy = 0.
    
    ### START CODE HERE ###
    if(len(y)):
        p_1 = y.sum()/len(y)
        if(p_1 != 0.0 and p_1 != 1.0):
            entropy = -p_1 * np.log2(p_1) - (1 - p_1) * np.log2(1 - p_1)
    ### END CODE HERE ###        
    
    return entropy
```



## Choosing a split - Information Gain 

在决策树中，熵的减少称为Information Gaim（信息增益）。

以猫咪预测的样例为例，首先构造根节点，分别验证三种特征划分后子集的熵

![image-20230801151309503](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801151309503.png)



求加权平均（Weighted arithmetic mean），因为决定混乱程度的因素还有集合的大小；在实际的构建流程中，还需要进一步求得在划分后获得的信息增益（information gain），即
$$
\begin{array}{l}
befor\ split:\ H(p) = H(0.5) = 1\\
ear\ shape:\ \frac 5 {10}H(0.8) + \frac 5 {10}H(0.2) = 0.72 \to 0.28\\
face\ shape:\ \frac 7 {10}H(0.57) + \frac 3 {10}H(0.33) = 0.969 \to 0.031\\
whiskers: \ \frac 4 {10}H(0.75) + \frac 6 {10}H(0.33) = 0.876 \to 0.124
\end{array}
$$
综上，根节点的划分特征应该选择Ear Shape.

计算信息增益的另一个好处是，当某次划分带来的增益过小的时候，可以直接终止递归，降低树的规模，并防止过拟合。



给出Information Gain的通式，设$p_1^{left}, p_1^{right}, p_1^{root}$分别代表左子、右子和根节点中正例的比例，$w^{left}, w^{right}$代表左右子的权重，则有
$$
Gain = H(p_1^{root}) - (w^{left}H(p_1^{left}) + w^{right}H(p_1^{right}))
$$

> 另一种度量的方法是Gini index，基尼指数。该指数反映了数据集中随机抽取两个样本，其target不一致的概率，Gini越小则纯度越高。
>
> 还有一种方法是gain ration（增益率），用于优化信息增益法可能存在的对数量较多属性的偏好。
>
> 信息增益、增益率和基尼指数分别对应了$ID3, C4.5和CART$三种决策树算法。



实现计算信息增益

```py
def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (ndarray):  List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (ndarray): Indices with feature value == 1
        right_indices (ndarray): Indices with feature value == 0
    """
    
    # You need to return the following variables correctly
    left_indices = []
    right_indices = []
    
    ### START CODE HERE ###
    left_indices = np.array(node_indices)[X[node_indices, feature] == 1] # 转array防止输入是list
    right_indices = np.array(node_indices)[X[node_indices, feature] == 0]
    ### END CODE HERE ###
        
    return left_indices.tolist(), right_indices.tolist()
```

```py
def compute_information_gain(X, y, node_indices, feature):
    
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
   
    Returns:
        cost (float):        Cost computed
    
    """    
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    # You need to return the following variables correctly
    information_gain = 0
    
    ### START CODE HERE ###
    
    # Weights 
    w_left = len(left_indices) / len(node_indices)
    w_right = len(right_indices) / len(node_indices)
    #Weighted entropy
    H_left = w_left * compute_entropy(y_left)
    H_right = w_right * compute_entropy(y_right)
    #Information gain                                                   
    information_gain = compute_entropy(y_node) - H_left - H_right
    ### END CODE HERE ###  
    
    return information_gain
```



选择合适的feature

```py
def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    
    # Some useful variables
    num_features = X.shape[1]
    
    # You need to return the following variables correctly
    best_feature = -1
    best_gain = 0
    
    ### START CODE HERE ###
    for feature in range(num_features):
        gain = compute_information_gain(X, y, node_indices, feature)
        if(gain > best_gain):
            best_feature = feature
            best_gain = gain
       
    ### END CODE HERE ##    
   
    return best_feature
```



## Build a tree

```py
def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    tree.append((current_depth, branch_name, best_feature, node_indices))
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)
```



# Futher Refinements

## One-hot

在此前的例子中，特征往往都是二项的，即圆脸/非圆脸，有胡须/没胡须。假设某个特征存在超过两个的可能值，例如

![image-20230801155200331](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801155200331.png)

这种情况下，可以使用One-hot编码，将该列别划分为多个二项特征，可以观察到每个样例只有一个子特征的值为1，即One-hot

![image-20230801155307274](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801155307274.png)

> 很多开源库都提供了这个功能，例如pandas的`.get_dummies()`。
>
> One-hot不止适用于决策树，也可以用于神经网络。



## Continuous valued features

对于连续特征，例如下图，通常使用选定阈值的方式进行划分（连续值离散化，Discretization）

![image-20230801161727282](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801161727282.png)



选择多个阈值，分别计算信息增益，取增益最高的阈值与其他特征的信息增益进行比较。阈值通常从中位数开始取值。

![image-20230801161947400](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801161947400.png)



## Missing Value

对于不能包含缺失值的模型，需要填充缺失值，常用的方法为：

- 对连续属性，填充均值
- 对离散属性，填充众数

这样填充基于样本都是独立同分布的假设。需要注意的是，填充只能用于feature缺失，若target缺失，一般会直接抛弃该样本。

还有一种方法是矩阵补全（matrix completion），在低秩假设下恢复数据。

# Regression Trees

如果预测的目标值不是离散的类别，而是连续的值，即回归问题，例如

![image-20230801162259672](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801162259672.png)

此时决策树泛化为回归树。

![image-20230801162705293](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801162705293.png)



## Choosing a split - variance

和决策树一样，回归树也需要某个用来评估复杂度、选择合适划分特征的尺度。我们希望该尺度能够反映数据的波动情况（离散程度），很容易想到的方法是求方差。
$$
variance = \sum(x - \mu)^2
$$
![image-20230801163243790](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801163243790.png)

同理进行加权平均后，求取信息增益
$$
Gain = v^{root} - (w^{left}v^{left} + w^{right}v^{right})
$$

> 方差最小，均值更能够表现当前节点的共同特征。





# Multiple decision trees

单个决策树往往对数据的微小变化非常敏感，如下图所示，将其中一个猫咪样例的耳朵形状改变，树的构建就会发生非常大的变化

![image-20230801175214635](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801175214635.png)



解决这一问题的方式是使用多个树，又称tree ensemble，综合多个树的结果做出推断

![image-20230801175434700](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801175434700.png)



## Build tree ensemble - sampling with replacement

有放回抽样（sampling with replacement）即在抽样后，将抽出的样例放回样本中，再进行下一次抽样的过程。

举例来说，假设有四个硬币，分别为红、黄、绿、蓝色，有放回抽样，每次取一个

![image-20230802143030330](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230802143030330.png)



将这个过程应用于构建决策树：

1. 将所有的数据构成数据集，然后进行有放回的抽样，得到一个训练集（可能会重复，没关系）
2. 重复多次抽样，并进行训练



## Random forest algorithm

$$
\begin{array}{l}
训练集大小 = m;\\
\mathbf {for}\ b = 1\ to\ B:\\
\quad 有放回抽样创建大小同样为m的新数据集;\\
\quad 使用抽样得到的数据集训练一棵决策树;\\
\end{array}
$$

> 决策树的数量B通常在100左右，64到228之间都可以；过大的B不会带来更好的模型性能，反而会增大运行开销。
>
> 这种构建方式也被称为bagged decision tree，因为每棵树都是在一个虚拟的bag上训练而成的。



bagged decision tree的森林中可能会训练出相似的树结构，导致模型效果不好，可以进一步做出改进，增加特征选择的随机性，即**随机森林算法（Random forest algorithm）**：当需要从n个特征中选择当前节点的划分特征时，首先从n个特征中随机抽样出包含k个特征的子集，从该子集的k个特征中选择划分特征。k的大小一般是$\sqrt n$.

> 通过随机抽样特征，每个决策树都会更多考虑数据集发生微小变化的可能性，并通过组成森林平均这些扰动，从而提升了模型对微小变化的鲁棒性。



# XGBoost

通过对决策树算法进行些许修改，可以获得更好的模型性能。
$$
\begin{array}{l}
训练集大小 = m;\\
\mathbf {for}\ b = 1\ to\ B:\\
\quad 有放回抽样创建大小同样为m的新数据集;\\
\quad \quad 为当前森林预测错误的样例分配更高的抽样概率，使得新的子集更可能出现此前预测错误的样例\\
\quad 使用抽样得到的数据集训练一棵决策树;\\
\end{array}
$$
直观上理解，Boost tree算法使得新的树更多关注森林不能正确处理的样例，从而提升了模型的表现。

![image-20230802160617867](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230802160617867.png)



XGBoost（eXtreme Gradient Boosting）是Boost一个开源实现，提供了一系列boost算法的模型和方法，并且内置正则化方法，广泛在竞赛中使用。

```py
# Classification
from xgboost import XGBClassifier

model = XGBClassifer()
model.fit(X, y)
pred = model.predict(X_test)
```

```py
# Regression
from xgboost import XGBRegressor

model = XGBRegressor
model.fit(X, y)
pred = model.predict(X_test)
```



# When to use decision trees

**决策树**通常更适合表格（结构化）数据（tabular or structured data），数据类似于电子表格（spreadsheet）；

不适合在非结构化数据，例如图像、音频、文本等数据中使用结构树，这类数据通常不会存在电子表格里。

相比神经网络，决策树的训练时间通常更短；且较小的决策树是可解释的（interpretable），可以通过输出整个树来直观理解决策树如何进行决策。

> 决策树的可解释性通常需要结合可视化手法，尤其是树很大的时候。
>
> 通常XGBoost就足够大多数需求了



**神经网络**适合所有类型的数据，表格和非结构化数据都可以。

对图像、音频、文本等任务，神经网络一般是首选。

比决策树慢，因此周期会比较长。

不过可以使用迁移学习，一方面减少数据量。

多个神经网络可以组合成更大的系统，因为神经网络的输出通常是平滑或连续的，可微。



# 参考

- 吴恩达《机器学习2022》
- 西瓜书
- 南瓜书
