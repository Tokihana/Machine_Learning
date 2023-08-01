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

对于连续特征，例如下图，通常使用选定阈值的方式进行划分

![image-20230801161727282](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801161727282.png)



选择多个阈值，分别计算信息增益，取增益最高的阈值与其他特征的信息增益进行比较。阈值通常从中位数开始取值。

![image-20230801161947400](D:\CS\Machine Learning\Course_2_Advanced Algorithm\8-Decision Tree Model.assets\image-20230801161947400.png)



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

