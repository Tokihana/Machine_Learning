# Underfitting & Overfitting

在训练模型的过程中，我们通常希望模型能够很好地反映训练集的特征的同时，也能够泛化（generalization）到新的数据上（在新的数据集上也能表现得很好）。

在训练集上表现不好的模型，称为欠拟合（underfit），也称模型与训练集有很高的偏差（bias）；在训练集上表现得很好，在新数据上表现不好的模型，称为过拟合（overfit），也称有高方差（high variance）。如图所示，在房价预测模型中设计不同阶的特征，左边的模型欠拟合，右边的模型过拟合。

![image-20230628171136482](D:\CS\Machine Learning\9-Overfitting & Regularization.assets\image-20230628171136482.png)



对于分类任务，同样需要关注欠拟合与过拟合问题，如图所示，过拟合和欠拟合的模型都不能称得上是良好的模型。

![image-20230628171826833](D:\CS\Machine Learning\9-Overfitting & Regularization.assets\image-20230628171826833.png)



# Addressing Overfitting

- 获取更多的训练数据（复杂的特征需要更多数据的支持）。有的时候很难实现。

  ![image-20230628172058432](D:\CS\Machine Learning\9-Overfitting & Regularization.assets\image-20230628172058432.png)

- 使用更少的特征，尤其是高阶多项式特征，并尝试通过特征设计选择合适的特征（feature selection），但注意，排除掉部分特征可能导致信息丢失。可以通过自动化手段来选择这些特征。

- 正则化（Regularization）。保留所有的特征，减少过大特征的影响（缩小对应特征的数量级）；通常只需要正则化$w_j$，不需要正则化$b$





# Regularization





