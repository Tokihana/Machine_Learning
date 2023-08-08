# Unsupervised learning

在无监督学习中，数据只有features，没有label，例如下图

![image-20230808170308746](D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230808170308746.png)

无监督学习的目标是通过学习找到数据的内在规律。最常见的任务类型是聚类（Clustering）。

# Clustering

聚类的目标是将样本划分为若干个不相交的子集。既可以作为单独过程，也可以作为其他学习任务的前驱过程。 



# K-means Clustering

k均值是一种**原型聚类（prototype-based clustering）**方法，算法假设聚类能够通过原型进行刻画，原型指样本空间中**具有代表性的点**。k均值视图找到一组**原型向量（prototype vector）**，通过最小化样本围绕这些向量的紧密程度，使得这些向量最具代表性。

