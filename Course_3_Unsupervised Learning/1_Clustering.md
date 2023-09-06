# Unsupervised learning

在无监督学习中，数据只有features，没有label，例如下图

![image-20230808170308746](D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230808170308746.png)

无监督学习的目标是通过学习找到数据的内在规律。最常见的任务类型是聚类（Clustering）。

# Clustering

聚类的目标是将样本划分为若干个不相交的子集。既可以作为单独过程，也可以作为其他学习任务的前驱过程。 



# K-means Clustering

## Intuition

k均值是一种**原型聚类（prototype-based clustering）**方法，算法假设聚类能够通过原型进行刻画，原型指样本空间中**具有代表性的点**。k均值视图找到一组**原型向量（prototype vector）**，通过最小化样本围绕这些向量的紧密程度，使得这些向量最具代表性。

1. 随机选择一组点作为最开始的**质心（centroid）向量**，centroid和feature的维数相同

   <img src="D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230809151719487.png" alt="image-20230809151719487" style="zoom:50%;" />

2. 遍历所有的样例，根据距离匹配质心

   <img src="D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230809151850077.png" alt="image-20230809151850077" style="zoom:50%;" />

3. 计算每组样例的平均值，将质心移动到该点。

   <img src="D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230809152006035.png" alt="image-20230809152006035" style="zoom:50%;" />

4. 重复"遍历-移动到均值"的过程。直到收敛（质心位置变化变得非常小）

   <img src="D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230809152201738.png" alt="image-20230809152201738" style="zoom:50%;" />

   <img src="D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230809152244432.png" alt="image-20230809152244432" style="zoom:50%;" />



## Algorithm

算法的伪码表示如下
$$
\over{
\underline{
\begin{array}{l}
输入:样本D;\\
\quad\quad 类别数k;\\
过程:\\
1.初始化k个质心{\mu_1, \cdots, \mu_k}, 一般是选k个样例\\
2.\bf{repeat}\\
3.\quad \mathbf{for} \ x_i \ \mathbf{in}\ D:\\
4.\quad\quad min_k||x_i - \mu_k||^2, 划分到距离最近的类别\\
5.\quad \mathbf{for}\ \mu_i\ \mathbf{in}\ \{\mu_1, \cdots, \mu_k\}:\\
6.\quad\quad 计算新的均值\mu_i^{’}\\
7.\quad\quad \mathbf{if}\ \mu_i^{'} \neq \mu_i:\\
8.\quad\quad\quad 更新\mu_i\\
9.\mathbf{until}\ 没有任何质心向量被更新\\
输出: 划分的聚类
\end{array}
}
}
$$

> 在划分类别的时候，可能出现某些类别样例数为0的情况，此时可以选择删除该类别，或者重新随机化初始化这个类别。
>
> 算法的过程使用了贪心思想：每轮迭代都选择当前聚类的最优解（均值）

简单的代码框架

```py
# Initialize centroids
centroids = kMeans_init_centroids(X, K)

# intrative
for iter in range(iterations):
    # find closest centroid for each example
    idx = find_closest_centroids(X, centroids)
    
    # move centorids to new centroid
    centroids = compute_means(X, idx, K)
```



## Optimization

k均值所做的优化相当于最小化平方误差
$$
E = \sum^k_{i = 1}\sum_{x\in C_i}||x - \mu_i||_2^2\\
\mu_i = \frac 1 {|C_i|}\sum_{x\in C_i}x
$$
E表现出样例围绕当前质心组的紧密程度，E值越小则类别内的相似度越高。$||\ ||^2_2$这个写法代表L2范数的平方。

若设$\mu_{C_i}$为第$i$个样例对应的质心，上式也可以表示为
$$
E = \frac 1 m \sum_{i = 1}^m ||x_i - \mu_{C_i}||^2_2
$$

> 上式也有人称distortion function



从直观上理解k均值所做的最小化：在每轮迭代的第一个循环中，算法通过为每个样例匹配最接近的$\mu_i$来最小化损失函数，此时相当于最优化样例匹配的精确度；而在第二个循环中，算法通过取$C_i$中所有样例的均值，使$\mu_i$移动到当前类别的质心，此时相当于最优化聚类的离散度。

> 一些证明：
>
> 1. $\mu = \frac 1 m \sum_{i = 1}^m x_i$时，$E=\sum_{i = 1}^{m}||x_i - \mu||_2^2$最小
>
>    对$\mu$求导，
>    $$
>    \frac {\partial \sum_{i = 1}^{m}||x_i - \mu||_2^2} {\partial \mu} = \sum_{i=1}^n -2(x_i - \mu)\\
>    = -2\sum_{i = 1}^n x_i + 2 n\mu
>    $$
>    可知$\mu = \frac 1 m \sum_{i = 1}^m x_i$为函数驻点。因为E是严格凸的（$f(\frac {x_1 + x_2)} {2}) \leq \frac {f(x_1) + f(x_2)} 2$），因此$\mu = \frac 1 m \sum_{i = 1}^m x_i$为当前类别cost的唯一极小值点。
>
>    
>
> 2. 算法收敛性
>
>    根据函数式可知，$E\geq0$，且在迭代过程中，数列${E_n}$单调递减；根据单调有界准则，单调递减有下界，该数列一定收敛，$\lim_{n\to \infin}E$存在。
>
>    因此，k-means是一定会收敛的，不过对不同的初始值，收敛的结果可能不一样，所以初始化也是k-means的超参数。



## Initializing

初始质心的选择会影响最后收敛的结果，如图所示，第一种收敛效果明显更好一些

<img src="D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230809171811302.png" alt="image-20230809171811302" style="zoom:80%;" />

可以通过计算E来判断哪个收敛结果更好
$$
\over{
\underline{
\begin{array}{l}
\mathbf{for}\ i\ =\ 1\ \mathbf{to}\ randTimes: \\
\quad 随机初始化\mu\\
\quad 运行\ k-means, 计算\ E\\
\quad 选择E最小的
\end{array}
}}
$$

> randTimes通常为50~1000次，太多次数会比较费时间；太少可能找不到比较好的。



## Number of Clusters

k-means中k一定严格小于样例数m，因为当$k > m$的时候，样例数都不够均分给每个类别。

具体要划分多少类别有时会很难确定，例如下图，既可以说有两类，也可以说有四类。

![image-20230811101015945](D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230811101015945.png)



一种选择k值的方式是elbow method（肘方法），即尝试多个k值，绘制k-Cost曲线，选择下降速度变得非常慢的那个k值。因为曲线通常很像弯曲的手臂，且被选中的k值位于“手肘”处，所以叫elbow method

![image-20230811101521466](D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230811101521466.png)



当然，对很多应用来说，k-Cost曲线往往没有明显的肘点；所以elbow method实际上没那么常用。

![image-20230811101642103](D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230811101642103.png)

> 不可以通过$\arg \min_k Cost$的方式来选择k值，因为更多的分类的Cost肯定会更小。



选择k值也应该更多考虑后续的应用需求，例如，假设要聚类衬衫的尺寸，想要区分S, M, L三种，则可以选择k = 3；如果想要划分的型号为5种（XS, S, M, L, XL），则可以考虑k = 5。当然，在这个例子中，设计不同版型的衬衫以及运输的开销也需要被考虑在内。

![image-20230811102300681](D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230811102300681.png)



# Implements of k-means

## Find closest centroids

```py
# UNQ_C1
# GRADED FUNCTION: find_closest_centroids

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): k centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """

    # Set K
    K = centroids.shape[0]
    M, N= X.shape

    # You need to return the following variables correctly
    idx = np.zeros(M, dtype=int)

    ### START CODE HERE ###
    # reshape for broadcast
    X = X.reshape((1, M, N))
    centroids = centroids.reshape((K, 1, N))
    
    diff = X - centroids
    # print(diff)
    dots = np.einsum('...i,...i->...', diff, diff)
    # 等价于下面的循环
    '''
    dots = np.zeros((K, M))
    for i in range(K):
        for j in range(M):
            dots[i, j] = np.dot(diff[i, j], diff[i, j])
    ''' 
    # print(dots)
    # min idx
    idx = np.argmin(dots, axis = 0)
            
    ### END CODE HERE ###
    
    return idx
```

> reshape整理广播维度，这里将数据整理为(k, m, n)
>
> `np.argmin`返回array中对应轴向最小值的索引。
>
> `np.einsum()`计算`Einstein summation convention`，一种计算线性代数计算的表示方法。`dots = np.einsum('...i,...i->...', diff, diff)`中的`'...i,...i->...'`是说明符，意思是分别取两个数组的最后一维进行内积，并返回整个结果数组。
>
> `->`用于显式指明输出格式，`...`用于缺省维度，标记符`i`没有严格要求，也可以改成其他标记符，例如
>
> ```py
> >>> norm = np.einsum('...i,...i->...', dif, dif)
> >>> norm
> array([[  6,  86, 294, 630],
>        [  6,  54, 230, 534]])
>        
> >>> norm = np.einsum('...c,...c->...', dif, dif)
> >>> norm
> array([[  6,  86, 294, 630],
>        [  6,  54, 230, 534]])
> ```
>
> `einsum()`一些常见的其他操作
>
> ```py
> # 求转置
> >>> c = np.arange(6).reshape(2,3)
> >>> c
> array([[0, 1, 2],
>        [3, 4, 5]])
> >>> np.einsum('ij->ji', c)
> array([[0, 3],
>        [1, 4],
>        [2, 5]])
> ```
>
> ```py
> # 提对角线
> a = np.arange(25).reshape(5,5)
> >>> a
> array([[ 0,  1,  2,  3,  4],
>        [ 5,  6,  7,  8,  9],
>        [10, 11, 12, 13, 14],
>        [15, 16, 17, 18, 19],
>        [20, 21, 22, 23, 24]])
> >>> np.einsum('ii->i', a)
> array([ 0,  6, 12, 18, 24])
> ```
>
> ```py
> # 求迹（tr），即对角线和
> np.einsum('ii', a)
> >>> np.einsum('ii->...', a)
> 60
> >>> np.trace(a)
> 60
> ```



## Computing centroid means

```py
# UNQ_C2
# GRADED FUNCTION: compute_centpods

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    ### START CODE HERE ###
    for i in range(K):
        c_k = X[idx == i]
        centroids[i] = c_k.mean(axis = 0)
    ### END CODE HERE ## 
    
    return centroids
```



## Random initialization

```py
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    Args:
    X (ndarray): Data points 
    K (int):     number of centroids/clusters
	Returns:
    centroids (ndarray): Initialized centroids
	"""

	# Randomly reorder the indices of examples
	randidx = np.random.permutation(X.shape[0])
	
	# Take the first K examples as centroids
	centroids = X[randidx[:K]]
	
	return centroids
```

> `np.random.permutation()`随机洗牌所有的样例索引，`[:k]`选取前k个随机样例。



# Example: Compress image

尝试使用k-means来压缩图像。具体的思路为：

1. 将每个像素视为一个样例，每个样例是一个RGB值
2. 通过k-means统计出16个最主要的颜色（质心）
3. 将所有的颜色替换为这16个颜色



示例图像

![image-20230811155243316](D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230811155243316.png)


## Dataset

```py
# load image
img = plt.imread('bird.png')
img.shape
'''
(128, 128, 3)
'''
```

```py
# precessing data
img = img/255
X_img = np.reshape(img, (img.shape[0]*img.shape[1], 3))
```



## Comress img

```py
K = 16
max_iters = 10
# init centroids
centroids = kMeans_init_centroids(X_img, K)

# run k-means
centroids, idx = run_kMeans(X_img, centroids, max_iters)
```

```py
X_recovered = centroids[idx, :]
X_recovered = np.reshape(X_recovered, img.shape)
```



## plot

```py
fig, ax = plt.subplots(1, 2, figsize=(8,8))
plt.axis('off')

ax[0].imshow(original_img*255)
ax[0].set_title('Original')
ax[0].set_axis_off()

# Display compressed image
ax[1].imshow(X_recovered*255)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
```

![image-20230811160129697](D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230811160129697.png)

k=64，迭代30轮的结果

![image-20230811160429975](D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230811160429975.png)

图像大小对比

![image-20230811160512443](D:\CS\Machine Learning\Course_3_Unsupervised Learning\1_Clustering.assets\image-20230811160512443.png)



# 参考

- 吴恩达《机器学习2022》
- 西瓜书
- 南瓜书

