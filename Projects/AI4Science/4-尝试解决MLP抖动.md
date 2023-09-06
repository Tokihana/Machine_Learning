# 尝试过拟合单条数据

如果能够过拟合，说明网络结构应该是没问题的

首先尝试1000, 500, 100, 1

![image-20230828202025970](D:\CS\Machine Learning\Projects\AI4Science\4-尝试解决MLP抖动.assets\image-20230828202025970.png)

mae抖动很大且没法逼近0

再试试500, 500, 1

![image-20230828202240871](D:\CS\Machine Learning\Projects\AI4Science\4-尝试解决MLP抖动.assets\image-20230828202240871.png)

这次好像好一些，可能网络浅一些更好？

试试1000，1，没法拟合，还是得深一点

![image-20230828202930190](D:\CS\Machine Learning\Projects\AI4Science\4-尝试解决MLP抖动.assets\image-20230828202930190.png)

弄个4层的1000，500，500，100，1

![image-20230828203154084](D:\CS\Machine Learning\Projects\AI4Science\4-尝试解决MLP抖动.assets\image-20230828203154084.png)

再深一层1500，500，500，100，100，1

![image-20230828203946536](D:\CS\Machine Learning\Projects\AI4Science\4-尝试解决MLP抖动.assets\image-20230828203946536.png)

差不多能拟合单个数据了，试着上整个数据集吧



- 通常来说，第一层揭示线性关系
- 第二层非线性
- 第三层解释更加隐含的关系，常用于autoencoder，因此3~4隐层可能更好些



# 尝试修改loss

我一直觉得评估指标为mae就应该用mae做loss，结果今天换了个mse，发现模型抖动大幅度降低，且mae也同步下降了！！

用1024/1024/1024/1的网络跑200epoch，测试不同的loss

mse在小数据集上表现还可以，能逼近0，但是大数据集就不行了

mae抖动很厉害

hinge会梯度归零

log_cosh和mae一样，抖动剧烈，且没法逼近0

huber也还行，不过没mse效果好

目前来看mse应该是效果最好的
