# Logistic的由来

logistic这个名称明显具有很大的迷惑性，西瓜书将其翻译为“对数几率函数”，也有人觉得这个词来源于logic，因此翻译为“逻辑函数”。从函数定义上来看
$$
g(z) = \frac {1} {1 + e^{-z}}
$$
该函数似乎和对数也没有明显的关联。



根据参考文献[1]("D:\CS\Machine Learning\阅读材料\MathAlive.pdf")，我们可以找到一些线索，显示出logistic这个名字，很可能来取自“log-like"，而在当时那个时期，所谓的”对数曲线（logarithm curve）“，其实是现在通称的指数曲线；即，提出该函数的作者，可能是想表达该函数在一定区间内，具有”类似指数函数“的性质，因此命名为”logistic function"。如图所示。

<img src="D:\CS\Machine Learning\7-Classification.assets\Courbe_logistique,_Verhulst,_1845.png" alt="Courbe_logistique,_Verhulst,_1845" style="zoom:50%;" />