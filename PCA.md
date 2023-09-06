PCA 降维是一种有损压缩方式，对任意$\mathbb R^n$区间内的点$x$，将其压缩到$\mathbb R^l$空间，压缩的结果为$c$，则有
$$
x \approx g(f(x))\\
c = f(x)
$$
设解码函数为$D \in \mathbb R^{n \times l}$，则有$g(c) = Dc$

对每个点$x$，希望解码后的值尽可能与$x$相同，两者的误差可以表示为
$$
err = ||x - g(c)||_2^2
$$
如果能够找到使$err$最小的$c$，则此时的c为最优编码，记$c*$，有
$$
\begin{array}{l}
c* = \arg \min_c ||x - g(c)||_2^2\\
= \arg \min_c (x - g(c))^T(x - g(c))\\
= \arg \min_c x^Tx - 2x^Tg(c) + g(c)^Tg(c)\\
\because \arg \min_c 与 x无关\\
\therefore c* = \arg \min_c -2x^Tg(c) + g(c)^Tg(c)\\
代入g(c) = Dc, 有\\
c* = \arg \min_c -2x^TDc + c^Tc\\
\because \nabla_c (-2x^TDc + c^Tc) = -2D^Tx + 2c\\
\therefore 令上式等于0，有\\
c = D^Tx\\
即最优编码只需要使用f(x) = D^Tx, PCA操作可以改写为g(f(x)) = DD^Tx
\end{array}
$$
 

已知对单个点$x$，其对应矩阵$D$的最优编码为$c* = D^Tx$，接下来应该筛选合适的D，使得D能在所有的点上取得最佳效果

设重构后的点为$r(x) = DD^Tx$，则对每个点$x$，需要考虑每个维数上每个点组成的误差矩阵，可以使用Forbenius范数来衡量矩阵的大小

> 对矩阵A，衡量其大小的Forbenius范数为
> $$
> ||A||_F = \sqrt{\sum_{i, j}A_{i, j}^2}
> $$

综上，最优编码矩阵$D*$可以表示为
$$
\begin{array}{l}
D*= \arg \min_D \sqrt{\sum_{i,j}\left(x^{(i)}_j - r(x^{(i)}_j)\right)^2}\\
\end{array}
$$
其中，$i$代表当前的样例，j代表当前的编码维数。