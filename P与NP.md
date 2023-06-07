经常被提到的一个术语，因为机器学习问题大多是NP难的。



# 概念定义

**决定性问题（Decision Problem）**：指输出只有“是”或“否”的问题。

**P问题**：能够在多项式时间（Polynomial time）内解决的决定性问题，即$m(n) = O(n^k)$，k为常量。

**多项式时间（Polynomial time）**：问题的计算时间$m(n)$不大于问题规模n的多项式倍数，即对于规模为n的输入，最坏情况下运行时间为$O(n^k)$。

**NP问题**：能够在多项式时间内验证是否正确，但不能保证在多项式时间内求解的决定性问题。所有的P问题同时也是NP类问题，因为P问题不需要验证就可以在多项式时间内解决。

**NPC（NP Complete）NP完全问题**：如果一个NP问题和其他NP问题一样”不易解决“，则称为NP完全问题。

> 目前的理解先停在NP不易解决吧，《算法导论》对怎么证明不易解决有说明，后面会补充。

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/P_np_np-complete_np-hard.svg/300px-P_np_np-complete_np-hard.svg.png" alt="img" style="zoom:200%;" />

