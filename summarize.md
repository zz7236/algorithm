# 机器学习综述

## 机器学习介绍
* 机器学习是什么:机器学习是专门研究计算机怎样模拟或实现人类的学习行为，以获取新的知识或技能，重新组织已有的知识结构使之不断改善自身的性能。
* 机器学习是怎么来的: 机器学习是在人工智能的基础上面经过人们不断的改进建立起来的
* 理论基础是：概率论、统计学、逼近论、凸分析、算法复杂度理论等
* 是为了解决什么问题：
 1. 需要进行大量手工调整或需要拥有长串规则才能解决的问题
 2. 问题复杂且有大量数据的问题

### 机器学习分类
1. 按学习方式分
   * 监督学习：按我的理解来说，就是指所训练的数据集有label，会提供相应的对错提示，通过不断地重复训练，使其找到给定的训练数据集中的某种模式或规律，当新的数据到来时，可以根据这个函数预测结果。
   * 非监督学习：按我的理解来说，就是指所训练的数据集没有label，是通过学习其中的结构特征，来判读哪些数据比较相似。非监督学习目标不是告诉计算机怎么做，而是让它去学习怎样做事情。
   * 半监督学习：是监督学习和半监督学习的结合，所训练的数据集中部分有标签，部分没有标签。
2. 按任务类型分
   * 回归：从一组数据出发，确定某些变量之间的定量关系式，即建立数学模型并估计未知参数，它的目的是预测数值型的目标值，是有监督学习。
   * 分类：为了确定一个点的类别，具体有哪些类别是已知的，是有监督学习。
   * 聚类：具体有哪些类别是未知的，是无监督学习。

   * 生成模型：关注数据是如何生成的
   * 回归模型：关注类别之间的差别
   * 详解请见[生成模型和判别模型的区别](https://blog.csdn.net/qq_20011607/article/details/81744614)

### 机器学习方法三要素
1. 模型：用来描述客观世界的数学模型
2. 策略：在我的理解中就是模型的比较标准和选取标准，即通过loss funtion:
    * 绝对值损失函数$L(y,f(x))=|y-f(x)|$ 
    * 平方损失函数$L(y,f(x))=(y-f(x))^2$
    * log对数损失函数$L(y,f(x))=log(1+e^{-yf(x)})$
    * 指数损失函数$L(y,f(x))=exp(-yf(x))$
    * Hinge损失函数$L(w,b)=max\{0,1-yf(x)\}$
3. 算法：指学习模型的具体的计算方法，也就是求模型中的具体的参数的方法。一般会用到最优化得算法，比如梯度下降，牛顿法，拟牛顿法等。
    * 梯度下降：
    梯度下降是最常用的优化方法之一，它使用梯度的反方向$\nabla_\theta J(\theta)$更新参数$\theta$，使得目标函数$J(\theta)$达到最小化的一种优化方法，这种方法我们叫做梯度更新. 
        1. (全量)梯度下降
        $\theta=\theta-\eta\nabla_\theta J(\theta)$
        2. 随机梯度下降
        $\theta=\theta-\eta\nabla_\theta J(\theta;x^{(i)},y^{(i)})$
        3. 小批量梯度下降
        $\theta=\theta-\eta\nabla_\theta J(\theta;x^{(i:i+n)},y^{(i:i+n)})$
        4. 引入动量的梯度下降
        $\begin{cases}v_t=\gamma v_{t-1}+\eta \nabla_\theta J(\theta)  \\\theta=\theta-v_t\end{cases}$
        5. 自适应学习率的Adagrad算法
        $\begin{cases}g_t= \nabla_\theta J(\theta)  \\\theta_{t+1}=\theta_{t,i}-\frac{\eta}{\sqrt{G_t+\varepsilon}} \cdot g_t\end{cases}$
    * 牛顿法：
    牛顿法的原理是使用函数$f(x)$的泰勒级数的前面几项来寻找方程 $f(x)=0$的根。
    * 拟牛顿法：
    牛顿法虽然收敛速度快，但是需要计算海塞矩阵的逆矩阵  ，而且有时目标函数的海塞矩阵无法保持正定，从而使得牛顿法失效。为了克服这两个问题，人们提出了拟牛顿法。这个方法的基本思想是：不用二阶偏导数而构造出可以近似海塞矩阵（或海塞矩阵的逆）的正定对称阵。不同的构造方法就产生了不同的拟牛顿法。
4. 模型评估指标：
    * R2  $R2(y,\hat{y}) = 1-\frac{\sum_{i=1}^{N}(\hat{y}-y_i)^2}{\sum_{i=1}^{N}(\overline{y}-y_i)^2}$
    * MSE(Mean Squared Error)$MSE(y,f(x))=\frac{1}{N}\sum_{i=1}^{N}(y-f(x))^2$
    * MAE(Mean Absolute Error)$MSE(y,f(x))=\frac{1}{N}\sum_{i=1}^{N}|y-f(x)|$
    * RMSE(Root Mean Squard Error)
    $RMSE(y,f(x))=\frac{1}{1+MSE(y,f(x))}$
    * Top-k准确率
    $Top_k(y,pre_y)=\begin{cases}1, {y \in pre_y}  \\0, {y \notin pre_y}\end{cases}$
    * 混淆矩阵

    混淆矩阵|Predicted as Positive|Predicted as Negative
    |:-:|:-:|:-:|
    |Labeled as Positive|True Positive(TP)|False Negative(FN)|
    |Labeled as Negative|False Positive(FP)|True Negative(TN)|

    * 真正例(True Positive, TP):真实类别为正例, 预测类别为正例
    * 假负例(False Negative, FN): 真实类别为正例, 预测类别为负例
    * 假正例(False Positive, FP): 真实类别为负例, 预测类别为正例 
    * 真负例(True Negative, TN): 真实类别为负例, 预测类别为负例

    * 真正率(True Positive Rate, TPR): 被预测为正的正样本数 / 正样本实际数
    $TPR=\frac{TP}{TP+FN}$
    * 假负率(False Negative Rate, FNR): 被预测为负的正样本数/正样本实际数
    $FNR=\frac{FN}{TP+FN}$

    * 假正率(False Positive Rate, FPR): 被预测为正的负样本数/负样本实际数，
    $FPR=\frac{FP}{FP+TN}$
    * 真负率(True Negative Rate, TNR): 被预测为负的负样本数/负样本实际数，
    $TNR=\frac{TN}{FP+TN}$
    * 准确率(Accuracy)
    $ACC=\frac{TP+TN}{TP+FN+FP+TN}$
    * 精准率
    $P=\frac{TP}{TP+FP}$
    * 召回率
    $R=\frac{TP}{TP+FN}$
    * F1-Score
    $\frac{2}{F_1}=\frac{1}{P}+\frac{1}{R}$
5. 复杂度度量
    * 偏差与方差
        * 偏差(Bias):可以简单的理解为在射击时没有瞄准目标
        * 方差(Variance):可以理解为在射击的过程中手抖了
        如图所示：
    <img src="https://pic1.zhimg.com/80/v2-22287dec5b6205a5cd45cf6c24773aac_hd.jpg">

    * 过拟合与欠拟合
        * 过拟合:一般表示模型对数据的表现能力不足，通常是模型的复杂度不够，并且Bias高，训练集的损失值高，测试集的损失值也高.
        * 欠拟合:一般表示模型对数据的表现能力过好，通常是模型的复杂度过高，并且Variance高，训练集的损失值低，测试集的损失值高
    * 结构风险与经验风险
        * 经验风险:训练集的总损失定义为经验风险
        * 结构风险:结构化风险是为了缓解数据集过小而导致的过拟合现象，其等价于正则化，本质上反应的是模型的复杂度。认为经验风险越小，参数越多，模型越复杂，因此引入对模型复杂度的惩罚机制
    * 泛化能力:是指一个机器学习算法对于没有见过的样本的识别能力,也叫做举一反三的能力，或者叫做学以致用的能力
    * 正则化:是一种为了减小测试误差的行为,减少每一个参数的值,防止过拟合
5. 模型选择
    * 交叉验证
        - 交叉验证是一种模型的验证技术用于评估一个统计分析模型在独立数据集上的概括能力。主要用于在使用ML模型进行预测时，准确衡量一个模型在实际数据集上的效果。具体来说就是将整个数据集划分为若干部分，一部分用以训练模型、一部分用以测试最终模型的优劣、一部分验证模型结构和超参数。

    * k-折叠交叉验证
        - 假设训练集为S ，将训练集等分为k份:$\{S_1, S_2, ..., S_k\}$. 
        - 然后每次从集合中拿出k-1份进行训练
        - 利用集合中剩下的那一份来进行测试并计算损失值
        - 最后得到k次测试得到的损失值，并选择平均损失值最小的模型
6. 特征处理
    * 归一化:数据压缩到 [0,1] 或者 [−1,1] 区间上
    * 标准化:减去均值后再除以方差
    * 离散化:特征的连续值在不同的区间的重要性是不一样的，所以希望连续特征在不同的区间有不同的权重，实现的方法就是对特征进行划分区间，每个区间为一个新的特征。常用做法，就是先对特征进行排序，然后再按照等频离散化为N个区间。
    * one-hot编码：又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都有它独立的寄存器位，并且在任意时候，其中只有一位有效。即，只有一位是1，其余都是零值。
8. 模型调优
    * 网格搜索寻优:穷举搜索,在所有候选的参数选择中，通过循环遍历，尝试每一种可能性，表现最好的参数就是最终的结果
    * 随机搜索寻优:并未尝试所有参数值，而是从指定的分布中采样固定数量的参数设置
    * 贝叶斯优化算法:考虑了上一次参数的信息，从而更好的调整当前的参数。