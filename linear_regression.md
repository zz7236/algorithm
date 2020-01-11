# 线性回归

## 线性回归的原理
* 线性回归试图学得一个通过属性的线性组合来进行预测的函数。
* 有数据集$\{(x_1,y_1),(x_2,y_2),...,(x_n,y_n)\}$,其中,$x_i = (x_{i1};x_{i2};x_{i3};...;x_{id}),y_i\in R$<br> 其中n表示变量的数量，d表示每个变量的维度。  
* 可以用以下函数来描述y和x之间的关系：
    $f(x) = \sum_{i=0}^{d}\theta_ix_i$
* 即我们要通过对数据的学习，得到参数$\theta$

## 线性回归损失函数、代价函数、目标函数
* 损失函数(Loss Function)：度量单样本预测的错误程度，损失函数值越小，模型就越好。
* 代价函数(Cost Function)：度量全部样本集的平均误差。
* 目标函数(Object Function)：代价函数和正则化函数，最终要优化的函数。在本问题中为
$\underset{f\in F}{min}\, \frac{1}{n}\sum^{n}_{i=1}L(y_i,f(x_i))+\lambda J(F)$

## 最小二乘法
* 是一种数学优化技术。它通过最小化误差的平方和寻找数据的最佳函数匹配

``` python
class LR_LS():
    def __init__(self):
        self.w = None      
    def fit(self, X, y):
        # 最小二乘法矩阵求解
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) 
    def predict(self, X):
        # 用已经拟合的参数值预测新自变量
        y_pred = X.dot(self.w)
        return y_pred
```

## 优化方法
* 梯度下降:在我的理解中，梯度下降就是让梯度中所有偏导函数都下降到最低点的过程

``` python
class LR_GD():
    def __init__(self):
        self.w = None     
    def fit(self,X,y,alpha=0.02,loss = 1e-10): # 设定步长为0.002,判断是否收敛的条件为1e-10
        y = y.reshape(-1,1) #重塑y值的维度以便矩阵运算
        [m,d] = np.shape(X) #自变量的维度
        self.w = np.zeros((d)).reshape(d,1) #将参数的初始值定为0
        tol = 1e5
        while tol > loss:
            self.w = self.w - alpha * X.T.dot(X.dot(self.w) - y) * 2. / m
            tol = np.sum((X.dot(self.w) - y) ** 2) / m

    def predict(self, X):
        # 用已经拟合的参数值预测新自变量
        y_pred = X.dot(self.w)
        return y_pred
```
## 线性回归的评价指标
* 均方误差(MSE)

``` python
def MSE(y_true, y_predict):
    MSE = np.sum((y_true - y_predict) ** 2) / len(y_true)
```
* 均方根误差(RMSE)
``` python
def RMSE(y_true, y_predict):
    RMSE = np.sum((y_true - y_predict) ** 2) / len(y_true) ** 0.5 
```
* 平均绝对误差(MAE)
``` python
def MAE(y_true, y_predict):
    MAE = np.sum(abs(y_true - y_predict)) / len(y_true)
```
* R2
``` python
def r2_score(y_true, y_predict):
    MSE = np.sum((y_true - y_predict) ** 2) / len(y_true)
    return 1 - MSE / np.sum((y_true - np.var(y_true)) ** 2) / len(y_true)
```

* 注
MSE、RMSE、MAE无法消除量纲不一致而导致的误差值差别大的问题，回归模型可以成功解释的数据方差部分在数据固有方差中所占的比例，越接近1，表示可解释力度越大，模型拟合的效果越好。

* [sklearn.linear_model.LinearRegression详解](https://blog.csdn.net/weixin_39175124/article/details/79465558)
``` python
lr = LinearRegression(fit_intercept=True)
# 训练模型
lr.fit(x,y)
print("估计的参数值为：%s" %(lr.coef_))
# 计算R平方
print('R2:%s' %(lr.score(x,y)))
# 任意设定变量，预测目标值
x_test = np.array([2,4,5]).reshape(1,-1)
y_hat = lr.predict(x_test)
print("预测值为: %s" %(y_hat))
```