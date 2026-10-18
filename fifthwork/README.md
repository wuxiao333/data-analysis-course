\# SVM支持向量机 分类实验 

课程作业：使用SVM算法完成垃圾邮件二分类



\## 一、实验简介 SVM（支持向量机）是经典机器学习分类算法。

 本次任务：根据邮件的多项统计特征，自动区分【正常邮件 / 垃圾邮件】 

数据集：Spambase垃圾邮件数据集（UCI，Kaggle平台可检索到） 

\- 一共4601条邮件样本

\- 57个特征：邮件内词频、特殊符号、大写字母长度等统计信息

\- 标签含义：0=正常邮件，1=垃圾邮件



\## 二、实验5个核心步骤 

1. 加载数据集（代码在线读取数据源，无需手动下载csv文件） 
2. 划分数据集：训练集70%，测试集30% 
3. 特征标准化（SVM算法必须预处理，消除不同特征量纲差异） 
4. 创建RBF核SVM模型，使用训练集完成模型训练 
5. 在测试集进行预测，计算模型准确率，评估效果并查看支持向量



\## 三、环境依赖 

打开命令行，执行命令安装依赖库： 

\```bash

pip install pandas numpy scikit-learn



**完整 Python 代码**

```
# 导入需要的工具包
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ====================== 1.读取数据集 ======================
# 在线读取垃圾邮件数据集，不用手动下载文件
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
data = pd.read_csv(url, header=None)

# X：特征（用来判断邮件是不是垃圾邮件的57个指标）
# y：标签，0=正常邮件，1=垃圾邮件
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("总样本数量：", X.shape[0])
print("特征数量：", X.shape[1])
print("垃圾邮件占比：", np.mean(y).round(3))

# ====================== 2.划分训练集、测试集 ======================
# 训练集：用来训练模型（70%数据）
# 测试集：用来检验模型效果（30%数据）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ====================== 3.特征标准化【SVM必须做！】 ======================
# SVM对数字大小很敏感，统一缩放到相同范围
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 训练集：学习缩放规则+转换
X_test = scaler.transform(X_test)        # 测试集：只用训练好的规则转换

# ====================== 4.创建SVM模型并训练 ======================
# 使用RBF核函数，C=10，gamma=0.01
svm_model = SVC(kernel="rbf", C=10, gamma=0.01)
svm_model.fit(X_train, y_train)  # 喂训练数据，模型开始学习

# ====================== 5.模型预测 & 评估 ======================
y_pred = svm_model.predict(X_test) # 用训练好的模型预测测试集

print("\n==== SVM模型预测结果 ====")
acc = accuracy_score(y_test, y_pred)
print("测试集准确率：", round(acc, 4))
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# ====================== 6.查看支持向量（SVM核心知识点） ======================
print("\n==== 支持向量信息 ====")
support_vec_num = svm_model.n_support_
print("正常邮件类支持向量数量：", support_vec_num[0])
print("垃圾邮件类支持向量数量：", support_vec_num[1])
total_sv = sum(support_vec_num)
print("全部支持向量数量：", total_sv)
print("训练集总样本：", X_train.shape[0])
print("支持向量占训练样本比例：", round(total_sv / X_train.shape[0],3))
```



**本次运行输出结果**

```
总样本数量： 4601
特征数量： 57
垃圾邮件占比： 0.394
==== SVM模型预测结果 ====
测试集准确率： 0.9421
分类报告：
              precision    recall  f1-score   support
           0       0.94      0.97      0.95       804
           1       0.95      0.91      0.93       577
    accuracy                           0.94      1381
   macro avg       0.94      0.94      0.94      1381
weighted avg       0.94      0.94      0.94      1381
==== 支持向量信息 ====
正常邮件类支持向量数量： 391
垃圾邮件类支持向量数量： 360
全部支持向量数量： 751
训练集总样本： 3220
支持向量占训练样本比例： 0.233
```



**结果分析**

1. **准确率 94.21%** 模型在 1381 条测试邮件中，大约 94% 能正确区分正常邮件和垃圾邮件，分类效果很好。
2. 分类指标简单解释

- precision（精确率）：判定为垃圾邮件的样本里，真正是垃圾邮件的比例。
- recall（召回率）：所有真实垃圾邮件里，被模型成功识别出来的比例。
- f1-score：精确率和召回率的综合分数，越接近 1 代表效果越好。

> 正常邮件召回率 0.97，说明绝大多数正常邮件不会被误判为垃圾邮件； 垃圾邮件召回率 0.91，说明 91% 的垃圾邮件能被识别，少量漏判。

1. **支持向量（SVM 核心知识点）** 训练集一共 3220 条样本，只需要 751 条样本作为支持向量来画出分类边界，只占全部训练样本的 23.3%。 其他样本不会影响分类边界，这个特点叫做**SVM 稀疏性**，对应课件 SVM 理论。
2. 参数说明

- `kernel="rbf"`：径向基核函数，让 SVM 可以处理非线性的分类问题。
- C：惩罚系数。C 越大，尽量减少错分样本；C 越小，优先保证分类间隔更大。
- gamma：控制单个样本对周围数据的影响范围。