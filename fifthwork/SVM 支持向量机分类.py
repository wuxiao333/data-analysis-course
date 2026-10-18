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
