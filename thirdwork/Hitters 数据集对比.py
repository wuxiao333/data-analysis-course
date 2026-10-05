import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import lasso_path, enet_path
from sklearn.datasets import load_diabetes


# 1. 读取数据
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 2. 训练3种正则模型，10折交叉验证
ridge_model = RidgeCV(cv=10, alphas=np.logspace(-3,4,100)).fit(X_train_scaled, y_train)
lasso_model = LassoCV(cv=10, alphas=100, max_iter=20000).fit(X_train_scaled, y_train)
elastic_model = ElasticNetCV(cv=10, l1_ratio=0.5, alphas=100, max_iter=20000).fit(X_train_scaled, y_train)


# 3. 预测，计算测试集RMSE
y_pred_ridge = ridge_model.predict(X_test_scaled)
y_pred_lasso = lasso_model.predict(X_test_scaled)
y_pred_elastic = elastic_model.predict(X_test_scaled)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
rmse_elastic = np.sqrt(mean_squared_error(y_test, y_pred_elastic))

count_nonzero_ridge = np.sum(np.abs(ridge_model.coef_) > 1e-6)
count_nonzero_lasso = np.sum(np.abs(lasso_model.coef_) > 1e-6)
count_nonzero_elastic = np.sum(np.abs(elastic_model.coef_) > 1e-6)


# 4. 打印结果
print("===== 模型结果汇总 =====")
print(f"Ridge: 测试集RMSE={rmse_ridge:.3f}, 保留变量数={count_nonzero_ridge}")
print(f"Lasso: 测试集RMSE={rmse_lasso:.3f}, 保留变量数={count_nonzero_lasso}")
print(f"ElasticNet: 测试集RMSE={rmse_elastic:.3f}, 保留变量数={count_nonzero_elastic}")
print(f"\nLasso最优惩罚系数lambda: {lasso_model.alpha_:.4f}")
print(f"ElasticNet最优惩罚系数lambda: {elastic_model.alpha_:.4f}")
print(f"Ridge最优惩罚系数lambda: {ridge_model.alpha_:.4f}")


# 5. 绘制系数路径图
alphas_lasso, coefs_lasso, _ = lasso_path(X_train_scaled, y_train, alphas=100)
alphas_elastic, coefs_elastic, _ = enet_path(X_train_scaled, y_train, l1_ratio=0.5, alphas=100)

alphas_ridge = np.logspace(-3,4,100)
coefs_ridge = []
for lam in alphas_ridge:
    temp_ridge = Ridge(alpha=lam).fit(X_train_scaled, y_train)
    coefs_ridge.append(temp_ridge.coef_)
coefs_ridge = np.array(coefs_ridge).T

fig, axes = plt.subplots(1,3, figsize=(18,5))

# Ridge
for line in coefs_ridge:
    axes[0].plot(alphas_ridge, line, alpha=0.6)
axes[0].set_xscale("log")
axes[0].set_title("Ridge Coefficient Path")
axes[0].set_xlabel(r"$\lambda$")
axes[0].set_ylabel("Coefficient")
axes[0].axvline(ridge_model.alpha_, linestyle="--", color="red", label="Best λ")
axes[0].legend()

# Lasso
for line in coefs_lasso:
    axes[1].plot(alphas_lasso, line, alpha=0.6)
axes[1].set_xscale("log")
axes[1].set_title("Lasso Coefficient Path")
axes[1].set_xlabel(r"$\lambda$")
axes[1].axvline(lasso_model.alpha_, linestyle="--", color="red", label="Best λ")
axes[1].legend()

# ElasticNet
for line in coefs_elastic:
    axes[2].plot(alphas_elastic, line, alpha=0.6)
axes[2].set_xscale("log")
axes[2].set_title("ElasticNet Coefficient Path")
axes[2].set_xlabel(r"$\lambda$")
axes[2].axvline(elastic_model.alpha_, linestyle="--", color="red", label="Best λ")
axes[2].legend()

plt.tight_layout()
plt.show()


# 6. Lasso的1SE法则计算
cv_mse_mean = lasso_model.mse_path_.mean(axis=1)
cv_mse_std = lasso_model.mse_path_.std(axis=1)

min_cv_error = cv_mse_mean.min()
threshold = min_cv_error + cv_mse_std[np.argmin(cv_mse_mean)]

valid_index = np.where(cv_mse_mean <= threshold)[0]
index_1se = valid_index[-1]
lambda_1se = lasso_model.alphas_[index_1se]

print(f"\n===== Lasso 1SE法则结果 =====")
print(f"最小交叉验证MSE: {min_cv_error:.3f}")
print(f"1SE阈值: {threshold:.3f}")
print(f"lambda.1se = {lambda_1se:.4f}")

lasso_1se = Lasso(alpha=lambda_1se, max_iter=20000).fit(X_train_scaled, y_train)
count_1se_nonzero = np.sum(np.abs(lasso_1se.coef_) > 1e-6)
y_pred_1se = lasso_1se.predict(X_test_scaled)
rmse_1se = np.sqrt(mean_squared_error(y_test, y_pred_1se))

print(f"1SE模型：测试RMSE={rmse_1se:.3f}, 保留变量数={count_1se_nonzero}")
