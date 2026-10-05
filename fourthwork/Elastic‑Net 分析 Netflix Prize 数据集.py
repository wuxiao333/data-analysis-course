import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix
from scipy.linalg import svd

warnings.filterwarnings("ignore")
np.random.seed(42)

# 模拟Netflix电影评分数据
n_users = 500
n_movies = 200
n_samples = 25000

user_ids = np.random.randint(0, n_users, size=n_samples)
movie_ids = np.random.randint(0, n_movies, size=n_samples)

ratings = np.random.normal(loc=3.2, scale=1.1, size=n_samples)
ratings = np.clip(ratings, 1, 5)
ratings = np.round(ratings, 0)

df = pd.DataFrame({
    "user_id": user_ids,
    "movie_id": movie_ids,
    "rating": ratings
})

print("===== 模拟Netflix数据集信息 =====")
print(f"总样本条数：{df.shape[0]}")
print(f"总用户数量：{df['user_id'].nunique()}")
print(f"总电影数量：{df['movie_id'].nunique()}")
print(f"评分最小值：{df['rating'].min()}，最大值：{df['rating'].max()}")
print("\n前5行数据：")
print(df.head())

# 数据预处理 One‑Hot编码
enc = OneHotEncoder(sparse_output=True)
X_sparse = enc.fit_transform(df[["user_id", "movie_id"]])
y = df["rating"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_sparse, y, test_size=0.2, random_state=42
)

print(f"\n经过One‑Hot之后总特征维度：{X_sparse.shape[1]}")

# ElasticNet网格搜索调参
param_grid = {
    "alpha": [0.01, 0.1, 1, 5, 10],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}

en_model = ElasticNet(random_state=42)
grid_search = GridSearchCV(
    estimator=en_model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5
)
grid_search.fit(X_train, y_train)
best_en = grid_search.best_estimator_

print("\n==== 网格搜索找到的最优参数 ====")
print(grid_search.best_params_)

# 计算RMSE
def calculate_rmse(真实值, 预测值):
    mse = mean_squared_error(真实值, 预测值)
    rmse = np.sqrt(mse)
    return rmse

# Elastic‑Net预测
y_pred_en = best_en.predict(X_test)
y_pred_en = np.clip(y_pred_en, 1, 5)
rmse_en = calculate_rmse(y_test, y_pred_en)

# Ridge预测
ridge_model = Ridge(alpha=grid_search.best_params_["alpha"], random_state=42)
ridge_model.fit(X_train, y_train)
y_pred_ridge = np.clip(ridge_model.predict(X_test), 1, 5)
rmse_ridge = calculate_rmse(y_test, y_pred_ridge)

# Lasso预测
lasso_model = Lasso(alpha=grid_search.best_params_["alpha"], random_state=42)
lasso_model.fit(X_train, y_train)
y_pred_lasso = np.clip(lasso_model.predict(X_test), 1, 5)
rmse_lasso = calculate_rmse(y_test, y_pred_lasso)

print("\n===== 各个模型测试集RMSE对比，数值越小预测越准 =====")
print(f"Elastic‑Net RMSE: {rmse_en:.4f}")
print(f"Ridge(L2) RMSE:   {rmse_ridge:.4f}")
print(f"Lasso(L1) RMSE:   {rmse_lasso:.4f}")

# 稀疏统计
coef_list = best_en.coef_
zero_threshold = 1e-6
count_zero = np.sum(np.abs(coef_list) < zero_threshold)
count_total = len(coef_list)
sparse_percent = count_zero / count_total

print(f"\nElastic‑Net稀疏统计：")
print(f"全部特征数量：{count_total}")
print(f"被压缩为0的特征数量：{count_zero}")
print(f"稀疏率：{sparse_percent:.2%}")

# SVD降维 + Elastic‑Net
print("\n======== SVD降维 + Elastic‑Net 实验 ========")
n_u = df["user_id"].nunique()
n_m = df["movie_id"].nunique()

user_to_index = {}
for idx, uid in enumerate(df["user_id"].unique()):
    user_to_index[uid] = idx

movie_to_index = {}
for idx, mid in enumerate(df["movie_id"].unique()):
    movie_to_index[mid] = idx

row_list = []
col_list = []
value_list = []
for index, row in df.iterrows():
    u = row["user_id"]
    m = row["movie_id"]
    r = row["rating"]
    row_list.append(user_to_index[u])
    col_list.append(movie_to_index[m])
    value_list.append(r)

A_sparse = csr_matrix((value_list, (row_list, col_list)), shape=(n_u, n_m))
A = A_sparse.toarray()

k = 10
U, sigma, Vt = svd(A, full_matrices=False)
U_k = U[:, :k]
Vt_k = Vt[:k, :].T

svd_features = []
for index, row in df.iterrows():
    u_id = row["user_id"]
    m_id = row["movie_id"]
    user_vec = U_k[user_to_index[u_id], :]
    movie_vec = Vt_k[movie_to_index[m_id], :]
    combine_vec = np.concatenate([user_vec, movie_vec])
    svd_features.append(combine_vec)

X_svd = np.array(svd_features)

Xsvd_train, Xsvd_test, ysvd_train, ysvd_test = train_test_split(
    X_svd, y, test_size=0.2, random_state=42
)

en_svd = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
en_svd.fit(Xsvd_train, ysvd_train)
pred_svd = np.clip(en_svd.predict(Xsvd_test), 1, 5)
rmse_svd = calculate_rmse(ysvd_test, pred_svd)

print(f"SVD隐向量特征 + Elastic‑Net RMSE：{rmse_svd:.4f}")
print("说明：SVD做特征工程，用低维隐向量代替OneHot，解决维度爆炸")
