# pandas：用来处理表格数据
import pandas as pd
# statsmodels：用来做线性回归建模
import statsmodels.api as sm
# 专门用来计算 VIF（方差膨胀因子）的工具
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 本地读取文件
# 把Carseats.csv放到 D:\python课设\Carseats.csv
carseats = pd.read_csv("Carseats.csv")

# 你的回归公式，完全不变！
my_formula = "Sales ~ Price + Income + Advertising + C(ShelveLoc)"

# 构建 OLS 普通最小二乘回归模型
model = sm.OLS.from_formula(my_formula, data=carseats)
# 拟合模型（自动计算系数、p值、R²等所有指标）
result = model.fit()

# 输出回归结果
print("===== 多元线性回归完整报告 =====")
print(result.summary())

# 计算 VIF（多重共线性检验）
# 提取模型里所有自变量的数据
X = result.model.exog
# 提取所有自变量的名字
col_names = result.model.exog_names

# 挨个计算每个变量的 VIF 值，存进表格
vif_table = pd.DataFrame()
vif_table["变量名"] = col_names
vif_table["VIF值"] = [variance_inflation_factor(X, i) for i in range(len(col_names))]

print("\n===== 多重共线性检验结果 =====")
print(vif_table)

# 作业解答：
# 1. 基准组是 Bad（差货架位置）
# 2. ShelveLoc[T.Good]系数为4.8359：在控制其他变量不变时，好货架比差货架平均多卖4.84单位
# 3. ShelveLoc[T.Medium]：中等货架相比Bad，销量提升约2.01单位
