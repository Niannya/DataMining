# 期末作业：电商用户分群

## 选题

方向 2：电商用户分群。

本项目计划使用 UCI Online Retail II 数据集，完成用户分群和关联规则挖掘，并根据结果给出电商运营建议。

## 数据集

数据集：Online Retail II  
来源：UCI Machine Learning Repository  
下载地址：https://archive.ics.uci.edu/dataset/502/online+retail+ii

请将以下任一文件放入 `FinalProject/data/raw/`：

- `online_retail_ii.zip`
- `online_retail_II.xlsx`
- `Online Retail.xlsx`
- 其他包含 Online Retail II 原始数据的 `.xlsx` 文件

## 初步流程

1. 数据读取
2. 缺失值和异常值处理
3. 构建用户 RFM 特征
4. 使用 K-Means 进行用户分群
5. 使用 Apriori 挖掘商品关联规则
6. 结合分群和规则结果提出运营建议
7. 撰写科技小论文

## 主要输出

| 文件 | 说明 |
| --- | --- |
| `data/processed/rfm_features.csv` | 用户 RFM 特征表 |
| `data/processed/customer_segments.csv` | 用户分群结果 |
| `data/processed/association_rules.csv` | 商品关联规则结果 |
| `reports/期末论文.md` | 期末小论文草稿 |

