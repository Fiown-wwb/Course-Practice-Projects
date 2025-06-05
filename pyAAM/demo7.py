import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']            # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False              # 用来正常显示负号
from scipy.cluster.hierarchy import linkage, dendrogram

# 参数初始化
outputfile = './output/AgglomerativeCluster.xlsx'        # 保存结果的文件名

## 1. 导入数据
wine = load_wine()
# print(wine)
# print(wine.target_names)
# print(wine.feature_names)
data = wine.data
label = wine.target
# print(data.shape)
# print(label.shape)


## 2. 数据特征工程
# 2.1.数据特征值标准化
sc = StandardScaler()
sc.fit(data)
dataStd = sc.transform(data)


## 3. 构建模型
# 3.1.创建模型
ac = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
# 3.2.训练模型
ac.fit(dataStd)
# 3.3.查看模型结果
print(ac.get_params())          # 获取模型参数
print(ac.labels_)               # 获取模型分类结果
# 3.4.将聚类结果导出至本地存储
dfdata = pd.DataFrame(wine.data, columns=wine.feature_names)
dflabel = pd.DataFrame(wine.target, columns=['label'])
labelPred = ac.fit_predict(dataStd)
dflabelPred = pd.DataFrame(labelPred, columns=['labelPred'])
df = pd.concat([dfdata, dflabel, dflabelPred], axis=1)
df.to_excel(outputfile)
dfsta = pd.crosstab(df['label'], df['labelPred'], margins=True)
print(dfsta)

## 4. 模型效果可视化
# 4.1.聚类效果对比
# 准备数据
x = range(label.shape[0])
# 准备画布
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9, 9))
# 预测类别画图
axs[0].scatter(x, labelPred, c=labelPred, marker='o')
axs[0].set_title('Predict cluster')
axs[0].set_xlabel('samples')
axs[0].set_ylabel('Predict label')
# 真实类别画图
axs[1].scatter(x, label, c=label, marker='s')
axs[1].set_title('True cluster')
axs[1].set_xlabel('samples')
axs[1].set_ylabel('label')
# 保存图表
plt.savefig("./img/AgglomerativeCluster.png")
# 4.2.绘制层次聚类树状图
plt.figure(figsize=(20, 6))
Z = linkage(dataStd, method='ward', metric='euclidean')
'''
层次聚类编码为一个linkage矩阵 Z。共有四列组成：
    第一字段与第二字段分别为聚类簇的编号，在初始距离前每个初始值被从0~n-1进行标识，每生成一个新的聚类簇就在此基础上增加一对新的聚类簇进行标识，
    第三个字段表示前两个聚类簇之间的距离；
    第四个字段表示新生成聚类簇所包含的元素的个数。
'''
p = dendrogram(Z)    # 绘制层次聚类树状图
plt.savefig("./img/AgglomerativeClusterTree.png")

