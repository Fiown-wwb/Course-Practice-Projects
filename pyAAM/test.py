import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
outputfile = './out/AgglomerativeCluster.xlsx'

#数据准备
excel_file = pd.ExcelFile('./data/consumption_data.xlsx')
df = excel_file.parse('Sheet1')
data = df.drop(columns=['Id'])

# 数据特征工程
sc = StandardScaler()
sc.fit(data)
dataStd = sc.transform(data)

# 构建模型
ac = AgglomerativeClustering(n_clusters=3, linkage='ward')
ac.fit(dataStd)
print(ac.get_params())
print(ac.labels_)

# 保存结果
dflabelPred = pd.DataFrame(ac.labels_, columns=['labelPred'])
df = pd.concat([df, dflabelPred], axis=1)
df.to_excel(outputfile)

# 模型效果可视化
#预测类别画图
x = range(df.shape[0])
plt.figure(figsize=(9, 6))  # 单独创建一个Figure对象
plt.scatter(x, ac.labels_, c=ac.labels_, marker='o')
plt.title('Predict cluster')
plt.xlabel('samples')
plt.ylabel('Predict label')
plt.savefig("./out/AgglomerativeCluster.png")
#层次聚类树状图
plt.figure(figsize=(20, 6))
Z = linkage(dataStd, method='ward', metric='euclidean')
p = dendrogram(Z)
plt.savefig("./out/AgglomerativeClusterTree.png")