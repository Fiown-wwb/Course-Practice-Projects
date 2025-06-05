import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def img2vector(filename):
    VectList = []
    with open(filename) as fin:
        lineStr = fin.readlines()
        for lineStri in lineStr:
            VectList.extend(lineStri.replace("\n", ""))
            returnVect=np.array(VectList)
    return returnVect

def getData(path):
    filelist = os.listdir(path)
    m = len(filelist)
    fileMat = np.zeros((m, 1024), dtype=int)
    labels = []
    for i in range(m):
        fileNameStr = filelist[i]
        class_num = int(fileNameStr.split('_')[0])
        labels.append(class_num)
        filepath = os.path.join(path, fileNameStr)
        fileMat[i, :] = img2vector(filepath)
    return fileMat, np.array(labels)

training_path = "./data/knn_data/trainingDigits/"
test_path = "./data/knn_data/testDigits/"

# 加载数据
X_train, y_train = getData(training_path)
print(f"已加载{len(y_train)}个训练样本数据")
X_test, y_test = getData(test_path)
print(f"已加载{len(y_test)}个测试样本数据")

# 模型训练
k = 4
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(X_train, y_train)

# 模型评估
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"手写数字分类准确率：{accuracy:.2f}%")

# 保存与加载模型
joblib.dump(knn, "./model/myKNN.pkl")
knn = joblib.load("./model/myKNN.pkl")

# 预测
Xis = np.array([X_test[89]])
Xis_pred = knn.predict(Xis)
print("预测值：", Xis_pred)