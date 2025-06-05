import numpy as np
import pandas as pd
import sklearn.metrics as sm
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler                # 数据标准化
from sklearn.model_selection import train_test_split            # 数据集分割


## 分割函数：不均衡样本数据集采用下采样处理
def undersampleSplit(df):
    '''
    针对样本数据采用下采样处理，将处理后样本数据分割为训练数据集和测试数据集
        下采样（undersample）：随机选择和异常样本一样多的正常数据和异常数据一同进行训练进行
    :param df: 样本数据
    :return:
        X_undersample_train     训练样本特征
        y_undersample_train     训练样本标签
        X_undersample_test      测试样本特征
        y_undersample_test      测试样本标签
    '''
    # 1.筛选异常样本
    num_records_fraud = len(df[df.Class == 1])
    fraud_indexes = df[df.Class == 1].index
    fraud_indexes = np.array(fraud_indexes)
    # 2.筛选正常样本
    normal_indexes = df[df.Class == 0].index
    # 3.随机选择和异常样本一样多的正常数据
    random_normal_indexes = np.random.choice(normal_indexes, num_records_fraud, replace=False)
    random_normal_indexes = np.array(random_normal_indexes)
    # 4.异常样本和正常样本（和异常样本一样多）组成新样本数据集
    under_sample_index = np.concatenate([fraud_indexes, random_normal_indexes])
    under_sample_data = df.iloc[under_sample_index, :]
    ## 5.新样本数据集分割：训练数据集、测试数据集
    X_under_sample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
    y_under_sample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']
    X_undersample_train, X_undersample_test, y_undersample_train, y_undersample_test = train_test_split(X_under_sample,
                                                                                                        y_under_sample,
                                                                                                        test_size=0.3,
                                                                                                        random_state=0)
    return (X_undersample_train, y_undersample_train, X_undersample_test, y_undersample_test)
if __name__ == "__main__":
    ## 1.导入数据
    dataPath = r'./data/creditcard.csv'
    df = pd.read_csv(dataPath)
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)
    ## 样本不平衡情况下，采用下采样将数据集分割（训练数据集、测试数据集）
    X_undersample_train, y_undersample_train, X_undersample_test, y_undersample_test = undersampleSplit(df)
    #实例化线性模型
    LR = LinearRegression()
    LR.fit(X_undersample_train, y_undersample_train)
    print("回归系数：", LR.coef_)
    print("常数项（截距）是：", LR.intercept_)
    height_prd = LR.predict(X_undersample_test)
    print("平均绝对值误差:", sm.mean_absolute_error(y_undersample_test, height_prd))
    print("中位绝对值误差:", sm.median_absolute_error(y_undersample_test, height_prd))
    print("平均平方误差:", sm.mean_squared_error(y_undersample_test, height_prd))
    print("R2得分:", sm.r2_score(y_undersample_test, height_prd))
    joblib.dump(LR,  "./model/LR.pkl")
    LR = joblib.load("./model/LR.pkl")
    Xis=np.array([[1.22965763450793,0.141003507049326,0.0453707735899449,1.20261273673594,0.191880988597645,0.272708122899098,-0.005159003,0.0812129398830894,0.464959994783886,-0.099254321,-1.416907243,-0.153825826,-0.751062716,0.16737196252175,0.0501435942254188,-0.443586798,0.00282051247234708,-0.61198734,-0.045575045,-0.219632553,-0.167716266,-0.270709726,-0.154103787,-0.780055415,0.75013693580659,-0.257236846,0.0345074297438413,0.00516776890624916,0]])
    Xis_pred=LR.predict(Xis)
    print(Xis_pred)

