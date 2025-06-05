import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


if __name__ == "__main__":
    # 定义基本参数
    min_support = 0.01 # 最小支持度
    min_confidence = 0.2  # 最小置信度
    #1.导入数据
    inputfile = './data/US Superstore data Sub-Category list.csv'
    # 使用pandas读取数据
    df = pd.read_csv(inputfile, header=None)
    # 去除包含空值的行
    df = df.dropna(how='all')
    # print(df)
    #将数据装入数组中
    data = df.values.tolist()
    # print(data)

    #2.移除每行的列表中的空值
    clean_data = []
    for row in data:
        clean_row = []
        for item in row:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    clean_row.append(item)
        clean_data.append(clean_row)

    #3.数据特征工程，转换格式满足模型需求
    encoder = TransactionEncoder()
    trans_encoded = encoder.fit_transform(clean_data)
    df_encoded = pd.DataFrame(trans_encoded, columns=encoder.columns_)
    print(df_encoded.head())

    #4.挖掘强关联规则
    frequent_itemSets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    print("频繁项集:\n", frequent_itemSets)
    # 使用association_rules函数查找强关联规则
    rules = association_rules(frequent_itemSets, metric="confidence", min_threshold=min_confidence)
    print("关联规则:\n", rules[['antecedents', 'consequents','support', 'confidence']])