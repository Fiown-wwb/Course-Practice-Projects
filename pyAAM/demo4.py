from prefixspan import PrefixSpan
import pandas as pd

inputfile = pd.ExcelFile('./data/data_new.xlsx')
df = inputfile.parse('Sheet1')

columns_name = ['肝气郁结证型系数—分段', '热毒蕴结证型系数—分段','冲任失调证型系数—分段', '气血两虚证型系数—分段','脾胃虚弱证型系数—分段', '肝肾阴虚证型系数—分段', 'TNM分期']
lists = df[columns_name].values.tolist()
# print(lists)
ps = PrefixSpan(lists)
min_support = 50
frequent_sequences = ps.frequent(min_support)
# print(frequent_sequences)
for i in range(1,5):
    for support, sequence in frequent_sequences:
        if str(sequence[-1]).startswith(f'H{i}'):
            print(f"支持度: {support}, 序列: {sequence}")