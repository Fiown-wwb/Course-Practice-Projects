import pandas as pd
from sklearn.preprocessing import StandardScaler

#导入数据
df=pd.read_excel('dataraw.xlsx')

#数据清理
df = df.drop(columns='URL')
df = df.dropna()

#针对 Price 异常（远低于 Community average）的数据，以 Community average 填充
df.loc[df['Price'] < df['Community average'], 'Price'] = df['Community average']

#对 Total price、Price、Square 进行标准化处理，并将结果存到新列(Z-Score标准化)
scaler = StandardScaler()
scaled_Total = scaler.fit_transform(df[['Total price']])
df['Total price_scaled'] = scaled_Total
scaled_Price = scaler.fit_transform(df[['Price']])
df['Price_scaled'] = scaled_Price
scaled_Square = scaler.fit_transform(df[['Square']])
df['Square_scaled'] = scaled_Square

#将 floors 拆分为两列，分别是高低层类型和楼层
df[['高低层类型', '楼层']] = df['floors'].str.split(' ', expand=True, n=1)

#将 Elevator、Property rights for five years、Subway 的数值进行转换：0-否，1-是
replace_dict = {0: '否', 1: '是'}
df[['Elevator', 'Property rights for five years', 'Subway']] = df[['Elevator', 'Property rights for five years', 'Subway']].replace(replace_dict)

#保存文件
file_path = 'new_dataraw.xlsx'
df.to_excel(file_path, index=False)