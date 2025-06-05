import pandas as pd
inputfile = pd.ExcelFile('./data/data.xlsx')
df = inputfile.parse('Sheet1')

data_A = df['肝气郁结证型系数'].sort_values()
data_A_len = len(data_A) // 4
A_level = ['A1', 'A2', 'A3', 'A4']
A_list = []
for i in range(4):
    start = i * data_A_len
    end = (i + 1) * data_A_len if i < 3 else len(data_A)
    A = data_A.iloc[start:end]
    A_list.extend([A_level[i]] * len(A))
df['肝气郁结证型系数—分段'] = pd.Series(A_list, index=data_A.index)

data_B = df['热毒蕴结证型系数'].sort_values()
data_B_len = len(data_B) // 4
B_level = ['B1', 'B2', 'B3', 'B4']
B_list = []
for i in range(4):
    start = i * data_B_len
    end = (i + 1) * data_B_len if i < 3 else len(data_B)
    B = data_B.iloc[start:end]
    B_list.extend([B_level[i]] * len(B))
df['热毒蕴结证型系数—分段'] = pd.Series(B_list, index=data_B.index)

data_C = df['冲任失调证型系数'].sort_values()
data_C_len = len(data_C) // 4
C_level = ['C1', 'C2', 'C3', 'C4']
C_list = []
for i in range(4):
    start = i * data_C_len
    end = (i + 1) * data_C_len if i < 3 else len(data_C)
    C = data_C.iloc[start:end]
    C_list.extend([C_level[i]] * len(C))
df['冲任失调证型系数—分段'] = pd.Series(C_list, index=data_C.index)

data_D = df['气血两虚证型系数'].sort_values()
data_D_len = len(data_D) // 4
D_level = ['D1', 'D2', 'D3', 'D4']
D_list = []
for i in range(4):
    start = i * data_D_len
    end = (i + 1) * data_D_len if i < 3 else len(data_D)
    D = data_D.iloc[start:end]
    D_list.extend([D_level[i]] * len(D))
df['气血两虚证型系数—分段'] = pd.Series(D_list, index=data_D.index)

data_E = df['脾胃虚弱证型系数'].sort_values()
data_E_len = len(data_E) // 4
E_level = ['E1', 'E2', 'E3', 'E4']
E_list = []
for i in range(4):
    start = i * data_E_len
    end = (i + 1) * data_E_len if i < 3 else len(data_E)
    E = data_E.iloc[start:end]
    E_list.extend([E_level[i]] * len(E))
df['脾胃虚弱证型系数—分段'] = pd.Series(E_list, index=data_E.index)

data_F = df['肝肾阴虚证型系数'].sort_values()
data_F_len = len(data_F) // 4
F_level = ['F1', 'F2', 'F3', 'F4']
F_list = []
for i in range(4):
    start = i * data_F_len
    end = (i + 1) * data_F_len if i < 3 else len(data_F)
    F = data_F.iloc[start:end]
    F_list.extend([F_level[i]] * len(F))
df['肝肾阴虚证型系数—分段'] = pd.Series(F_list, index=data_F.index)

new_path = './data/data_new.xlsx'
df.to_excel(new_path, index=False)