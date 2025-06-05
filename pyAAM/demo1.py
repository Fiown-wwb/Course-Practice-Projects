import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.colors.groups import white

#避免乱码和中文不显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#引入文件
df=pd.read_excel("prodata.xlsx")

#不同CPU的占比
def CPU_percentage(df=df):
    # 去除CPU的空值
    df=df.dropna(subset=['CPU'])
    # 获取不同CPU的数量
    anCPU_num=df["CPU"].value_counts()
    num=len(df)#获取总数量
    anCPU_percentage=anCPU_num/num*100#计算占比
    #饼图绘制
    plt.figure(figsize=(10,10))
    plt.pie(anCPU_percentage, labels=anCPU_percentage.index, autopct='%1.1f%%', startangle=140)
    plt.title('不同CPU的占比饼图')
    plt.axis('equal')
    plt.savefig("E:/pythonProject/pyAAM/CPU_percentage.png",dpi=300)
    plt.show()

#各个价格段的电脑价格数量
def price_bracket(df=df):
    #获取最高最低的价格
    max_price=df["pPrice"].max()
    min_price=df["pPrice"].min()
    #计算合理的分区个数,并进行划分
    bins=range(int(min_price),int(max_price)+1000,1000)
    price_bins=pd.cut(df["pPrice"],bins=bins)
    #统计各个区间内的数量
    bracket_num=price_bins.value_counts().sort_index()
    #绘制图形
    plt.bar(bracket_num.index.astype(str),bracket_num)
    plt.xlabel('价格段')
    plt.xticks(rotation=45)
    plt.ylabel('电脑价格数量')
    plt.title('各个价格段的电脑价格数量条形图')
    plt.show()

#不同硬盘容量的平均价格
def average_HD(df=df):
    #获取不同硬盘容量下的电脑平均价格
    average_pHD = df.groupby('HD')['pPrice'].mean().round(2)
    # 绘制柱状图
    plt.figure(figsize=(10, 8))
    bars = plt.bar(average_pHD.index.astype(str), average_pHD,color='skyblue')
    plt.xlabel('硬盘容量大小')
    plt.xticks(rotation=45)
    plt.ylabel('平均价格')
    plt.title('不同硬盘容量的电脑平均价格')
    # 对条形进行显示优化
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
    plt.savefig("E:/pythonProject/pyAAM/average_HD.png",dpi=300)
    plt.show()

#是否自营的平均评论数
def own_pC(df=df):
    #获取是否自营的不同平均评论数
    average_pC = df.groupby('isOwn')['pCommitNum'].mean()
    #绘制横行柱状图
    plt.barh(average_pC.index, average_pC, height=0.2,color='red')
    plt.xlabel('平均评论数')
    plt.xticks(rotation=45)
    plt.ylabel('是否自营')
    plt.title('是否自营的平均评论数')
    #优化显示
    for index, value in enumerate(average_pC):
        plt.text(value, index, f'{value:.2f}', ha='left', va='center')
    plt.savefig("E:/pythonProject/pyAAM/own_pC.png",dpi=300)
    plt.show()

#是否新品的平均价格
def new_price(df=df):
    # 获取是否新品的平均价格
    average_price = df.groupby('isNew')['pPrice'].mean().round(2)
    # 绘制条形图
    plt.figure(figsize=(10, 8))
    bars = plt.bar(average_price.index, average_price,width=0.2,color='skyblue')
    plt.xlabel('是否新品')
    plt.xticks(rotation=45)
    plt.ylabel('平均价格')
    plt.title('是否新品的商品平均价格对比')
    # 显示优化
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
    plt.savefig("E:/pythonProject/pyAAM/new_price.png",dpi=300)
    plt.show()

#不同内存下的平均评论数
def MEN_pC(df=df):
    # 去除空值并获取不同内存下的评论数
    df = df.dropna(subset=['MEN'])
    average_pC = df.groupby('MEN')['pCommitNum'].mean().reset_index()
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(average_pC['MEN'], average_pC['pCommitNum'])
    plt.xlabel('内存大小')
    plt.ylabel('平均评论数')
    plt.title('不同内存大小下的平均评论数折线图')
    plt.xticks(rotation=45)
    plt.grid(True)
    #优化显示
    for x, y in zip(average_pC['MEN'], average_pC['pCommitNum']):
        plt.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(0, 5), ha='center')
    plt.savefig("E:/pythonProject/pyAAM/MEN_pC.png", dpi=300)
    plt.show()

# price_bracket()
# CPU_percentage()
# average_HD()
# own_pC()
# new_price()
# MEN_pC()