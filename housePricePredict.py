#加载Python库
import numpy as np
#加载数据预处理模块
import pandas as pd
#加载绘图模块
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
#导入网格搜索模块
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sns.set_style(style="darkgrid")

#读取数据
df = pd.read_csv("kc_house_data.csv")
print(df.head())

#属性信息
print(df.info())

#描述信息
print(df.describe())

#输出数据空值情况
print(df.isnull().any()) #这里主要调用DataFrame中的isnull方法进行属性空值检测
#删除原始数据中的索引id
df.drop(["id"],axis=1,inplace=True)
'''
#画图分析
#获取第三列开始往后的所有属性名称，由于第一列为序号，第二列为房屋记录时间，第三列房屋价格
#因此此处从第四列开始获取属性集。
x_vars=df.columns[3:]
#分别分析所取的属性与价格的分布关系图
for x_var in x_vars:
   df.plot(kind='scatter',x=x_var,y='price')  #设置绘图的行和列
plt.show()

#计算属性间的相关系数图
corr = df.corr()
#绘制属性相关系数的热力图
plt.figure(figsize=(16,8))
sns.heatmap(corr,annot=True,cmap="RdBu")
plt.show()

plt.figure(figsize=(16,8))
#配置下三角热力图区域显示模式
mask = np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set_style(style="white")
#对相关系数图进行下三角显示
sns.heatmap(corr,annot=True,cmap="RdBu",mask=mask)
plt.show()

plt.figure(figsize=(10,10))
#调用散点图模块，依据经纬度绘制散点图
plt.scatter(df.lat, df.long)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()

#绘制各区域具体的价格散点分布情况，了解每个区域的价格分布区间
plt.scatter(df.zipcode,df.price)
plt.title("Which is the pricey location by zipcode?")
plt.show()
'''

#使用线性回归模型进行数据分析
#选择用于进行回归分析的属性集，由于数据拆分函数对数据类型有要求，因此此处对属性集进行矩阵转换
#此处由于属性维度较少，使用部分属性可能会造成信息丢失，因此此处采用全量属性进行分析，
#但在实际问题中，若是属性维度较大，一般会依据前面相关系数的结果对属性集进行筛选。
X = df[['bedrooms', 'bathrooms', 'sqft_living',\
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',\
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',\
       'lat', 'long', 'sqft_living15', 'sqft_lot15']]
#选择价格作为回归更新的标签值
y = df['price']

#将数据拆分为训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

#调用数据标准化模块
sc = StandardScaler()
#对属性数据进行标准化处理
sc.fit(X_train)
#对训练数据属性集进行标准化处理
X_train= sc.transform(X_train)
#对测试数据属性集进行标准化处理
X_test = sc.transform(X_test)

#1线性回归
model = LinearRegression()
#采用线性回归进行模型训练
model.fit(X_train, y_train)
#let us predict
#获取模型预测结果
y_pred=model.predict(X_test)
#打印模型评分结果
print ("线性回归的准确率是",model.score(X_test, y_test))

#2随机森林
#配置模型中回归树的个数为500
model = RandomForestRegressor(n_estimators=500)
#采用随机森林回归模型进行模型训练
model.fit(X_train, y_train)
#采用随机森林回归模型进行预测
y_pred=model.predict(X_test)
#打印模型评分结果
print ("随机森林的准确率是",model.score(X_test, y_test))

#3GBDT回归模型
#配置GBDT回归模型的分类器个数
model = GradientBoostingRegressor(n_estimators=500)
#采用训练数据集进行模型训练
model.fit(X_train, y_train)
#采用测试数据集进行模型预测
y_pred=model.predict(X_test)
#输出模型评估值
print ("GBDT回归模型的准确率是",model.score(X_test, y_test))

#4最近邻回归模型
#配置最近邻回归模型参数
model = KNeighborsRegressor(n_neighbors=10)
#采用最近邻回归模型进行训练
model.fit(X_train, y_train)
#采用最近邻模型进行预测
y_pred=model.predict(X_test)
#打印最近邻回归模型评估值
print ("最近邻回归模型的准确率是",model.score(X_test, y_test))

#5梯度提升树模型
#配置梯度提升树模型参数，树的棵数
model = GradientBoostingRegressor(n_estimators=500)
#采用训练数据进行模型训练
model.fit(X_train, y_train)
#采用测试数据进行模型预测
y_predicted = model.predict(X_test)
#导入模型结果评估模块平均绝对误差，均方根误差和r2值
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#计算平均绝对误差，均方根误差，r2模型值
mean_absolute_error(y_test,y_predicted)
mean_squared_error(y_test,y_predicted)
r2_score(y_test,y_predicted)
#输出平均绝对误差，均方根误差，r2模型值
print('r2模型值',r2_score(y_test,y_predicted))
print('平均绝对误差',mean_absolute_error(y_test,y_predicted))
print('均方根误差',mean_squared_error(y_test,y_predicted))
'''
#优化
#采用网格搜索算法进行模型参数优化
model_gbr = GradientBoostingRegressor()
#对loss，min_samples_leaf，alpha三个参数值进行最优化网格搜索
parameters = {'loss': ['ls','lad','huber','quantile'],'min_samples_leaf': [1,2,3,4,5],'alpha': [0.1,0.3,0.6,0.9]}
#调用网格搜索模型进行最优化参数搜索
model_gs = GridSearchCV(estimator=model_gbr, param_grid=parameters, cv=5)
model_gs.fit(X_train,y_train)
#输出最优的模型评估值和模型参数值
print('Best score is:', model_gs.best_score_)
print('Best parameter is:', model_gs.best_params_)
'''
#采用最优参数进行数据建模分析
#配置最优模型参数的模型
model = GradientBoostingRegressor(n_estimators=500,alpha=0.3,loss='ls',min_samples_leaf=2)
#调用最优模型参数进行训练
model.fit(X_train, y_train)
#使用最优模型进行模型预测
y_pred=model.predict(X_test)
#计算平均绝对误差，均方根误差，r2模型值
#输出计算平均绝对误差，均方根误差，r2模型值
print('r2模型值',model.score(X_test, y_test))
print('平均绝对误差',mean_absolute_error(y_test,y_pred))
print('均方根误差',mean_squared_error(y_test,y_pred))













