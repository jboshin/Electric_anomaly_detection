import pandas as pd

# 輸入資料
csvfile_path = [r"/Desktop/rearch_test/vn_csvfile"]
df_info = pd.read_csv('Info_Agu_date.csv')
# 找出FacName、NodeName
FacName = df_info['FacName']
FacName = FacName.drop_duplicates()
NodeName = df_info['NodeName']
NodeName = NodeName.drop_duplicates()
# 刪除數值單位
df_info['CurrentAVG'] = df_info['CurrentAVG'].str.replace('A', '')
df_info['VoltageAVG'] = df_info['VoltageAVG'].str.replace('V', '')
df_info['PowerTotal'] = df_info['PowerTotal'].str.replace('kW', '')
df_info['PowerFactorTotal'] = df_info['PowerFactorTotal'].str.replace('Lead', '')
df_info['PowerFactorTotal'] = df_info['PowerFactorTotal'].str.replace('Unity', '')
df_info['PowerFactorTotal'] = df_info['PowerFactorTotal'].str.replace('Lagg', '')
df_info['ActiveEnergyDelivered'] = df_info['ActiveEnergyDelivered'].str.replace('GWh', '')
df_info['ActiveEnergyDelivered'] = df_info['ActiveEnergyDelivered'].str.replace('MWh', '000')
df_info['ActiveEnergyDelivered'] = df_info['ActiveEnergyDelivered'].str.replace('KWh', '000000')
df_info['ActiveEnergyDelivered'] = df_info['ActiveEnergyDelivered'].str.replace('Wh', '000000000')
df_info['CurrentA'] = df_info['CurrentA'].str.replace('A', '')
df_info['CurrentB'] = df_info['CurrentB'].str.replace('A', '')
df_info['CurrentC'] = df_info['CurrentC'].str.replace('A', '')
df_info['VoltageAB'] = df_info['VoltageAB'].str.replace('V', '')
df_info['VoltageBC'] = df_info['VoltageBC'].str.replace('V', '')
df_info['VoltageCA'] = df_info['VoltageCA'].str.replace('V', '')
df_info['THDIA'] = df_info['THDIA'].str.replace('%', '')
df_info['THDIB'] = df_info['THDIB'].str.replace('%', '')
df_info['THDIC'] = df_info['THDIC'].str.replace('%', '')
df_info['THDVAB'] = df_info['THDVAB'].str.replace('%', '')
df_info['THDVBC'] = df_info['THDVBC'].str.replace('%', '')
df_info['THDVCA'] = df_info['THDVCA'].str.replace('%', '')
df_info['THDVLL'] = df_info['THDVLL'].str.replace('%', '')
df_info['QTot'] = df_info['QTot'].str.replace('kVAR', '')
df_info['QTot'] = df_info['QTot'].str.replace('k', '')
df_info['STot'] = df_info['STot'].str.replace('kVA', '')
# df_info['Datetime'] = df_info['Datetime'].str.replace('AM', '')

# 刪除NA
df_info = df_info.dropna()

# 資料型態轉換
# 欄位"CurrentAVG"
df_info['CurrentAVG'] = df_info['CurrentAVG'].astype('float')

# 欄位"VoltageAVG"
df_info['VoltageAVG'] = df_info['VoltageAVG'].astype('float')

# 欄位"PowerTotal"
df_info['PowerTotal'] = df_info['PowerTotal'].astype('float')

# 欄位"PowerFactorTotal"
df_info['PowerFactorTotal'] = df_info['PowerFactorTotal'].astype('float')

# 欄位"ActiveEnergyDelivered"
df_info['ActiveEnergyDelivered'] = df_info['ActiveEnergyDelivered'].astype('float')

# 欄位"CurrentA"
df_info['CurrentA'] = df_info['CurrentA'].astype('float')

# 欄位"CurrentB"
df_info['CurrentB'] = df_info['CurrentB'].astype('float')

# 欄位"CurrentC"
df_info['CurrentC'] = df_info['CurrentC'].astype('float')

# 欄位"VoltageAB"
df_info['VoltageAB'] = df_info['VoltageAB'].astype('float')

# 欄位"VoltageBC"
df_info['VoltageBC'] = df_info['VoltageBC'].astype('float')

# 欄位"VoltageCA"
df_info['VoltageCA'] = df_info['VoltageCA'].astype('float')

# 欄位"THDIA"
df_info['THDIA'] = df_info['THDIA'].astype('float')

# 欄位"THDIB"
df_info['THDIB'] = df_info['THDIB'].astype('float')

# 欄位"THDIC"
df_info['THDIC'] = df_info['THDIC'].astype('float')

# 欄位"THDVAB"
df_info['THDVAB'] = df_info['THDVAB'].astype('float')

# 欄位"THDVBC"
df_info['THDVBC'] = df_info['THDVBC'].astype('float')

# 欄位"THDVCA"
df_info['THDVCA'] = df_info['THDVCA'].astype('float')

# 欄位"THDVLL"
df_info['THDVLL'] = df_info['THDVLL'].astype('float')

# 欄位"QTot"
df_info['QTot'] = df_info['QTot'].astype('float')

# 欄位"STot"
df_info['STot'] = df_info['STot'].astype('float')

# # 欄位"Datetime"
# df_info['Datetime'] = df_info['Datetime'].astype('float')

# 欄位篩選
filt = (df_info['FacName'] == 'Public area')
public_areadf = df_info.loc[filt]
public_areadf.isnull().sum().sum()
pa_NodeName = df_info['NodeName']
pa_NodeName = pa_NodeName.drop_duplicates()
filt = (public_areadf['NodeName'] == 'NGU?N T?NG KHU C?NG C?NG')
pa_NTKCC = public_areadf.loc[filt]

from sklearn import cluster, metrics
import numpy as np

# Kmeans演算法, 分三群
CurrentAVG_train = pa_NTKCC['CurrentAVG']
CurrentAVG_train = np.array(CurrentAVG_train).reshape(-1, 1)
CurrentAVG_clf = cluster.KMeans(n_clusters = 3).fit(CurrentAVG_train)

# 印出分群結果
CurrentAVG_labels = CurrentAVG_clf.labels_
print("分群結果：")
print(CurrentAVG_labels)
print("---")

# 印出績效
CurrentAVG_result = metrics.silhouette_score(CurrentAVG_train, CurrentAVG_labels)
print("CurrentAVG分群績效(3)：")
print(CurrentAVG_result)
print("---------")

# Kmeans演算法, 分五群
CurrentAVG_train_5 = pa_NTKCC['CurrentAVG']
CurrentAVG_train_5 = np.array(CurrentAVG_train_5).reshape(-1, 1)
CurrentAVG_clf_5 = cluster.KMeans(n_clusters = 5).fit(CurrentAVG_train_5)

# 印出分群結果
CurrentAVG_labels_5 = CurrentAVG_clf_5.labels_
print("分群結果：")
print(CurrentAVG_labels_5)
print("---")

# 印出績效
CurrentAVG_result_5 = metrics.silhouette_score(CurrentAVG_train_5, CurrentAVG_labels_5)
print("CurrentAVG分群績效(5)：")
print(CurrentAVG_result_5)
print("---------")

# Kmeans演算法, 分十群
CurrentAVG_train_10 = pa_NTKCC['CurrentAVG']
CurrentAVG_train_10 = np.array(CurrentAVG_train_10).reshape(-1, 1)
CurrentAVG_clf_10 = cluster.KMeans(n_clusters = 10).fit(CurrentAVG_train_10)

# 印出分群結果
CurrentAVG_labels_10 = CurrentAVG_clf_10.labels_
print("分群結果：")
print(CurrentAVG_labels_10)
print("---")

# 印出績效
CurrentAVG_result_10 = metrics.silhouette_score(CurrentAVG_train_10, CurrentAVG_labels_10)
print("CurrentAVG分群績效(10)：")
print(CurrentAVG_result_10)
print("---------")

# Kmeans演算法, 分十五群
CurrentAVG_train_15 = pa_NTKCC['CurrentAVG']
CurrentAVG_train_15 = np.array(CurrentAVG_train_15).reshape(-1, 1)
CurrentAVG_clf_15 = cluster.KMeans(n_clusters = 15).fit(CurrentAVG_train_15)

# 印出分群結果
CurrentAVG_labels_15 = CurrentAVG_clf_15.labels_
print("分群結果：")
print(CurrentAVG_labels_15)
print("---")

# 印出績效
CurrentAVG_result_15 = metrics.silhouette_score(CurrentAVG_train_15, CurrentAVG_labels_15)
print("CurrentAVG分群績效(15)：")
print(CurrentAVG_result_15)
print("---------")

# Kmeans演算法, 分二十群
CurrentAVG_train_20 = pa_NTKCC['CurrentAVG']
CurrentAVG_train_20 = np.array(CurrentAVG_train_20).reshape(-1, 1)
CurrentAVG_clf_20 = cluster.KMeans(n_clusters = 20).fit(CurrentAVG_train_20)

# 印出分群結果
CurrentAVG_labels_20 = CurrentAVG_clf_20.labels_
print("分群結果：")
print(CurrentAVG_labels_20)
print("---")

# 印出績效
CurrentAVG_result_20 = metrics.silhouette_score(CurrentAVG_train_20, CurrentAVG_labels_20)
print("CurrentAVG分群績效(20)：")
print(CurrentAVG_result_20)
print("---------")

# Kmeans演算法, 分25群
CurrentAVG_train_25 = pa_NTKCC['CurrentAVG']
CurrentAVG_train_25 = np.array(CurrentAVG_train_25).reshape(-1, 1)
CurrentAVG_clf_25 = cluster.KMeans(n_clusters = 25).fit(CurrentAVG_train_25)

# 印出分群結果
CurrentAVG_labels_25 = CurrentAVG_clf_25.labels_
print("分群結果：")
print(CurrentAVG_labels_25)
print("---")

# 印出績效
CurrentAVG_result_25 = metrics.silhouette_score(CurrentAVG_train_25, CurrentAVG_labels_25)
print("CurrentAVG分群績效(25)：")
print(CurrentAVG_result_25)
print("---------")

# Kmeans演算法, 分三群
VoltageAVG_train = pa_NTKCC['VoltageAVG']
VoltageAVG_train = np.array(VoltageAVG_train).reshape(-1, 1)
VoltageAVG_clf = cluster.KMeans(n_clusters = 3).fit(VoltageAVG_train)

# 印出分群結果
VoltageAVG_labels = VoltageAVG_clf.labels_
print("分群結果：")
print(VoltageAVG_labels)
print("---")

# 印出績效
VoltageAVG_result = metrics.silhouette_score(VoltageAVG_train, VoltageAVG_labels)
print("VoltageAVG分群績效(3)：")
print(VoltageAVG_result)
print("---------")

# Kmeans演算法, 分五群
VoltageAVG_train_5 = pa_NTKCC['VoltageAVG']
VoltageAVG_train_5 = np.array(VoltageAVG_train_5).reshape(-1, 1)
VoltageAVG_clf_5 = cluster.KMeans(n_clusters = 5).fit(VoltageAVG_train_5)

# 印出分群結果
VoltageAVG_labels_5 = VoltageAVG_clf_5.labels_
print("分群結果：")
print(VoltageAVG_labels_5)
print("---")

# 印出績效
VoltageAVG_result_5 = metrics.silhouette_score(VoltageAVG_train_5, VoltageAVG_labels_5)
print("VoltageAVG分群績效(5)：")
print(VoltageAVG_result_5)
print("---------")

# Kmeans演算法, 分十群
VoltageAVG_train_10 = pa_NTKCC['VoltageAVG']
VoltageAVG_train_10 = np.array(VoltageAVG_train_10).reshape(-1, 1)
VoltageAVG_clf_10 = cluster.KMeans(n_clusters = 10).fit(VoltageAVG_train_10)

# 印出分群結果
VoltageAVG_labels_10 = VoltageAVG_clf_10.labels_
print("分群結果：")
print(VoltageAVG_labels_10)
print("---")

# 印出績效
VoltageAVG_result_10 = metrics.silhouette_score(VoltageAVG_train_10, VoltageAVG_labels_10)
print("VoltageAVG分群績效(10)：")
print(VoltageAVG_result_10)
print("---------")

# Kmeans演算法, 分十五群
VoltageAVG_train_15 = pa_NTKCC['VoltageAVG']
VoltageAVG_train_15 = np.array(VoltageAVG_train_15).reshape(-1, 1)
VoltageAVG_clf_15 = cluster.KMeans(n_clusters = 15).fit(VoltageAVG_train_15)

# 印出分群結果
VoltageAVG_labels_15 = VoltageAVG_clf_15.labels_
print("分群結果：")
print(VoltageAVG_labels_15)
print("---")

# 印出績效
VoltageAVG_result_15 = metrics.silhouette_score(VoltageAVG_train_15, VoltageAVG_labels_15)
print("VoltageAVG分群績效(15)：")
print(VoltageAVG_result_15)
print("---------")

# Kmeans演算法, 分二十群
VoltageAVG_train_20 = pa_NTKCC['VoltageAVG']
VoltageAVG_train_20 = np.array(VoltageAVG_train_20).reshape(-1, 1)
VoltageAVG_clf_20 = cluster.KMeans(n_clusters = 20).fit(VoltageAVG_train_20)

# 印出分群結果
VoltageAVG_labels_20 = VoltageAVG_clf_20.labels_
print("分群結果：")
print(VoltageAVG_labels_20)
print("---")

# 印出績效
VoltageAVG_result_20 = metrics.silhouette_score(VoltageAVG_train_20, VoltageAVG_labels_20)
print("VoltageAVG分群績效(20)：")
print(VoltageAVG_result_20)
print("---------")

# Kmeans演算法, 分25群
VoltageAVG_train_25 = pa_NTKCC['VoltageAVG']
VoltageAVG_train_25 = np.array(VoltageAVG_train_25).reshape(-1, 1)
VoltageAVG_clf_25 = cluster.KMeans(n_clusters = 25).fit(VoltageAVG_train_25)

# 印出分群結果
VoltageAVG_labels_25 = VoltageAVG_clf_25.labels_
print("分群結果：")
print(VoltageAVG_labels_25)
print("---")

# 印出績效
VoltageAVG_result_25 = metrics.silhouette_score(VoltageAVG_train_25, VoltageAVG_labels_25)
print("VoltageAVG分群績效(25)：")
print(VoltageAVG_result_25)
print("---------")

# Kmeans演算法, 分30群
VoltageAVG_train_30 = pa_NTKCC['VoltageAVG']
VoltageAVG_train_30 = np.array(VoltageAVG_train_30).reshape(-1, 1)
VoltageAVG_clf_30 = cluster.KMeans(n_clusters = 30).fit(VoltageAVG_train_30)

# 印出分群結果
VoltageAVG_labels_30 = VoltageAVG_clf_30.labels_
print("分群結果：")
print(VoltageAVG_labels_30)
print("---")

# 印出績效
VoltageAVG_result_30 = metrics.silhouette_score(VoltageAVG_train_30, VoltageAVG_labels_30)
print("VoltageAVG分群績效(30)：")
print(VoltageAVG_result_30)
print("---------")

# Kmeans演算法, 分35群
VoltageAVG_train_35 = pa_NTKCC['VoltageAVG']
VoltageAVG_train_35 = np.array(VoltageAVG_train_35).reshape(-1, 1)
VoltageAVG_clf_35 = cluster.KMeans(n_clusters = 35).fit(VoltageAVG_train_35)

# 印出分群結果
VoltageAVG_labels_35 = VoltageAVG_clf_35.labels_
print("分群結果：")
print(VoltageAVG_labels_35)
print("---")

# 印出績效
VoltageAVG_result_35 = metrics.silhouette_score(VoltageAVG_train_35, VoltageAVG_labels_35)
print("VoltageAVG分群績效(35)：")
print(VoltageAVG_result_35)
print("---------")

# Kmeans演算法, 分40群
VoltageAVG_train_40 = pa_NTKCC['VoltageAVG']
VoltageAVG_train_40 = np.array(VoltageAVG_train_40).reshape(-1, 1)
VoltageAVG_clf_40 = cluster.KMeans(n_clusters = 40).fit(VoltageAVG_train_40)

# 印出分群結果
VoltageAVG_labels_40 = VoltageAVG_clf_40.labels_
print("分群結果：")
print(VoltageAVG_labels_40)
print("---")

# 印出績效
VoltageAVG_result_40 = metrics.silhouette_score(VoltageAVG_train_40, VoltageAVG_labels_40)
print("VoltageAVG分群績效(40)：")
print(VoltageAVG_result_40)
print("---------")

# Kmeans演算法, 分45群
VoltageAVG_train_45 = pa_NTKCC['VoltageAVG']
VoltageAVG_train_45 = np.array(VoltageAVG_train_45).reshape(-1, 1)
VoltageAVG_clf_45 = cluster.KMeans(n_clusters = 45).fit(VoltageAVG_train_45)

# 印出分群結果
VoltageAVG_labels_45 = VoltageAVG_clf_45.labels_
print("分群結果：")
print(VoltageAVG_labels_45)
print("---")

# 印出績效
VoltageAVG_result_45 = metrics.silhouette_score(VoltageAVG_train_45, VoltageAVG_labels_45)
print("VoltageAVG分群績效(45)：")
print(VoltageAVG_result_45)
print("---------")

# Kmeans演算法, 分100群
VoltageAVG_train_100 = pa_NTKCC['VoltageAVG']
VoltageAVG_train_100 = np.array(VoltageAVG_train_100).reshape(-1, 1)
VoltageAVG_clf_100 = cluster.KMeans(n_clusters = 100).fit(VoltageAVG_train_100)

# 印出分群結果
VoltageAVG_labels_100 = VoltageAVG_clf_100.labels_
print("分群結果：")
print(VoltageAVG_labels_100)
print("---")

# 印出績效
VoltageAVG_result_100 = metrics.silhouette_score(VoltageAVG_train_100, VoltageAVG_labels_100)
print("VoltageAVG分群績效(100)：")
print(VoltageAVG_result_100)
print("---------")

# Kmeans演算法, 分200群
VoltageAVG_train_200 = pa_NTKCC['VoltageAVG']
VoltageAVG_train_200 = np.array(VoltageAVG_train_200).reshape(-1, 1)
VoltageAVG_clf_200 = cluster.KMeans(n_clusters = 200).fit(VoltageAVG_train_200)

# 印出分群結果
VoltageAVG_labels_200 = VoltageAVG_clf_200.labels_
print("分群結果：")
print(VoltageAVG_labels_200)
print("---")

# 印出績效
VoltageAVG_result_200 = metrics.silhouette_score(VoltageAVG_train_200, VoltageAVG_labels_200)
print("VoltageAVG分群績效(200)：")
print(VoltageAVG_result_200)
print("---------")

# Kmeans演算法, 分1000群
VoltageAVG_train_1000 = pa_NTKCC['VoltageAVG']
VoltageAVG_train_1000 = np.array(VoltageAVG_train_1000).reshape(-1, 1)
VoltageAVG_clf_1000 = cluster.KMeans(n_clusters = 1000).fit(VoltageAVG_train_1000)

# 印出分群結果
VoltageAVG_labels_1000 = VoltageAVG_clf_1000.labels_
print("分群結果：")
print(VoltageAVG_labels_1000)
print("---")

# 印出績效
VoltageAVG_result_1000 = metrics.silhouette_score(VoltageAVG_train_1000, VoltageAVG_labels_1000)
print("VoltageAVG分群績效(1000)：")
print(VoltageAVG_result_1000)
print("---------")

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pandas import DataFrame

df = DataFrame(pa_NTKCC,columns=['CurrentAVG','VoltageAVG'])

kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

# result_3 = metrics.silhouette_score(df, centroids)

plt.scatter(df['CurrentAVG'], df['VoltageAVG'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

kmeans_5 = KMeans(n_clusters=5).fit(df)
centroids_5 = kmeans_5.cluster_centers_
print(centroids_5)

plt.scatter(df['CurrentAVG'], df['VoltageAVG'], c= kmeans_5.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids_5[:, 0], centroids_5[:, 1], c='red', s=50)
plt.show()

kmeans_10 = KMeans(n_clusters=10).fit(df)
centroids_10 = kmeans_10.cluster_centers_
print(centroids_10)

plt.scatter(df['CurrentAVG'], df['VoltageAVG'], c= kmeans_10.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids_10[:, 0], centroids_10[:, 1], c='red', s=50)
plt.show()
