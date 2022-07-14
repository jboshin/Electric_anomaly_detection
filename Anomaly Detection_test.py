import pandas as pd
csvfile_path = [r"/Desktop/rearch_test/vn_csvfile"]

# 輸入資料
df_info = pd.read_csv('Info_Agu8.csv')

# 找出FacName、NodeName
FacName = df_info['FacName']
FacName = FacName.drop_duplicates()
NodeName = df_info['NodeName']
NodeName = NodeName.drop_duplicates()

# NA值更換
df_info = df_info.replace('***', 'NaN')
df_info.head(10)

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

# 刪除NA
df_info.isnull().sum().sum()
df_info = df_info.dropna()
df_info.isnull().sum().sum()

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
        
# 資料清洗
len(public_areadf.index)
public_areadf.isnull().sum().sum()
public_areadf = public_areadf.dropna()
public_areadf.isnull().sum().sum()
len(public_areadf.index)
        
# 刪除重複
public_areadf = public_areadf.drop_duplicates()
len(public_areadf.index)
pa_NodeName = df_info['NodeName']
pa_NodeName = pa_NodeName.drop_duplicates()
filt = (public_areadf['NodeName'] == 'NGUỒN TỔNG KHU CÔNG CỘNG')
pa_NTKCC = public_areadf.loc[filt]
        
# 提取數值列的名稱
NodeName = public_areadf.drop(['NodeName'], axis=1 ) 
names=NodeName.columns
df3 = public_areadf.set_index("Datetime")
df = df3.loc[:, ['CurrentAVG']] 
# 將index轉為時間型態
df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))
        
# 平穩性和自相關
import matplotlib.pyplot as plt
# Resample the entire dataset by daily average
CurrentAVG_mean = df.resample(rule='D').mean()
CurrentAVG_std = df.resample(rule='D').std()
# Plot time series for each sensor with its mean and standard deviation

_ = plt.figure(figsize=(18,3))
_ = plt.plot(df['CurrentAVG'], color='blue', label='Original')
_ = plt.plot(CurrentAVG_mean['CurrentAVG'], color='red', label='CurrentAVG Mean')
_ = plt.plot(CurrentAVG_std['CurrentAVG'], color='black', label='CurrentAVG Std' )
_ = plt.legend(loc='best')
_ = plt.title('CurrentAVG')
plt.show()
        
# IQR
# 計算第一主成分 (pc1) 的 IQR
q1_pc1, q3_pc1 = principalDf['pc1'].quantile([0.25, 0.75]) 
iqr_pc1 = q3_pc1 - q1_pc1
# 計算 pc1 異常值的上下限
lower_pc1 = q1_pc1 - (1.5*iqr_pc1) 
upper_pc1 = q3_pc1 + (1.5*iqr_pc1)
print(q1_pc1)
print(q3_pc1)
print(lower_pc1)
print(lower_pc1)
principalDf['pc1']
        
# 過濾掉 pc1 中的異常值
principalDf['anomaly_pc1'] = ((principalDf['pc1']>upper_pc1) | (principalDf['pc1']<lower_pc1)).astype('int')
# 計算第二主成分（pc2）的 IQR
q1_pc2, q3_pc2 = principalDf['pc2'].quantile([0.25, 0.75]) 
iqr_pc2 = q3_pc2 - q1_pc2
# 計算 pc2 異常值的上下限
lower_pc2 = q1_pc2 - (1.5*iqr_pc2) 
upper_pc2 = q3_pc2 + (1.5*iqr_pc2)
# 過濾掉 pc2 中的異常值
principalDf['anomaly_pc2'] = ((principalDf['pc2']>upper_pc2) | (principalDf['pc2']<lower_pc2)).astype('int')
        
# 將index轉為時間型態
principalDf = principalDf.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))
pd.concat([df, principalDf], axis=1) 
        
a = df[principalDf['anomaly_pc1'] == 1] #anomaly 
_ = plt.figure(figsize=(18,6)) 
_ = plt.plot(df['CurrentAVG'], color='blue', label ='normal') 
_ = plt.plot(a['CurrentAVG'], linestyle='none', marker='X', color='red',markersize=12, label='Anomaly') 
_ = plt. xlabel('datetime') 
_ = plt.ylabel('CurrentAVG') 
_ = plt.title('CurrentAVG Anomalies') 
_ = plt.legend(loc='best') 
plt.show();
