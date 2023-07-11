import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np

data=pd.read_csv("C:/Users/user/Desktop/python/MAKİNE ÖĞRENMESİ/data16.csv")
veri=data.copy()

#Description değişkeni modelde kullanılmayacağı için atıldı
veri=veri.drop(columns="Description")

#CustomerID kısmındaki eksik verileri silindi
veri= veri.dropna(subset=['CustomerID'])

#customerId de sondaki .0 yapısından kurtulduk
veri["CustomerID"]=veri["CustomerID"].astype("int")

#tarihler object den tarihe çevrildi
veri["InvoiceDate"]=pd.to_datetime(veri["InvoiceDate"])


#Toplam harcama hesaplandı ve Total denen değişkene atandı
veri["Total"]=veri["Quantity"]*veri["UnitPrice"]

#Total değişkeni 0 ve altındaysa bu müşterinin alımı iptal ettiğini gösterir o yüzden silindi
veri=veri.drop(veri[veri["Total"]<=0].index)

#aykırı değerler
# sns.boxplot(veri["Total"])
# plt.show()

#aykırı gözlem değerleri baskılama yöntemi kullanılarak yeniden hesaplandı
Q1=veri["Total"].quantile(0.25)
Q3=veri["Total"].quantile(0.75)
IQR=Q3-Q1
altsınır=Q1-1.5*IQR
ustsınır=Q3+1.5*IQR
veri2=veri.loc[veri["Total"]<altsınır,"Total"]=altsınır
veri2=veri.loc[veri["Total"]>ustsınır,"Total"]=ustsınır
# sns.boxplot(veri["Total"])
# plt.show()

#index değerleri yenilendi
veri2=veri.reset_index(drop=True)


#tekrarsız müşteri ID si sayısını gösterir
#bu örnekte 406 bin satır olmasına rağmen 4338 farklı müşteri var sadece
# print(veri["CustomerID"].nunique())
#tekrarsız fatura sayısını gösterir 18532 fatura var. Bu demek oluyor ki herhangi bir müşteriye birden fazla fatura kesilmiş olabilir
# print(veri["InvoiceNo"].nunique())

#RFM ANALYSİS


#elimizdeki veri eski olduğunu için Recency hesaplarken verideki en son tarihi baz alıyoruz
sontarih=veri["InvoiceDate"].max()
sontarih=dt.datetime(2011,12,9,12,50,0)
# print(sontarih)

#customerıd leri grupluyor çünkü veride bir customer Id içerisinde birden fazla fatura var
#bu grupladığı id lerin en son tarihini alıp bugünden çıkarıyor. böylelikle müşterinin en son ne zaman geldiği bulunuyor
Recency=(sontarih-veri.groupby("CustomerID").agg({"InvoiceDate":"max"})).apply(lambda x:x.dt.days)
# print(Recency)

#her bir customerId nin kaç tane faturası olduğunu yani frekansı ortaya çıkardı
Frequency=veri.groupby(["CustomerID","InvoiceNo"]).agg({"InvoiceNo":"count"})
Frequency=Frequency.groupby("CustomerID").agg({"InvoiceNo":"count"})
# print(Frequency)

#müşterilerin ne kadar harcadığını gösteriyor
Monetary=veri.groupby("CustomerID").agg({"Total":"sum"})
# print(Monetary)

#r f ve m yi dataframe halinde birleştirildi ve isimleri değiştirildi
RFM=Recency.merge(Frequency,on="CustomerID").merge(Monetary,on="CustomerID")
RFM=RFM.rename(columns={"InvoiceDate":"Recency","InvoiceNo":"Frequency",
"Total":"Monetary"})
RFM=RFM.reset_index()

# print(RFM)

#costumer sütunu olmadan df değişkeni oluşturuldu
df=RFM.iloc[:,1:]

#değerler standartlaştırıldı
sc=StandardScaler()
dfnorm=sc.fit_transform(df)
dfnorm=pd.DataFrame(dfnorm,columns=df.columns)
# print(dfnorm)


# k sayısı bulma 4 çıktı
# kmodel=KMeans(random_state=0)
# grafik=KElbowVisualizer(kmodel,k=(2,10))
# grafik.fit(dfnorm)
# grafik.poof()

kmodel=KMeans(random_state=0,n_clusters=4,init="k-means++")
kfit=kmodel.fit(dfnorm)
labels=kfit.labels_

sns.scatterplot(x="Recency",y="Frequency",data=dfnorm,hue=labels,palette="deep")
plt.show()


#labels diye bir sütün oluşturduk ve müşterilerin hangi kümeye ait olduğunu gösteriyor
RFM["Labels"]=labels
print(RFM)

#grupların harcamaları
sns.scatterplot(x="Labels",y="Monetary",data=RFM,hue=labels,palette="deep")
plt.show()



#her bir kümede kaç müşteri var
customercount=RFM.groupby("Labels")["CustomerID"].count()
print(customercount)

#müşterilerin kümelere göre sayısını gösteren grafik
x = [0, 1, 2, 3]
y = [2967, 297, 9, 1065]

plt.bar(x, y)

plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Histogram')

plt.show()

#grupların ortalamalrını alıyor etiketlere göre grupluyor burda
meaninfo=RFM.groupby("Labels").mean().iloc[:,1:]
print(meaninfo)

#müşterileri 1 den 5 e kadar skorluyoruz
RFM['recency_score']= pd.qcut(RFM['Recency'], 5 , [5, 4, 3, 2, 1])
RFM['frequency_score']= pd.qcut(RFM['Frequency'].rank(method='first'), 5 , [1, 2, 3, 4, 5])
RFM['monetary_score']= pd.qcut(RFM['Monetary'], 5 , [1, 2, 3, 4, 5])
print(RFM.head())


#skorlarını toplayıp total skor elde ediyoruz
RFM['RF_SCORE']= RFM['recency_score'].astype(str) + RFM['frequency_score'].astype(str)
print(RFM.head())


#müşterileri segmentlerine ayırıyoruz
segmentation  = { r'[1-2][1-2]' : 'hibernating',
         r'[1-2][3-4]' : 'at_risk',
         r'[1-2]5' : 'cant_loose',
         r'3[1-2]' : 'about_to_sleep',
         r'33' : 'need_attention',
         r'[3-4][4-5]' : 'loyal_customers',
         r'41' : 'promising',
         r'51' : 'new_customers',
         r'[4-5][2-3]' : 'potantial_loyalist',
         r'5[4-5]' : 'champions' }

RFM['RF_SEGMENTS'] = RFM['RF_SCORE'].replace( segmentation, regex=True )
print(RFM.head())

#pasta grafiği oluşturuyoruz
RFM['RF_SEGMENTS'].value_counts().plot(kind='pie', autopct=' %1.1f%%',figsize=(8,8))
plt.show(block=True)



