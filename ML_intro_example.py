# -*- coding: utf-8 -*-
#kutuphaneler
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yukleme

veriler = pd.read_csv('data.csv')
#pd.read_csv("veriler.csv")

print(veriler)

#veri on isleme

boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

x = 10

class insan:
    boy = 180
    def yakilankalori(self,sure):
        return sure*10 

ali = insan() #insan tipinde nesne
print(ali.boy)
print(ali.yakilankalori(90))
 
#eksik verileri  veri ortalamaları ile tamamlama
#sci - kit learn
#impute kutuphanesi simpleimputer class Si
from sklearn.impute import SimpleImputer 

# costructer obj. sutunun ortalamasi ile eksik verileri(nan) doldur
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 

#yas degiskeninin icine 1.2.3. indexli sutunlari atiyoruz
Yas = veriler.iloc[:,1:4].values 
print(Yas)
imputer = imputer.fit(Yas[:,1:4])  #1,2,3 e sutunlarini ogren
print(Yas) #olmasi gerektigi gibi degisiklik yok
Yas[:,1:4] = imputer.transform(Yas[:,1:4]) #ogrenileni nan degiskenlerine ortalamayi uygula
print(Yas)
#encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values  # 0 ı yaz demek verilerden indeksi 0 olan degerleri al
print(ulke)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

#0. sutundaki verileri encode ediyoruz
# fit ve transfor ayni satirda uygulaninca degisim gerceklesir
ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) 

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray() #one code encoder ogren ve uygula 
print(ulke)

print(list(range(22)))

#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us']) # ındesxler ve konu baslıkları olan veri yapıları
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])
print(sonuc3)
s=pd.concat([sonuc,sonuc2]) # dataframe birleştirme axissiz nan li goruntu
print(s)

s=pd.concat([sonuc,sonuc2], axis=1) # dataframe birleştirme  nan siz hale gertirme
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)


from sklearn.model_selection import train_test_split
#verilerin egitim ve test icin bolunmesi
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


































