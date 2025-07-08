import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data={
    'weight':[150,170,140,130,120,180,110,90,100],
    'size':[7.0,7.5,6.8,6.5,5.5,7.8,5.2,5.0,5.3],
    'fruit':["Apple","Apple","Apple","Apple","Orange","Apple","Orange"
             ,"Grapes","Grapes"]
}
df=pd.DataFrame(data)
df['encode_fruit']=df['fruit'].map({"Apple":0,"Orange":1,"Grapes":2})

#split data
x=df[['weight','size']]
y=df['encode_fruit']

#train-test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy:",accuracy_score(y_test,y_pred))
sample=[[180,8.1]]
prediction=model.predict(sample)
label_map={0:"Apple",1:"Orange",2:"Grapes"}
print([prediction[0]])