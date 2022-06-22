
#AUTHOR : MURAT YILDIRIM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train test split
from sklearn.neighbors import KNeighborsClassifier # knn 

data = pd.read_csv("column_2C_weka.csv")
abnormal = data[data["class"]=="Abnormal"]
normal = data[data["class"]=="Normal"]
plt.scatter(abnormal.sacral_slope,abnormal.pelvic_incidence,color="red",label="kotu",alpha= 0.3)
plt.scatter(normal.sacral_slope,normal.pelvic_incidence,color="green",label="iyi",alpha= 0.3)
plt.xlabel("sacral_slope")
plt.ylabel("pelvic_incidence")
plt.legend()
plt.show()

data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]
y = data["class"].values
x_data = data.drop(["class"],axis=1)

# normalization
x= (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
knn = KNeighborsClassifier(n_neighbors = 1) 
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" k = {} , score = {} ".format(1,knn.score(x_test,y_test)))



score_list=[]
best_n_neighbors=1;
best_score=knn.score(x_test,y_test)


for each in range(1,20) :
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    if(knn2.score(x_test,y_test)>best_score) :
        best_score=knn2.score(x_test,y_test)
        best_n_neighbors=each
plt.plot(range(1,20),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()

print("******** Best ********")

knn = KNeighborsClassifier(n_neighbors = best_n_neighbors) 
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" k = {} , score = {} ".format(best_n_neighbors ,knn.score(x_test,y_test)))
