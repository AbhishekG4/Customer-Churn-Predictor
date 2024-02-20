from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\abhis\OneDrive\Desktop\DataCamp\Supervised learning\telecom_churn_clean.csv', index_col=0)
X = df.iloc[:,2:18]
y = df.loc[:,'churn']

col=[]
for i in y.values:
    if i==0:
        col.append('blue')
    else: col.append('red')

plt.figure(figsize=(8,6))
plt.scatter(X.iloc[:,5].values, X.iloc[:,8].values, c=col, s=4)
plt.xlabel('Total Day Charge')
plt.ylabel('Total Eve Charge')
plt.title('Dataset')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=39, stratify=y)
'''knn = KNeighborsClassifier(n_neighbors = 23)
knn.fit(X_train, y_train)'''
n = list(range(40))
testa=[]
traina=[]
for i in n[1:]:
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    testa.append(knn.score(X_test, y_test))
    traina.append(knn.score(X_train, y_train))
plt.plot(n[1:], traina, c='black', label = 'train accuracy')
plt.plot(n[1:], testa, c='blue', label = 'test accuracy')
plt.title('Accuracy chart')
plt.legend()
plt.xlabel('Nearest neighbors')
plt.ylabel('Accuracy')
plt.show()
print('max test acc: ',max(testa),' NN: ',testa.index(max(testa))+1)
print('max train acc: ',max(traina),' NN: ',traina.index(max(traina))+1)


t = np.array([[0,1,25,264,110,45,200,98,15,245,100,12,10,3,2.7,1]])
#print(type(t))
#print(t.shape)
knn = KNeighborsClassifier(n_neighbors = testa.index(max(testa))+1)
knn.fit(X,y)
print('prediction is: ', knn.predict(t))
print('train accuracy is: ', knn.score(X_train, y_train))
print('=============')
print('test accuracy is: ',knn.score(X_test, y_test))
