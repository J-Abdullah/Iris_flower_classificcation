import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df=pd.read_csv('IRIS.csv')
x=df.drop(['species'],axis=1)
y=df['species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot the histograms
axs[0, 0].hist(x['sepal_length'], bins=10, edgecolor='black')
axs[0, 0].set_title('Sepal Length')

axs[0, 1].hist(x['sepal_width'], bins=10, edgecolor='black')
axs[0, 1].set_title('Sepal Width')

axs[1, 0].hist(x['petal_length'], bins=10, edgecolor='black')
axs[1, 0].set_title('Petal Length')

axs[1, 1].hist(x['petal_width'], bins=10, edgecolor='black')
axs[1, 1].set_title('Petal Width')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()

plt.scatter(x['sepal_length'],x['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs. Sepal Width')
plt.show()

knn=KNeighborsClassifier(n_neighbors=12)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print(y_pred)
print(classification_report(y_pred,y_test))

#predicting new species with unkonwn data
print(knn.predict([[4.6,2.9,1.4,0.3]]))

