#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix

X_training = np.array(df.values)

best_k = 0
best_score = 0
silhouette_array = []
for k in range(2,21):
     kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
     kmeans.fit(X_training)
     #for each k, calculate the silhouette_coefficient by using: 
     silhouette_array.append(silhouette_score(X_training, kmeans.labels_))
     if silhouette_array[k-2] > best_score:
         best_k = k
         best_score = silhouette_array[k-2]
         print("Best K: " + best_k.__str__())


#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.plot(range(2,21), silhouette_array)
plt.title("Silhouette Coefficient for each K")
plt.xlabel("K")
plt.ylabel("Silhouette Coefficient")
plt.show()

kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)
kmeans.fit(X_training)
#reading the test data (clusters) by using Pandas library
X_test = pd.read_csv('testing_data.csv', sep=',', header=None).values.reshape(1,-1)[0]


#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = X_test.reshape(1,-1)[0]

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
