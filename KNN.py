# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 19:41:40 2023

@author: JALAL
"""

import numpy as np
import matplotlib.pyplot as plt

class KNNClassifier :
  def __init__(self, k):
    self.k = k
  
  def euclidean_distance(self, x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
  
  def fit(self, X, y):
    self.X_train = X
    self.y_train = y 
    
  def _predict(self, x):

        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)
        
        k_nearest_labels = [self.y_train[i] for i in k_indices[:self.k]]

        label_counts = {}  
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        max_count = 0
        most_common_label = None
        for label, count in label_counts.items():
            if count > max_count:
                max_count = count
                most_common_label = label

        return most_common_label
        
  def predict(self, X):
    y_pred = [self._predict(x) for x in X]
    return np.array(y_pred)

# Exemple d'utilisation avec le jeu de données Iris
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn_classifier = KNNClassifier(k=3)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle k-NN avec k={} : {:.2f}%".format(knn_classifier.k, accuracy * 100))

plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', label='Vraies étiquettes')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='Blues', marker='x', s=100, label='Prédictions')
plt.xlabel('Caractéristique 1')
plt.ylabel('Caractéristique 2')
plt.legend()
plt.title('Graphe de dispersion des vraies étiquettes vs prédictions')
plt.gca().set_facecolor('lightgray')
plt.show()
