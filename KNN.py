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

  
def score(y1,y2):
  n=0
  for i in y1:
    if y1[i] == y2[i]:
      n += 1
  return n/len(y1)
