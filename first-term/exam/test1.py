import numpy as np
from point1 import KNN

x1 = np.array([2.3, 2.1, 3.7, 6.1, 6, 9.2, 11.0])
x2 = np.array([1.3, 5.1, 0.7, 9.1, 4, 3.3, 5.0])
X_train=[[x1],[x2]]
print(X_train)
y = np.array([0, 1, 1, 0, 1, 0, 0])
knn = KNN(3)
knn.fit(X_train,y)
print(knn.predict([[3,3]]))