import numpy as np
from point1 import KNN

x1 = np.array([2.3, 2.1, 3.7, 6.1, 6, 9.2, 11.0])
x2 = np.array([1.3, 5.1, 0.7, 9.1, 4, 3.3, 5.0])
X_train=[[x,y] for x,y in zip(x1,x2)]

y = np.array(["no pasa", "pasa", "pasa", "no pasa", "pasa", "no pasa", "no pasa"])
knn = KNN(3)
knn.fit(X_train,y)
print(knn.predict([[9,5],[3,3]]))
print(knn.predict_proba([[9,5],[3,3]]))