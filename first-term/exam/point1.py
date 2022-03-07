from base_predict import Model

class KNN(Model):
    def __init__(self,k):
        """
        Constructor of KNN class
        k - that is the neighbor number
        """
        super().__init__()
        self.neighbours=k
        self.X_train=None
        self.y=None
    def fit(self,X,y):
        """
        Fit the model with KNN algorithm
        X - features
        y - labels
        """
        self.X_train=X
        self.y=y
        if len(X)!=len(y):
            raise ValueError("X and y must have the same length")
        return self
    def predict(self,X):
        """
        Predict the labels of the data
        X - features
        """
        for x in X:
            distances=[]
            for x_train in self.X_train:
                distances.append(self.distance(x,x_train))
            distances.sort()
            k_nearest=distances[:self.neighbours]
            k_nearest_labels=[]
            for k in k_nearest:
                k_nearest_labels.append(self.y[self.X_train.index(k)])
            self.y.append(self.majority(k_nearest_labels))
        return self.y
    def predict_proba(self,X):
        """
        Predict the probabilities of the data
        X - features
        Returns the probabilities of the data (for each label)
        """
        for x in X:
            distances=[]
            for x_train in self.X_train:
                distances.append(self.distance(x,x_train))
            distances.sort()
            k_nearest=distances[:self.neighbours]
            k_nearest_labels=[]
            for k in k_nearest:
                k_nearest_labels.append(self.y[self.X_train.index(k)])
            self.y.append(self.probability(k_nearest_labels))
        return (self.y,[1-self.y[i] for i in range(len(self.y))])
    def distance(self,x):
        """
        Calculate the distance between two data
        x - data
        x_train - data
        """
        return sum([(x[i]-self.X_train[i])**2 for i in range(len(x))])
    def majority(self,k_nearest_labels):
        """
        Calculate the majority of the labels
        k_nearest_labels - labels of the k nearest data
        """
        return max(set(k_nearest_labels),key=k_nearest_labels.count)
    def probability(self,k_nearest_labels):
        """
        Calculate the probabilities of the labels
        k_nearest_labels - labels of the k nearest data
        """
        return k_nearest_labels.count(self.majority(k_nearest_labels))/self.neighbours