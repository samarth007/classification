import pandas as pd

class Test:
    Y=None
    def __init__(self):
        pass


# for i in X:
#     upperlimit = X[i].mean() + 3 * X[i].std()
#     lowerlimit = X[i].mean() - 3 * X[i].std()
#     X=X[(X[i]>lowerlimit) & (X[i]< upperlimit)]
#
# print(X)


    def detectOutliers(self, X):
     for i in X:
        upperlimit = X[i].mean() + 3 * X[i].std()
        lowerlimit = X[i].mean() - 3 * X[i].std()
        Y = X[(X[i] > lowerlimit) & (X[i] < upperlimit)]
     return Y


data=pd.read_csv('Train_Files/diabetes.csv')
X=data.drop('Outcome',axis=1)
print(X.shape)
c=Test()
print(c.detectOutliers(X))
