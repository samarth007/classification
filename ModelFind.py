from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

class Model_Finder:

    def __init__(self,logger,fil_obj):
        self.logger=logger
        self.file_object=fil_obj


    def getbestmodel(self,x_train,x_test,y_train,y_test):
        self.logger.log(self.file_object,"Retriving best model for each cluster")
        self.randfrm=RandomForestClassifier(n_estimators=50)
        self.randfrm.fit(x_train,y_train)
        y_pred=self.randfrm.predict(x_test)
        self.randscore=accuracy_score(y_pred,y_test)


        self.ada=LogisticRegression(penalty='none')
        self.ada.fit(x_train,y_train)
        y_pred_ada=self.ada.predict(x_test)
        self.adascore=accuracy_score(y_pred_ada,y_test)

        if(self.randscore > self.adascore):
            return "RandomForest",self.randfrm
        else:
            return "Logistic",self.ada
