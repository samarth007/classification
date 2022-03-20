
from HyperParameter import Tuner

from sklearn.metrics import recall_score

class Model_Finder:

    def __init__(self,logger,fil_obj):
        self.logger=logger
        self.file_object=fil_obj


    def getbestmodel(self,x_train,x_test,y_train,y_test):
        self.logger.log(self.file_object,"Retriving best model for each cluster")
        tun=Tuner(self.logger,self.file_object)

        self.randomFrst=tun.bestParamsForRandomFrst(x_train,y_train)
        y_pred_randfrst=self.randomFrst.predict(x_test)
        self.randscore=recall_score(y_pred_randfrst,y_test)


        self.xgboost=tun.bestParamsForXGboost(x_train,y_train)
        y_pred_xgboost=self.xgboost.predict(x_test)
        self.xgbst_score=recall_score(y_pred_xgboost,y_test)

        if(self.randscore > self.xgbst_score):
            return "RandomForest",self.randomFrst
        else:
            return "XGBOOST",self.xgboost
