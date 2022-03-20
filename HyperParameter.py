from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class Tuner:
    def __init__(self,logger,file_obj):
        self.logger=logger
        self.file_object=file_obj

    def bestParamsForRandomFrst(self,x_train,y_train):
        self.logger.log(self.file_object,"selecting best params for RandomFrst")
        self.params_grid_randfrst={
            'criterion':['gini','entropy'],
            'max_depth':[20,30,40],
            'n_estimators':[20,50,100]
        }
        randomForest=RandomForestClassifier()
        self.grid=GridSearchCV(randomForest,param_grid=self.params_grid_randfrst)
        self.grid.fit(x_train,y_train)

        self.criterion=self.grid.best_params_['criterion']
        self.max_depth=self.grid.best_params_['max_depth']
        self.n_estimator=self.grid.best_params_['n_estimators']

        self.tunedRandomFrst=RandomForestClassifier(criterion=self.criterion,max_depth=self.max_depth,n_estimators=self.n_estimator)
        self.tunedRandomFrst.fit(x_train,y_train)
        self.logger.log(self.file_object,"Tunning on randomforest done")
        return self.tunedRandomFrst


    def bestParamsForXGboost(self,x_train,y_train):
        self.logger.log(self.file_object, "selecting best params for XGBoost")
        xgbost=XGBClassifier(objective='binary:logistic')
        self.params_grid_xgboost={
            'learning_rate':[0.01,0.05,0.001],
            'n_estimators':[25,40,50],
            'max_depth':[20,30,50]
        }
        self.grid_xgbosst=GridSearchCV(xgbost,param_grid=self.params_grid_xgboost)
        self.grid_xgbosst.fit(x_train,y_train)

        self.learning_xgboost=self.grid_xgbosst.best_params_['learning_rate']
        self.max_depth_xgboost=self.grid_xgbosst.best_params_['max_depth']
        self.n_est_xgboost=self.grid_xgbosst.best_params_['n_estimators']

        self.tunedXgboost=XGBClassifier(learning_rate=self.learning_xgboost,n_estimator=self.n_est_xgboost,
                                        max_depth=self.max_depth_xgboost)
        self.tunedXgboost.fit(x_train,y_train)
        self.logger.log(self.file_object, "Tunning on XGBOOST done")
        return self.tunedXgboost











