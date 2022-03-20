import pandas as pd
from preprocessing import preprocessing
from Logs import logger
from Create_File import CreateModel

class Predict:

    def __init__(self,path):
      self.d=path
      self.logger=logger.App_logger()
      self.file_obj=open('TestingLogs/Test_logs.txt','a+')

    def predictionFromModel(self):
        self.logger.log(self.file_obj,"Prediction started")
        self.data=pd.read_csv(self.d+"/"+'testdata.csv')
        preprocess = preprocessing(self.logger, self.file_obj)
        X=self.data  #assigning to new variable X
        null_present=preprocess.isNullPresent(X)  #checking null value
        if null_present:
            X=preprocess.impute(X)          #imputing null value
        X = preprocess.detectOutliers(X)    # removing outlier
        cols_to_drop=preprocess.is_std_zero(X)   #getting column with same value
        X=preprocess.removeColumns(X,cols_to_drop)  #removing column with same value
        getModel=CreateModel(self.logger,self.file_obj)
        Kmeans=getModel.clusterModelLoading('Kmeans_Cluster_model') #getting cluster model
        cluster=Kmeans.predict(X)  #creating cluster for test data
        X['cluster']=cluster
        clusters_count=X['cluster'].unique()
        for i in clusters_count:
            cluster_data=X[X['cluster']==i]
            cluster_feature=cluster_data.drop('cluster',axis=1)
            model=getModel.load_model(i)
            result=list(model.predict(cluster_feature))
            result=pd.DataFrame(list(zip(result)),columns=['prediction'])
            result.to_csv('Prediction/predict'+str(i)+'.csv',index=False)
            self.logger.log(self.file_obj,"Model prediction done on {} cluster".format(i))



