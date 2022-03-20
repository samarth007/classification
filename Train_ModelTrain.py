from Logs import logger
from preprocessing import preprocessing
from clusters import clustrngs
import pandas as pd
from ModelFind import Model_Finder
from sklearn.model_selection import train_test_split
from Create_File import CreateModel

class train_model:
    def __init__(self):
        self.logger=logger.App_logger()
        self.file_object=open("TrainingLogs/TrainModel_log.txt",'a+')

    def training_model(self):
        self.logger.log(self.file_object, 'Model training starts')
        self.data=pd.read_csv("InputCsv/diabetes.csv")
        preprocess=preprocessing(self.logger,self.file_object)
        X,y=preprocess.separate_label(self.data,label_column_name='Outcome')
        X=preprocess.detectOutliers(X)
        is_null_present=preprocess.isNullPresent(X)

        if is_null_present:
            X=preprocess.impute(X)

        cols_to_drop=preprocess.is_std_zero(X)
        X=preprocess.removeColumns(X,cols_to_drop)

        #CLUSTERING
        kmeans=clustrngs(self.logger,self.file_object)
        #no_of_clust=kmeans.elbowPlot(X)
        X=kmeans.CreateCluster(X,2)
        X['Labels']=y
        list_of_cluster=X['Cluster'].unique()

        for i in list_of_cluster:
            cluster_data=X[X['Cluster']==i]
            cluster_feature=cluster_data.drop(['Labels','Cluster'],axis=1)
            cluster_label=cluster_data['Labels']
            x_train,x_test,y_train,y_test= train_test_split(cluster_feature,cluster_label,test_size=0.3,random_state=42)
            model_finder=Model_Finder(self.logger,self.file_object)
            best_model_name,best_model=model_finder.getbestmodel(x_train,x_test,y_train,y_test)
            save_file= CreateModel(self.logger,self.file_object)
            save_file.save_file(best_model,best_model_name+str(i))





