import pickle


class CreateModel:


  def __init__(self,logger,fil_obj):
   self.logger=logger
   self.file_obj=fil_obj

  def save_file(self,model,model_name):
   pickle.dump(model,open('Model/'+model_name,'wb'))
   self.logger.log(self.file_obj, "Model saved")

  def save_cluster(self,Kmeans,clusterName):
   pickle.dump(Kmeans,open('ClusterModel/'+clusterName,'wb'))
   self.logger.log(self.file_obj,"Cluster model saved")

  def clusterModelLoading(self,x):
      fileName= open('ClusterModel/'+x,'rb')
      self.logger.log(self.file_obj,'Kmeans_Cluster_model_loaded')
      return pickle.load(fileName)

  def load_model(self,i):
   if(i==1):
       fileName=open('Model/RandomForest1','rb')
       self.logger.log(self.file_obj,'Model with i=1 loaded and performed on cluster on 1')
       return  pickle.load(fileName)
   else:
       fileName=open('Model/XGBOOST0','rb')
       self.logger.log(self.file_obj, 'Model with i=0 loaded and performed on cluster on 0')
       return pickle.load(fileName)