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
