from sklearn.cluster import KMeans
from Logs import logger
from matplotlib import pyplot as plt
from Create_File import CreateModel

class clustrngs:

    def __init__(self,logger,fil_obj):
        self.logger=logger
        self.file_obj=fil_obj

    # def elbowPlot(self,data):
    #     self.logger.log(self.file_obj,"elbow plot started")
    #     wcss=[]
    #     for i in range(1,11):
    #         km=KMeans(n_clusters=i,init='k-means++',random_state=42)
    #         km.fit(data)
    #         wcss.append(km.inertia_)
    #         plt.plot(range(1,11),wcss)
    #         plt.title('Elbow method')
    #         plt.xlabel('Number of clusters')
    #         plt.ylabel('WCSS')
    #         plt.savefig('TraningLogs/K-means_elbow.PNG')
    #     self.logger.log(self.file_obj,"elbow plot ended")

    def CreateCluster(self,data,clustNum):
        km=KMeans(n_clusters=clustNum,init='k-means++',random_state=42)
        y_means=km.fit_predict(data)
        data['Cluster']=y_means
        save_model=CreateModel(self.logger,self.file_obj)
        save_model.save_cluster(km,"Kmeans_Cluster_model")
        self.logger.log(self.file_obj,"cluster created successfully")
        return data


