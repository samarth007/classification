from Logs import logger
import pandas as pd

class RawDatavalidation:
    def __init__(self,path):
        self.logger=logger.App_logger()
        self.data=path

    def replaceMissingvalues(self):
        log_file=open("TrainingLogs/dataTransform.txt",'a+')
        self.newdata=pd.read_csv(self.data+"/"+'diabetes.csv')
        for i in self.newdata:
            self.newdata=self.newdata[i].fillna(self.newdata[i].mode(),inplace=True)

        self.logger.log(log_file,"File missing values replaced successfully")

    def detectOutliers(self):
        log_file=open("TrainingLogs/dataOutlier.txt",'a+')

        for i in self.newdata:
          upperlimit=self.newdata[i].mean() + 3 * self.newdata[i].std()
          lowerlimit=self.newdata[i].mean() - 3 * self.newdata[i].std()
          self.newdata.drop((self.newdata[i]> upperlimit) | (self.newdata[i] < lowerlimit),inplace=True)
        self.logger.log(log_file,'Outliers removed successfully')

    def returnData(self):
        self.newdata.to_csv('InputCsv/NewData.csv',index=False)
