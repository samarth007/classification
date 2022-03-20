import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

class preprocessing:

    def __init__(self,logger,fil_obj):
        self.logger=logger
        self.file_obj=fil_obj

    def separate_label(self,data,label_column_name):
        self.X=data.drop(label_column_name,axis=1)
        self.y=data[label_column_name]
        self.logger.log(self.file_obj,"label separation done")
        return  self.X,self.y

    def removeColumns(self,data,colsRem):
        self.data=data
        self.remCol=colsRem
        self.newcols=self.data.drop(self.remCol,axis=1)
        self.logger.log(self.file_obj,"columns removed")
        return self.newcols


    def isNullPresent(self,X):
        self.null_present=False
        self.null_counts=X.isna().sum()
        for i in self.null_counts:
            if i >0:
                self.null_present=True
                break
        self.logger.log(self.file_obj,'Missing values identified')
        return self.null_present


    def impute(self,X):
       imputer=KNNImputer(n_neighbors=3,weights='uniform',missing_values=np.nan)
       self.newAray=imputer.fit_transform(X)
       self.X=pd.DataFrame(data=self.newAray,columns=X.columns)
       self.logger.log(self.file_obj,'Imputation done')
       return self.X


    def is_std_zero(self,X):
        self.data_n=X.describe()
        self.col_to_drop=[]
        self.columns=X.columns
        for i in self.columns:
            if(self.data_n[i]['std']==0):
                self.col_to_drop.append(i)
        self.logger.log(self.file_obj,"features with std are zero")
        return self.col_to_drop
    
    def detectOutliers(self,X):
        for i in X:
          upperlimit=X[i].mean() + 3 * X[i].std()
          lowerlimit=X[i].mean() - 3 * X[i].std()
          self.remOut=X[(X[i] > lowerlimit) & (X[i] < upperlimit)]
        self.logger.log(self.file_obj,'Outliers removed successfully')
        return self.remOut


