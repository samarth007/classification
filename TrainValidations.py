from Raw_data_validation import RawDatavalidation
from Logs import logger

class train_valid:

    def __init__(self,path):
        self.dataTransform=RawDatavalidation(path)
        self.file_object=open("TrainingLogs/Training_log.txt",'a+')
        self.log_writer=logger.App_logger()

    def train_validations(self):
            self.log_writer.log(self.file_object,'Validation starts')

