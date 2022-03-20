from flask import Flask,request
from flask import Response
import requests
import  json
from TrainValidations import train_valid
from Train_ModelTrain import train_model


app= Flask(__name__)


@app.route("/train",methods=['POST'])
def train():
        if request.json['folderPath'] is not None:
            path=request.json['folderPath']
            # train_obj=train_valid(path)
            # train_obj.train_validations()
            model_train=train_model()
            model_train.training_model()
        return Response('Model Trained')

if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True)