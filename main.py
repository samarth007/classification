from flask import Flask,request
from flask import Response

from TrainValidations import train_valid
from Train_ModelTrain import train_model
from PredictModel import Predict


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


@app.route("/predict",methods=['POST'])
def predict():
    if request.json['filePath'] is not None:
        path=request.json['filePath']
        pred=Predict(path)
        pred.predictionFromModel()
        return Response('Prediction Done')


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True)