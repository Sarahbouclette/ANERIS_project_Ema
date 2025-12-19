# This projects aims at the classification of UVP6 images. 

To train and evaluate the CNN models we used the MLflow platform to streamline our machine learning experiment. 

|-ANERIS project Ema
|--Data
|--Outputs
|----dataset_split
|----datasets
|----output_training
|----output_prediction
|--Script
conda_classification_env.yml
README.md
                

## Set up MLFLOW
To use MLflow and track runs on the web while executing code on a server(here, ada or marie) you need to execute the following steps; 

1. Create an ssh tunnel between the server and your local host 
ssh -L 9090:localhost:5000 coicaudtou@192.168.77.70 

WARNING-- be sure the ports(in and out) are available

2. Then after activating the right conda environment, you can start the MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /home/coicaudtou/mlflow_user/mlflow_artifacts --host 0.0.0.0 --port 5000  --allowed-hosts "*" --cors-allowed-origins "*"

--host 0.0.0.0 allow multiple host
in this case all host because allowed-hosts "*". 

3. Go on the web interface by writing http://localhost:9090 in any navigator. 

## Fine-tuning your models

The fine-tuning step has only been done on the smartbay validated data. 

### Load models

If you want to try multiple architectures you can load other models in the script Loading_models.py
Two classifier heads are implemented with either;
- one layer depth
- two layers depth

### Start running 0_Model_fine_tunning

First you have to split your dataset intro train, valid and test dataset (by samples). The train and valid dataset will be used for the fine-tunning step. 
Then you create the dataset and load them. 

Once it's done don't forget to tsave them to avoid doing it again and biasing your fine-tuning step.

### Grid search 

To try multiple parameters, you can modify the grid search function. 
4 parameters are already tried: the learning rate, the scheduler, the weigth sentitivity and the gamma for the focal loss. 
You can try the multiples models architectures implemented and classifier head as well a crossentropy loss or focal loss. 

### Launch fine-tuning and track it on mlflow

You can set an experiment on mlflow which correpsond of the phase of your fine-tuning ( see the phase I had design in the file Deepl Experiment in the data). 
Parameters used and metrics defined in the training function will be load at each time step (epoch) on mlflow and then multiples tools are available to compare the results. 

## Training the fine-tuned models

Until now the steps has to be done for each project (1a_aneris and 1b_vilanova because they don't detect the same communities).

### run 1_Model_training
Use the same split as before except that this time train + valid set will be merged because we don't need to evaluate our data during the training and we need as much as validated images as their are available. 

The same training function is used in this step except that the grid-search can contain only one value for each parameter. 

trained models are saved in the Outputs in "output_training". For each loaded model, the model's weight of the epoch with the best balanced accuracy is saved. 

### track it on mlflow

You can set an experiment on mlflow which correpsond of the "training" phase. 
Parameters used and metrics defined in the training function will be load at each time step (epoch) on mlflow and then multiples tools are available to compare the results. 


## Test the fine-tuned and trained models

This is the last step before prediction. The testing step allow you to evaluate how your fine-tuned and trained models predict unseen data. 
In order to choose the best model you should predict the test set and choose a model good plankton-precision and gloablly good balanced accuracy (see confusion matrix). 

run 2_Model_testing

## Predictions the fine-tuned, trained and tested models

run 3_Prediction

For this step images aren't on local, you can access it directly via /vault. 

Predictions will be saved in Outputs/output_prediction

