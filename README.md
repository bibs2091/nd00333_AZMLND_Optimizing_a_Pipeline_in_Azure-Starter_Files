# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary

The problem involves analyzing a Bank Marketing dataset. This dataset includes information about bank clients and their responses to marketing campaigns promoting term deposit products. The objective is to perform a classification task to predict whether a client will subscribe to a term deposit, with input variables consisting of client demographic details and the output variable denoted by the 'y' column indicating the subscription status (binary Yes or No).

To summary, AutoML outperformed the Scikit-learn Pipeline when it comes to Accuracy. 

## Scikit-learn Pipeline
The pipeline structure encompasses a Python training script (train.py), a tabular dataset obtained from the URL repository, a Scikit-learn Logistic Regression Algorithm and Azure HyperDrive for hyperparameter tuning, ultimately generating a HyperDrive classifier. The execution of the training run is managed by a Jupyter Notebook hosted on a compute instance, while the experiment run it self is executed on a cluster with up to 4 nodes.

Detailed steps of the pipeline:
- Create a train.py file that contain the data preprocessing pipeline (feature engineering + data splitting)
- Create a random search hyperperameter sampler that tune 'C' and 'max_iter' hyperparams. C is the Regularization while max_iter is the maximum number of iterations.
- Create an early stop policy 
-  Setup the environment for the training run
- Create HyperDriveConfig that specify the policy, env, sampler and the run config.



I chose the random sampler because it is the faster and supports early termination of low-performance runs.



The chosen early termination strategy BanditPolicy automatically stops runs that are not meeting performance expectations, enhancing computational efficiency.

## AutoML
Since using x and y as input will be soon deprecated, I opted for combining the x and y in one dataframe called it 'training_data':
```
training_data = x.copy()
training_data['output'] = y
```
then I defined my AutoMLConfig like that:
```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=training_data,
    label_column_name='output',
    n_cross_validations=4)
```

***task='classification'***
The current task type is classification.

***primary_metric='accuracy'***
Accuracy as the primary metric to be comparable with the SKlearn apreach

***label_column_name='output'***
The label name is 'output' as shown on top.

***n_cross_validations=4***
The metrics are calculated with the average of the 4 validation values.

## Pipeline comparison


| HyperDrive Model|  |
| ----------- | ----------- |
| ID      |    HD_95c96b4b-1463-4ff2-bac6-d62267d2c976_12 |
| Accuracy   | 0.9120838        |

| AutoML Model	|  |
| ----------- | ----------- |
| ID      |    AutoML_2d63726a-c590-4485-b7bd-ccc151c6d99c_23 |
| Accuracy   | 0.9176326        |


HyperDrive only experimented with Logistic regression while AutoML tried multipled ML models (including logistic regression) which gave it the advantage of performing better with minimal work.   

## Future work
- Use a different metric than accuracy to tune and measure the performance of models such as F1score because the data is highly imbalanced.
- Experiment with different hyperparameter sampling methods than the random sampler.

