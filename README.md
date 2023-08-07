# pdk-churn

This example assumes that you have already set up your Pachyderm/MLDM + Determined/MLDE + KServe environment.

Starting point of this example is a Determined experiment that has not been created with Pachyderm in mind. This example goes through the different steps that need to be followed to integrate that experiment to a Pachyderm training pipeline and then to deploy the resulting model with KServe.

Dataset used for this example is an edited and simplified version of this [dataset](https://www.kaggle.com/datasets/abhinav89/telecom-customer) from Kaggle. The base Determined experiment will train a very simple binary classification model on this dataset to predict whether a customer will churn (1) or not (0).


**Note that this example show just one possibility to implement a training and a deployment pipeline, there are certainly other ways to achieve the same result.**

**Please note that this example is based on an experiment using the PyTorch Trial API on Determined, some additional changes will be required if your starting Determined experiment is using CoreAPI or TFKeras Trial API.**

# Files

**Base_experiment** contains files used to start a regular experiment on the Determined platform, training a small dense neural network with the PyTorchTrial API.

**PDK_implementation** contains three folders:
  * **container** contains all the necessary files to create the two images used in training and deployment pipelines.
  * **experiment** contains files to run the experiment when the training pipeline is triggered. These are adapted from the files in the Base_experiment folder.
  * **pipelines** contains the JSON files used to create both the training and the deployment pipelines on Pachyderm.

**data.zip** contains three files:
  * **data_part1.csv** containing 31009 samples, used in the base experiment. This is the first dataset you should commit to your Pachyderm repository.
  * **data_part2.csv** containing 31000 samples, to test the model retraining by commiting this data to your Pachyderm repository, in order to trigger the training pipeline a second time.
  * **customer_churn_sample.csv** containing 10 samples, to test model inference.

# Porting a model from a regular Determined experiment to PDK

Assuming installation of all PDK components is complete

## Step 1: Adapt you experiment files

### Step 1-1: Changes to your experiment config file
* Most likely, if you plan to use Pachyderm, you will use data from Pachyderm to train your model. Therefore, you should remove existing mention of data files or data paths if there was any in your experiment config file.
* Instead, add all the *pachyderm* keys to your config file, as shown in **const.yaml**.
* Additionally, if your original experiment was specifying training length in number of epochs, you may want to switch that length to a number of batches instead.
  * Indeed, the number of samples in your training set will vary as you commit new data to your Pachyderm repository, and knowing that number of samples is mandatory if you want to specify training length in number of epochs.
  * Note that you could always modify the training pipeline image to deal with that issue, but specifying the training length in batches is much simpler.
* Depending on your organization, your Determined cluster and where you intend to have these automatically triggered experiments running, you may also want to edit the workspace and project fields accordingly.

### Step 1-2: Add code to download data from pachyderm
* In startup-hook.sh, install python-pachyderm.
* In data.py (or in any python file you prefer) add the imports (os, shutil, python-pachyderm) that are required to define the two new functions to add: safe_open_wb, and download_pach_repo. The later one being used to download data from a Pachyderm repository.
  * **Note:** In this example, download_pach_repo will only download files corresponding to the difference between current and last commit on the Pachyderm repository. It won't redownload and retrain on the initial data_part1 if you've committed data_part2 afterwards.
* In model_def.py:
  * Add os, logging and download_pach_repo as imports
  * In __init__, check if we intend to train the model (requiring downloading data from pachyderm repository, and building the training and validation sets) or not. Don't hesitate to compare both versions of model_def if this is unclear.
  * Add the download_data function, that will call the download_pach_repo function to download files from the Pachyderm repository and return the list of those files.
		
### Step 1-3: Make sure your code handles the output of the download_data function

Your original code may not handle a list of files, as output by the download_data function. You will notice that, in the base experiment, we were expecting a single data file, while the PDK experiment, we can expect a list of files.
Depending on your original code, and how you expect your data to be commited to Pachyderm, this may or may not be straightforward.

In this example, we simply changed the get_train_and_validation_datasets from data.py, to concatenate csv files into a single pandas DataFrame.

## Step 2: create the training pipeline

### Step 2-1: Select or create an image to define the training pipeline

It's unlikely that you will need to change the current image used to define the training pipeline. 
All the files used to create this image are available in the container/train folder in case you need to check the details or change anything.
	
### Step 2-2: Define training-pipeline.json

* Name this Pachyderm pipeline by changing the pipeline.name.
* Make sure the input repo matches the Pachyderm repo where you plan to commit your data.
* Under transform:
  * Define the image you want to use. The current image corresponds to files in the container/train folder and should work well as it is.
  * Command in stdin is the command that will be run when the pipeline is triggered. Make sure to change all the relevant options, in particular:
    * --git-url if you plan to make any change on the experiment code
    * --sub-dir if the file structure of your git repository is different as this one
    * --repo should match your initial Pachyderm repository
    * --model will be the name of the model on the Determined cluster (in the model registry)
    * --project should match the Pachyderm project containing the repo you are working with
  * If you have already created the Kubernetes pipeline secrets, you shouldn't change anything under secrets
  * Under pod_patch? TO COMPLETE

## Step 3: create the deployment pipeline

### Step 3-1: Select or create an image to define the deployment pipeline

If you intend to run this example as it is, you won't need to change the current deployment image, entirely defined by the files under container/deploy.

However, if you intend to train another model, you will most likely need to change a few things in the current image:
* The PyTorch handler, currently customer_churn_handler.py extends the BaseHandler class from base_handler.py, which is a default handler. Depending on your model, you may want to extend another handler, such as the ImageClassifier or the TextClassifier handlers. Those default handlers are defined here (https://github.com/pytorch/serve/tree/master/ts/torch_handler)
* We overwrote the following methods from the default BaseHandler:
  * __init__, in which we read a json file to define a dictionary of values that are used to properly scale numerical features from the data we expect to read
  * preprocess, in which we read the json request and convert it to an input that is properly scaled, encoded and that can be processed by the model
  * inference, to apply a threshold to the model predictions
  * postprocess, although that was not necessary? -- CHECK IF THIS CAN BE REMOVED
* We also included new methods to this handler. We are using them to scale numerical features of the input, as well as to encode its categorical features.
* Depending on your model and data, you may define vastly different processing operations in this file.
* If you need to import specific libraries to perform the operations you defined in the handler file, you will have to add those libraries in requirements.txt.
* If you need to add new files to this image, you'll have to update the Dockerfile accordingly.
* Finally, in deploy.py, in the create_mar_file function, you may want to change the name of the python file defining your handler, and well as specifying extra files, if your handler files relies on them.

### Step 3-2: Define deployment-pipeline.json

* Similarly to training-pipeline.json, name this Pachyderm pipeline by changing pipeline.name and make sure the input repo matches the Pachyderm repo that corresponds to your training pipeline.
* Under transform:
  * Define the image you want to use. If not running this exact example, you will most likely have to edit this image, as explained in step 3-1.
  * Command in stdin is the command that will be run when the pipeline is triggered. Make sure to change all the relevant options, in particular:
    * --deployment-name, which will be the name of the KServe InferenceService
    * --service-account-name, which is the name of the Service Account for Pachyderm access if you aren't deploying in the cloud
    * --tolerations, --resource-requests and --resource-limits, to specify resources to be used by your deployment
    * If you are deploying in the cloud, make sure to check the full list of arguments in common.py
  * If you have already created the Kubernets pipeline secrets, you shouldn't change anything under secrets
* Under pod_patch? TO COMPLETE
