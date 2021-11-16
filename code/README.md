# COPML Example 1 - Churn Prediction
This repoistory is used as the first example in the [Continuous Operations for Production Machine Learning](https://linktothis.com) (COPML) document that provides a framework for maintaning machine learning projects in production. The goal is to build a classifier model using Logistic Regression to predict the churn probability for a group of customers from a telecoms company. 

This section follows the standard workflow for implementing machine learning projects. As this project should be deployed as an AMP, most of the artifacts should already be deployed. 

_Note: If you did not deploy the project as an AMP, you need to initialise the project first._

**Initialize the Project**

There are a couple of steps needed at the start to configure the Project and Workspace settings so each step will run sucessfully. If you did **not** deploy this project as an AMP, you **must** run the project bootstrap before running other steps. If you just want to launch the project with all the artefacts already built without going through each step manually, then you can also redploy deploy the complete project and an AMP.

*Project bootstrap*

Open the file `_bootstrap.py` in a normal workbench python3 session. You only need a 1 vCPU / 2 GiB instance. Once the session is loaded, click **Run > Run All Lines**. This will file will create an Environment Variable for the project called **STORAGE**, which is the root of default file storage location for the Hive Metastore in the DataLake (e.g. `s3a://my-default-bucket`). It will also upload the data used in the project to `$STORAGE/datalake/data/churn/`. The original file comes as part of this git repo in the `data` folder.

## Step 1: Clarify Business Requirements
A fictitious telco business wants to predict which customers are likely to churn in order to be able to reduce the current churn rate (e.g. from 10% to 5%). In order to fulfil this objective the business needs to be able to predict the probability of any of its customers churning. Those deemed as ‘high risk’ can be entered into some sort of remediation process. For example, an offer of a free data or text package, or whatever the business has decided is the best course of action for retention. 

The requirement from the business is to indentify the customers who are at risk of churning and make that available in a data table that other teams can then use to implement the remdial action. 


## Step 2: Assess Available Data
The data science team assesses the customer-related data that’s available in the organization’s data warehouse and any relevant data that’s been made available from other sources and confirms it is accessible to the systems the data science teams will use for EDA and model training. In this particular case, the customer-related data includes demographics information, usage data, product mix, monthly charges, total charges etc. It is important that the available data includes both customers that have left (i.e. those that have churned) as well as existing customers who have not churned. 

### Ingest Data
The `code/data_ingest.py` script will read in the data csv from the file uploaded to the object store (s3/adls) setup during the bootstrap and create a managed table in Hive. This is all done using Spark.

Open `code/data_ingest.py` in a Workbench session: python3, 1 CPU, 2 GB. Run the file. For this project, this is the only data that is used and is considered sufficient for the required task.

## Step 3: Develop the Data Science Plan
For this project, the plan is to build a binary classifier model that can classify a customer as a churn risk or not. This model can be applied to any existing customer. The data science team will create notebooks for conducting exploratory data analysis and for the machine learning model build.

### Explore Data
The project includes a Jupyter Notebook that does some basic data exploration and visualistaion. It is to show how this would be part of the data science workflow.

![data](../images/data.png)

Open a Jupyter Notebook session (rather than a work bench): python3, 1 CPU, 2 GB and open the `notebooks/data_exploration.ipynb` file. 

At the top of the page click **Cells > Run All**.

### Model Building
There is also a Jupyter Notebook to show the process of selecting and building the model to predict churn. It also shows more details on how the LIME model is created and a bit more on what LIME is actually doing.

Open a Jupyter Notebook session (rather than a work bench): python3, 1 CPU, 2 GB and open the `notebooks/model_building.ipynb` file. 

At the top of the page click **Cells > Run All**.

### Model Training
A model pre-trained is saved with the repo has been and placed in the `models` directory. If you want to retrain the model, open the `code/model_train.py` file in a workbench session: python3 1 vCPU, 2 GiB and run the file. The newly model will be saved in the models directory 
named `telco_linear`. The other way of running the model training process is by using a Job.

***Jobs***
The **[Jobs](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job.html)**
feature allows for adhoc, recurring and depend jobs to run specific scripts. To run this model training process as a job, create a new job by going to the Project window and clicking _Jobs > New Job_ and entering the following settings:
* **Name** : Train Model
* **Script** : code/model_train.py
* **Arguments** : _Leave blank_
* **Kernel** : Python 3
* **Schedule** : Manual
* **Engine Profile** : 1 vCPU / 2 GiB
The rest can be left as is. Once the job has been created, click **Run** to start a manual run for that job.

## Step 4: Model Deployment 
The model needs to be deployed into the place where its output is available to the relevant teams so they can fulfil the business requirement. Our illustrative code runs a batch update job and also builds a real time model that can make new churn predictions for customers on the fly. The batch job can be run once a week and its output updates a hive table the can be accessed by the team that’s tasked with implementing the customer retention strategy.

The metrics for the batch job are stored with a deployed model, therefore you need to first deploy the real-time model and then deploy the batch job.

### Serve Model
The **[Models](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-creating-and-deploying-a-model.html)** 
is used top deploy a machine learning model into production for real-time prediction. To deploy the model trailed in the previous step, from  to the Project page, click **Models > New Model** and create a new model with the following details:

* **Name**: Churn Model Endpoint
* **Description**: Churn explained model endpoint
* **File**: code/model_serve.py
* **Function**: explain
* **Input**: 
```
{
	"StreamingTV": "No",
	"MonthlyCharges": 70.35,
	"PhoneService": "No",
	"PaperlessBilling": "No",
	"Partner": "No",
	"OnlineBackup": "No",
	"gender": "Female",
	"Contract": "Month-to-month",
	"TotalCharges": 1397.475,
	"StreamingMovies": "No",
	"DeviceProtection": "No",
	"PaymentMethod": "Bank transfer (automatic)",
	"tenure": 29,
	"Dependents": "No",
	"OnlineSecurity": "No",
	"MultipleLines": "No",
	"InternetService": "DSL",
	"SeniorCitizen": "No",
	"TechSupport": "No"
}
```
* **Kernel**: Python 3
* **Engine Profile**: 1vCPU / 2 GiB Memory

Leave the rest unchanged. Click **Deploy Model** and the model will go through the build 
process and deploy a REST endpoint. Once the model is deployed, you can test it is working 
from the model Model Overview page.

_**Note: This is important**_

Once the model is deployed, you must disable the additional model authentication feature. In the model settings page, untick **Enable Authentication**.

![disable_auth](../images/disable_auth.png)

### Model Batch Job
The model batch job runs batch inference on a new, randomly selected data set and creates a new hive table. This job will run once a month.

To run this batch inference job, create a new job by going to the Project window and clicking _Jobs > New Job_ and entering the following settings:
* **Name** : Run Model Batch
* **Script** : code/model_run_batch.py
* **Arguments** : _Leave blank_
* **Kernel** : Python 3
* **Schedule** : Reccuring - Monthly
* **Engine Profile** : 1 vCPU / 2 GiB
The rest can be left as is. Once the job has been created, click **Run** to start a manual run for that job.

## Step 5: Model Operations
The model’s performance needs to be checked periodically. A good way to do this is to examine a proportion of the customers that the model made predictions for and assess the accuracy of those predictions  e.g. how many of those predicted to churn actually did? Alternatively, what are the precision and recall values of the model?  If the model performance falls below an acceptable level then it will be necessary to retrain the model.  The performance metrics in this scenario are relatively straightforward, but even one this simple can be hard to put into production within an enterprise context. For example, the choice about how often to assess the model’s performance is very dependent on the business circumstances and the consequences of inaccurate predictions or model drift. In this particular scenario, while timing matters it’s not the most important thing. Churn has business implications in the medium- to long-term but its short-term impact is limited. Therefore a delay of a few days for a performance assessment would probably be acceptable.

### Check Model Job
The check model job will retrieve the most recent `accuracy` metric fof the , randomly selected data set and creates a new hive table. This job will run once a month.

To run this batch inference job, create a new job by going to the Project window and clicking _Jobs > New Job_ and entering the following settings:
* **Name** : Check Model
* **Script** : code/model_check.py
* **Arguments** : _Leave blank_
* **Kernel** : Python 3
* **Schedule** : Reccuring - Monthly
* **Engine Profile** : 1 vCPU / 2 GiB
The rest can be left as is. Once the job has been created, click **Run** to start a manual run for that job.

## Explainability
This project has now gone through the steps to meet the business requirements described in step one. On top that, the model can then be interpreted using [LIME](https://github.com/marcotcr/lime). The deployed model has both the Logistic Regression and LIME models combined. There is an optional step to deploy a basic flask based web application that will let you interact with the real-time model to see which factors in the data have the most influence on the churn probability.

### Application Deployment
The next step is to deploy the Flask application. The **[Applications](https://docs.cloudera.com/machine-learning/cloud/applications/topics/ml-applications.html)** feature is still quite new for CML. For this project it is used to deploy a web based application that interacts with the underlying model created in the previous step.

From the Go to the **Applications** section and select "New Application" with the following:
* **Name**: Churn Analysis App
* **Subdomain**: churn-app _(note: this needs to be unique, so if you've done this before, pick a more random subdomain name)_
* **Script**: code/application.py
* **Kernel**: Python 3
* **Engine Profile**: 1vCPU / 2 GiB Memory


After the Application deploys, click on the blue-arrow next to the name. The initial view is a table of randomly selected from the dataset. This shows a global view of which features are most important for the predictor model. The reds show incresed importance for preditcting a cusomter that will churn and the blues for for customers that will not.

![table_view](../images/table_view.png)

Clicking on any single row will show a "local" interpreted model for that particular data point instance. Here you can see how adjusting any one of the features will change the instance's churn prediction.

![single_view_1](../images/single_view_1.png)

Changing the InternetService to DSL lowers the probablity of churn. *Note: this does not mean that changing the Internet Service to DSL cause the probability to go down, this is just what the model would predict for a customer with those data points*

![single_view_2](../images/single_view_2.png)