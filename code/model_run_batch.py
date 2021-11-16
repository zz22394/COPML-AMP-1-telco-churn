import cdsw, time, os, random, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from cmlbootstrap import CMLBootstrap
import seaborn as sns
import copy
from pyspark.sql.functions import lit


## Set the model ID
# Get the model id from the model you deployed in step 5. These are unique to each 
# model on CML.

cml = CMLBootstrap()
project_id = cml.get_project()['id']
params = {"projectId":project_id,"latestModelDeployment":True,"latestModelBuild":True}
model_id = cml.get_models(params)[0]['id']
#HOST = os.getenv("CDSW_API_URL").split(":")[0] + "://" + os.getenv("CDSW_DOMAIN")

# Grab the data from Hive.
from pyspark.sql import SparkSession
from pyspark.sql.types import *
spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .getOrCreate()

df = spark.sql("SELECT * FROM default.telco_churn").toPandas()

latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})

Model_CRN = latest_model ["crn"]
Deployment_CRN = latest_model["latestModelDeployment"]["crn"]
  
# Get 500 samples  
df_sample = df.sample(500)

df_sample_clean = df_sample.\
  replace({'SeniorCitizen': {"1": 'Yes', "0": 'No'}}).\
  replace(r'^\s$', np.nan, regex=True).\
  dropna()

def predict_dataframe(record):
  response = cdsw.call_model(latest_model["accessKey"],record.drop(['Churn','customerID']).to_dict())
  prediction = "Yes" if response["response"]["prediction"]["probability"] >= 0.5 else "No"
  return pd.Series([response["response"]["uuid"],prediction,record['customerID']])

def update_metrics(record):
  cdsw.track_delayed_metrics({"actual_value":record['prediction']}, record['uuid'])

prediciton_df = df_sample_clean.apply(predict_dataframe,axis=1)
prediciton_df.columns = ['uuid','prediction','customerID']

final_df = pd.merge(df_sample_clean,prediciton_df,left_on=["customerID"],right_on= ['customerID'], how='left')  

#write to hive
time_stamp = int(round(time.time() * 1000))
final_sdf = spark.createDataFrame(final_df)
final_sdf = final_sdf.withColumn("time_stamp",lit(time_stamp))

final_sdf.write.mode('append').format('parquet').saveAsTable('telco_churn_batch')

# update metrics

final_df.apply(update_metrics,axis=1)
accuracy = classification_report(final_df['Churn'].to_numpy(),final_df['prediction'].to_numpy(),output_dict=True)["accuracy"]
cdsw.track_aggregate_metrics({"accuracy": accuracy}, int(round(time.time() * 1000)) , int(round(time.time() * 1000))+100, model_deployment_crn=Deployment_CRN)
