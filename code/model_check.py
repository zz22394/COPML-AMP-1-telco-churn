# # Check Model
# This file should be run in a job that will periodically check the current model's accuracy and trigger the 
# model retrain job if its below the required thresh hold. 

import cdsw, time, os
import pandas as pd
from sklearn.metrics import classification_report
from cmlbootstrap import CMLBootstrap

cml = CMLBootstrap()

# replace this with these values relevant values from the project

for job in cml.get_jobs():
  if job['name'] == "Train Model":
    job_id = job['name']

project_id = cml.get_project()['id']
params = {"projectId":project_id,"latestModelDeployment":True,"latestModelBuild":True}
model_id = cml.get_models(params)[0]['id']


latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})

Model_CRN = latest_model ["crn"]
Deployment_CRN = latest_model["latestModelDeployment"]["crn"]

# Read in the model metrics dict.
model_metrics = cdsw.read_metrics(model_crn=Model_CRN,model_deployment_crn=Deployment_CRN)

# This is a handy way to unravel the dict into a big pandas dataframe.
metrics_df = pd.io.json.json_normalize(model_metrics["metrics"])

latest_aggregate_metric = metrics_df.dropna(subset=["metrics.accuracy"]).sort_values('startTimeStampMs')[-1:]["metrics.accuracy"]


if latest_aggregate_metric.to_list()[0] < 0.6:
  print("model is below threshold, retraining")
  cml.start_job(job_id,{})
  
else:
  print("model does not need to be retrained")
