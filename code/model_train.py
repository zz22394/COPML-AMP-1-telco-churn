# Part 4: Model Training


sys.path.insert(1, '/home/cdsw/code')

from pyspark.sql.types import *
from pyspark.sql import SparkSession
import sys
import os
import os
import datetime
import subprocess
import glob
import dill
import pandas as pd
import numpy as np
import cdsw
from cmlbootstrap import CMLBootstrap
import re
import time
from pyspark.sql.functions import lit

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

from lime.lime_tabular import LimeTabularExplainer

from churnexplainer import ExplainedModel, CategoricalEncoder

data_dir = '/home/cdsw'

idcol = 'customerID'
labelcol = 'Churn'
cols = (('gender', True),
        ('SeniorCitizen', True),
        ('Partner', True),
        ('Dependents', True),
        ('tenure', False),
        ('PhoneService', True),
        ('MultipleLines', True),
        ('InternetService', True),
        ('OnlineSecurity', True),
        ('OnlineBackup', True),
        ('DeviceProtection', True),
        ('TechSupport', True),
        ('StreamingTV', True),
        ('StreamingMovies', True),
        ('Contract', True),
        ('PaperlessBilling', True),
        ('PaymentMethod', True),
        ('MonthlyCharges', False),
        ('TotalCharges', False))


# This is a fail safe incase the hive table did not get created in the last step.
try:
    spark = SparkSession\
        .builder\
        .appName("PythonSQL")\
        .master("local[*]")\
        .getOrCreate()

    if (spark.sql("SELECT count(*) FROM default.telco_churn").collect()[0][0] > 0):
        df = spark.sql("SELECT * FROM default.telco_churn").toPandas()
except:
    print("Hive table has not been created")
    df = pd.read_csv(os.path.join(
        'data', 'WA_Fn-UseC_-Telco-Customer-Churn-.csv'))

# Clean and shape the data from lr and LIME
df = df.replace(r'^\s$', np.nan, regex=True).dropna().reset_index()
df.index.name = 'id'
data, labels = df.drop(labelcol, axis=1), df[labelcol]
data = data.replace({'SeniorCitizen': {1: 'Yes', 0: 'No'}})
# This is Mike's lovely short hand syntax for looping through data and doing useful things. I think if we started to pay him by the ASCII char, we'd get more readable code.
data = data[[c for c, _ in cols]]
catcols = (c for c, iscat in cols if iscat)
for col in catcols:
    data[col] = pd.Categorical(data[col])
labels = (labels == 'Yes')

# Prepare the pipeline and split the data for model training
ce = CategoricalEncoder()
X = ce.fit_transform(data)
y = labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
ct = ColumnTransformer(
    [('ohe', OneHotEncoder(), list(ce.cat_columns_ix_.values()))],
    remainder='passthrough'
)

# Experiments options
# If you are running this as an experiment, pass the cv, solver and max_iter values
# as arguments in that order. e.g. `5 lbfgs 100`.

if len(sys.argv) == 4:
    try:
        cv = int(sys.argv[1])
        solver = str(sys.argv[2])
        max_iter = int(sys.argv[3])
    except:
        sys.exit("Invalid Arguments passed to Experiment")
else:
    cv = 5
    solver = 'lbfgs'  # one of newton-cg, lbfgs, liblinear, sag, saga
    max_iter = 100

clf = LogisticRegressionCV(cv=cv, solver=solver, max_iter=max_iter)
pipe = Pipeline([('ct', ct),
                 ('scaler', StandardScaler()),
                 ('clf', clf)])

# The magical model.fit()
pipe.fit(X_train, y_train)
train_score = pipe.score(X_train, y_train)
test_score = pipe.score(X_test, y_test)
print("train", train_score)
print("test", test_score)
print(classification_report(y_test, pipe.predict(X_test)))
data[labels.name + ' probability'] = pipe.predict_proba(X)[:, 1]


# Create LIME Explainer
feature_names = list(ce.columns_)
categorical_features = list(ce.cat_columns_ix_.values())
categorical_names = {i: ce.classes_[c]
                     for c, i in ce.cat_columns_ix_.items()}
class_names = ['No ' + labels.name, labels.name]
explainer = LimeTabularExplainer(ce.transform(data),
                                 feature_names=feature_names,
                                 class_names=class_names,
                                 categorical_features=categorical_features,
                                 categorical_names=categorical_names)


# Create and save the combined Logistic Regression and LIME Explained Model.
explainedmodel = ExplainedModel(data=data, labels=labels, model_name='telco_linear',
                                categoricalencoder=ce, pipeline=pipe,
                                explainer=explainer, data_dir=data_dir)
explainedmodel.save()


# If running as as experiment, this will track the metrics and add the model trained in this
# training run to the experiment history.
#cdsw.track_metric("train_score", round(train_score, 2))
#cdsw.track_metric("test_score", round(test_score, 2))
#cdsw.track_metric("model_path", explainedmodel.model_path)
#cdsw.track_file(explainedmodel.model_path)

#write to hive
time_stamp = int(round(time.time() * 1000))
final_sdf = spark.createDataFrame(df)
final_sdf = final_sdf.withColumn("time_stamp",lit(time_stamp))
final_sdf.write.mode('append').format('parquet').saveAsTable('telco_churn_train')

# Update `lineage.yml` file
with open('/home/cdsw/lineage.yml', 'r+') as f:
    text = f.read()
    text = re.sub("training_date:\s\".*\"",'training_date: "{}"'.format(time_stamp),text)
    f.seek(0)
    f.write(text)
    f.truncate()


# Redploy the model
cml = CMLBootstrap()
project_id = cml.get_project()['id']
params = {"projectId":project_id,"latestModelDeployment":True,"latestModelBuild":True}
model_id = cml.get_models(params)[0]['id']
latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})


default_engine_details = cml.get_default_engine({})
default_engine_image_id = default_engine_details["id"]
build_model_params = {
  	"modelId": latest_model['latestModelBuild']['modelId'],
    "projectId": latest_model['latestModelBuild']['projectId'],
    "targetFilePath": "code/model_serve.py",
    "targetFunctionName": "explain",
    "engineImageId": default_engine_image_id,
    "kernel": "python3",
    "examples": latest_model['latestModelBuild']['examples'],
    "cpuMillicores": 1000,
    "memoryMb": 2048,
    "nvidiaGPUs": 0,
    "replicationPolicy": {"type": "fixed", "numReplicas": 1},
    "environment": {}}

cml.rebuild_model(build_model_params)

# Wrap up

# We've now covered all the steps to **running Experiments**.
#
# Notice also that any script that will run as an Experiment can also be run as a Job or in a Session.
# Our provided script can be run with the same settings as for Experiments.
# A common use case is to **automate periodic model updates**.
# Jobs can be scheduled to run the same model training script once a week using the latest data.
# Another Job dependent on the first one can update the model parameters being used in production
# if model metrics are favorable.
