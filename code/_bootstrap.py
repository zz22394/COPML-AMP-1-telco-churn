## Part 0: Bootstrap File
# You need to at the start of the project. It will install the requirements, creates the 
# STORAGE environment variable and copy the data from 
# raw/WA_Fn-UseC_-Telco-Customer-Churn-.csv into /datalake/data/churn of the STORAGE 
# location.

# The STORAGE environment variable is the Cloud Storage location used by the DataLake 
# to store hive data. On AWS it will s3a://[something], on Azure it will be 
# abfs://[something] and on CDSW cluster, it will be hdfs://[something]

# Install the requirements
!pip3 install --progress-bar off -r requirements.txt
!pip3 install --upgrade git+https://github.com/fastforwardlabs/cmlbootstrap#egg=cmlbootstrap

# Create the directories and upload data
from cmlbootstrap import CMLBootstrap
import os

# Instantiate API Wrapper
cml = CMLBootstrap()

# Set the STORAGE environment variable
# Set the STORAGE environment variable
try : 
  storage=os.environ["STORAGE"]
except:
  try:
    storage = cml.get_cloud_storage()
    storage_environment_params = {"STORAGE":storage}
    storage_environment = cml.create_environment_variable(storage_environment_params)
    os.environ["STORAGE"] = storage
  except:
    storage = "/user/" + os.environ["HADOOP_USER_NAME"]

# Upload the data to the cloud storage
!hdfs dfs -mkdir -p $STORAGE/user
!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME
!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME/data
!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME/data/churn
!hdfs dfs -copyFromLocal /home/cdsw/data/WA_Fn-UseC_-Telco-Customer-Churn-.csv $STORAGE/user/$HADOOP_USER_NAME/data/churn/WA_Fn-UseC_-Telco-Customer-Churn-.csv