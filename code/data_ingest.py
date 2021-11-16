# Part 1: Data Ingest

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *



spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .getOrCreate()

# **Note:**
# Our file isn't big, so running it in Spark local mode is fine but you can add the following config
# if you want to run Spark on the kubernetes cluster
#
# > .config("spark.yarn.access.hadoopFileSystems",os.getenv['STORAGE'])\
#
# and remove `.master("local[*]")\`
#

# Since we know the data already, we can add schema upfront. This is good practice as Spark will
# read *all* the Data if you try infer the schema.

schema = StructType(
    [
        StructField("customerID", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("SeniorCitizen", StringType(), True),
        StructField("Partner", StringType(), True),
        StructField("Dependents", StringType(), True),
        StructField("tenure", DoubleType(), True),
        StructField("PhoneService", StringType(), True),
        StructField("MultipleLines", StringType(), True),
        StructField("InternetService", StringType(), True),
        StructField("OnlineSecurity", StringType(), True),
        StructField("OnlineBackup", StringType(), True),
        StructField("DeviceProtection", StringType(), True),
        StructField("TechSupport", StringType(), True),
        StructField("StreamingTV", StringType(), True),
        StructField("StreamingMovies", StringType(), True),
        StructField("Contract", StringType(), True),
        StructField("PaperlessBilling", StringType(), True),
        StructField("PaymentMethod", StringType(), True),
        StructField("MonthlyCharges", DoubleType(), True),
        StructField("TotalCharges", DoubleType(), True),
        StructField("Churn", StringType(), True)
    ]
)

# Now we can read in the data from Cloud Storage into Spark...

storage = os.environ['STORAGE']
hadoop_user = os.environ['HADOOP_USER_NAME']

telco_data = spark.read.csv(
    "{}/user/{}/data/churn/WA_Fn-UseC_-Telco-Customer-Churn-.csv".format(
        storage,hadoop_user),
    header=True,
    schema=schema,
    sep=',',
    nullValue='NA'
)

# ...and inspect the data.

telco_data.show()

telco_data.printSchema()

# The data is now available in hive

spark.sql("show databases").show()

spark.sql("show tables in default").show()

# Create the Hive table
# This is here to create the table in Hive used be the other parts of the project, if it
# does not already exist.

if ('telco_churn' not in list(spark.sql("show tables in default").toPandas()['tableName'])):
    print("creating the telco_churn database")
    telco_data\
        .write.format("parquet")\
        .mode("overwrite")\
        .saveAsTable(
            'default.telco_churn'
        )

# Show the data in the hive table
spark.sql("select * from default.telco_churn").show()

# To get more detailed information about the hive table you can run this:
spark.sql("describe formatted default.telco_churn").toPandas()

# Other ways to access data

# To access data from other locations, refer to the
# [CML documentation](https://docs.cloudera.com/machine-learning/cloud/import-data/index.html).

# Scheduled Jobs
#
# One of the features of CML is the ability to schedule code to run at regular intervals,
# similar to cron jobs. This is useful for **data pipelines**, **ETL**, and **regular reporting**
# among other use cases. If new data files are created regularly, e.g. hourly log files, you could
# schedule a Job to run a data loading script with code like the above.

# > Any script [can be scheduled as a Job](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job.html).
# > You can create a Job with specified command line arguments or environment variables.
# > Jobs can be triggered by the completion of other jobs, forming a
# > [Pipeline](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-pipeline.html)
# > You can configure the job to email individuals with an attachment, e.g. a csv report which your
# > script saves at: `/home/cdsw/job1/output.csv`.

# Try running this script `1_data_ingest.py` for use in such a Job.

