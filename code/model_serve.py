## Part 5: Model Serving


from collections import ChainMap
import cdsw, numpy
from churnexplainer import ExplainedModel

#Load the model save earlier.
em = ExplainedModel(model_name='telco_linear',data_dir='/home/cdsw')

# *Note:* If you want to test this in a session, comment out the line 
#`@cdsw.model_metrics` below. Don't forget to uncomment when you
# deploy, or it won't write the metrics to the database 

@cdsw.model_metrics
# This is the main function used for serving the model. It will take in the JSON formatted arguments , calculate the probablity of 
# churn and create a LIME explainer explained instance and return that as JSON.
def explain(args):
    data = dict(ChainMap(args, em.default_data))
    data = em.cast_dct(data)
    probability, explanation = em.explain_dct(data)
    
    # Track inputs
    cdsw.track_metric('input_data', data)
    
    # Track our prediction
    cdsw.track_metric('probability', probability)
    
    # Track explanation
    cdsw.track_metric('explanation', explanation)
    
    return {
        'data': dict(data),
        'probability': probability,
        'explanation': explanation
        }

# To test this is a session, comment out the `@cdsw.model_metrics`  line,
# uncomment the and run the two rows below.
#x={"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"}
#explain(x)

## Wrap up
#
# We've now covered all the steps to **deploying and serving Models**, including the 
# requirements, limitations, and how to set up, test, and use them.
# This is a powerful way to get data scientists' work in use by other people quickly.
#
# In the next part of the project we will explore how to launch a **web application** 
# served through CML.
# Your team is busy building models to solve problems.
# CML-hosted Applications are a simple way to get these solutions in front of 
# stakeholders quickly.