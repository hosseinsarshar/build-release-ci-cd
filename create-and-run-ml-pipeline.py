#!/usr/bin/env python
# coding: utf-8

# # Generate ML Pipeline at DevOps Build Pipeline
# 
# In this notebook, we aim to make some modifications to the previous notebook () so that Azure DevOps Build Pipeline can generate a new ML Pipeline every time the master branch of the GitHub repo is changed.
# 
# This is an important step to build a fully automated CI/CD pipeline for our ML project. So the senario works like this:
# 
# As a new code hits the master branch (this time we like to trigger the build Pipeline at the CI "merge into the Master branch") that hosts our code for the training pipeline, we like to execute the code to generate a new ML Pipeline with the new code. The ML Pipeline then generates a new ML model. The ML models is evaluated and if the accuracy is higher than the existing model, it is pushed into production.
# 
# One major difference in this senario is that we have to generate the ML Pipeline from the Ubunto computer within Azure DevOps. That computer doesn't have access to our Azure's subscription and also we don't want to manually go through the authentication process. We want this to be automatic. Therefore, we need to create a mechanisim that the machine can log in in absence of us to access our Azure environment and in particular our Azure Workspace.
# 
# One way to do this is to create a user name of type Service Principle. This user name is designed to let applications authenticate into Azure. So first we need to create a Service Principle Account. The steps are provided here: https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal
# 
# Save the following pices of information: Application ID, Tenant ID, Secret Key and replace them in the code below:

# Like always we import some packages related to the Azure ML:

import azureml.core
from azureml.core import Workspace, Experiment, Datastore
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

import argparse

print("Pipeline SDK-specific imports completed")

parser = argparse.ArgumentParser()

parser.add_argument("--tenant_id",  type=str, dest="tenant_id")
parser.add_argument("--application_id",  type=str, dest="application_id")
parser.add_argument("--app_secret",  type=str, dest="app_secret")
parser.add_argument("--subscription_id",  type=str, dest="subscription_id")
parser.add_argument("--resource_group",  type=str, dest="resource_group")
parser.add_argument("--workspace_name",  type=str, dest="workspace_name")
parser.add_argument("--workspace_region",  type=str, dest="workspace_region")
parser.add_argument("--model_name", type=str, dest="model_name")
parser.add_argument("--cluster_name_cpu", type=str, dest="cluster_name_cpu")
parser.add_argument("--cluster_name_gpu", type=str, dest="cluster_name_gpu")
parser.add_argument("--pipeline_experiment_name", type=str, dest="pipeline_experiment_name")
parser.add_argument("--pipeline_name", type=str, dest="pipeline_name")

args = parser.parse_args()

tenant_id = args.tenant_id
application_id = args.application_id
app_secret = args.app_secret
subscription_id = args.subscription_id
resource_group = args.resource_group
workspace_name = args.workspace_name
workspace_region = args.workspace_region
model_name = args.model_name
cluster_name_cpu = args.cluster_name_cpu
cluster_name_gpu = args.cluster_name_gpu
pipeline_experiment_name = args.pipeline_experiment_name
pipeline_name = args.pipeline_name

from azureml.core.authentication import ServicePrincipalAuthentication

service_principal = ServicePrincipalAuthentication(
        tenant_id=tenant_id,
        service_principal_id=application_id,
        service_principal_password=app_secret)


ws = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            auth=service_principal)

# Retrieve the pointer to the default Blob storage.

def_blob_store = Datastore(ws, "workspaceblobstore")
print("Blobstore's name: {}".format(def_blob_store.name))

blob_input_data = DataReference(
    datastore=def_blob_store,
    data_reference_name="mnist_datainput",
    path_on_datastore="mnist_datainput")

print("DataReference object created")

# Create a GPU cluster of type NV6 with 1 node. (due to subscription's limitations we stick to 1 node)

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# choose a name for your cluster

try:
    compute_target_cpu = ComputeTarget(workspace=ws, name=cluster_name_cpu)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    # CPU: Standard_D3_v2
    # GPU: Standard_NV6
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', 
                                                           max_nodes=1,
                                                           min_nodes=1)

    # create the cluster
    compute_target_cpu = ComputeTarget.create(ws, cluster_name_cpu, compute_config)

    compute_target_cpu.wait_for_completion(show_output=True)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target_cpu.get_status().serialize())

# In[9]:

# try:
#     compute_target_gpu = ComputeTarget(workspace=ws, name=cluster_name_gpu)
#     print('Found existing compute target.')
# except ComputeTargetException:
#     print('Creating a new compute target...')
#     # CPU: Standard_D3_v2
#     # GPU: Standard_NV6
#     compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NV6', 
#                                                            max_nodes=1,
#                                                            min_nodes=1)
# 
#     # create the cluster
#     compute_target_gpu = ComputeTarget.create(ws, cluster_name_gpu, compute_config)
# 
#     compute_target_gpu.wait_for_completion(show_output=True)
# 
# # use get_status() to get a detailed status for the current cluster. 
# print(compute_target_gpu.get_status().serialize())


cts = ws.compute_targets
for ct in cts:
    print(ct)


# In[11]:


processed_mnist_data = PipelineData("processed_mnist_data", datastore=def_blob_store)
processed_mnist_data


# In[12]:


from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

# create a new runconfig object
run_config = RunConfiguration()

# enable Docker 
run_config.environment.docker.enabled = True

# set Docker base image to the default CPU-based image
run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE

# use conda_dependencies.yml to create a conda environment in the Docker image for execution
run_config.environment.python.user_managed_dependencies = False

# specify CondaDependencies obj
run_config.environment.python.conda_dependencies = CondaDependencies.create(pip_packages=['azureml-sdk',
                                                                                          'numpy'])


# In[13]:


# source directory
source_directory = 'DataExtraction'

extractDataStep = PythonScriptStep(
    script_name="extract.py", 
    arguments=["--output_extract", processed_mnist_data],
    outputs=[processed_mnist_data],
    compute_target=compute_target_cpu, 
    source_directory=source_directory,
    runconfig=run_config)

print("Data Extraction Step created")


# In[14]:


from azureml.train.dnn import TensorFlow

source_directory = 'Training'
est = TensorFlow(source_directory=source_directory,
                 compute_target=compute_target_cpu,
                 entry_script='train.py', 
                 use_gpu=False, 
                 framework_version='1.13')


# In[15]:

from azureml.pipeline.steps import EstimatorStep

trainingStep = EstimatorStep(name="Training-Step",
                             estimator=est,
                             estimator_entry_script_arguments=["--input_data_location", processed_mnist_data,
                                                               '--batch-size', 50,
                                                               '--first-layer-neurons', 300,
                                                               '--second-layer-neurons', 100,
                                                               '--learning-rate', 0.01,
                                                               "--release_id", 0,
                                                               '--model_name', model_name],
                             runconfig_pipeline_params=None,
                             inputs=[processed_mnist_data],
                             compute_target=compute_target_cpu)


print("Model Training Step is Completed")


# In[16]:


# source directory
source_directory = 'RegisterModel'

modelEvalReg = PythonScriptStep(
    name="Evaluate and Register Model",
    script_name="evaluate_model.py", 
    arguments=["--release_id", 0,
               '--model_name', model_name],
    compute_target=compute_target_cpu, 
    source_directory=source_directory,
    runconfig=run_config,
    allow_reuse=False)

modelEvalReg.run_after(trainingStep)
print("Model Evaluation and Registration Step is Created")


# In[17]:


from azureml.pipeline.core import Pipeline
from azureml.core import Experiment
pipeline = Pipeline(workspace=ws, steps=[extractDataStep, trainingStep, modelEvalReg])
pipeline_run = Experiment(ws, pipeline_experiment_name).submit(pipeline)


# In[18]:



# In[19]:


pipeline_run.id


# In[20]:
pipeline_run.wait_for_completion(show_output=True, raise_on_error=True)

published_pipeline = pipeline_run.publish_pipeline(name=pipeline_name, 
                                                   description="Steps are: data preparation, training, model validation and model registration", 
                                                   version="0.1", 
                                                   continue_on_step_failure=False)


# In[ ]:




