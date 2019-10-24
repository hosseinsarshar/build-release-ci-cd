#!/usr/bin/env python
# coding: utf-8

# In[1]:


import azureml.core
from azureml.core import Workspace, Experiment, Datastore
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os
# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

from azureml.core.compute import AksCompute
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.image import ContainerImage

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--script-root', type=str, dest='script_root', help='Root script')
parser.add_argument('--tenant_id', type=str, dest='tenant_id', help="Provide tenant_id")
parser.add_argument('--application_id', type=str, dest='application_id', help="Provide application_id")
parser.add_argument('--subscription_id', type=str, dest='subscription_id', help="Provide subscription_id")
parser.add_argument('--app_secret', type=str, dest='app_secret', help="Provide app_secret")
parser.add_argument('--resource_group', type=str, dest='resource_group', help="Provide resource_group")
parser.add_argument('--workspace_name', type=str, dest='workspace_name', help="Provide workspace_name")
parser.add_argument('--workspace_region', type=str, dest='workspace_region', help="Provide workspace_region")
parser.add_argument('--model_name', type=str, dest='model_name', help="Provide model_name")
parser.add_argument('--image_name', type=str, dest='image_name', help="Provide image_name")
parser.add_argument('--aks_name', type=str, dest='aks_name', help="Provide aks_name")
parser.add_argument('--aks_service_name', type=str, dest='aks_service_name', help="Provide aks_service_name")

args = parser.parse_args()

script_root = args.script_root
tenant_id = args.tenant_id
application_id = args.application_id
subscription_id = args.subscription_id
app_secret = args.app_secret
resource_group = args.resource_group
workspace_name = args.workspace_name
workspace_region = args.workspace_region
model_name = args.model_name
image_name = args.image_name
aks_name = args.aks_name
aks_service_name = args.aks_service_name

print(f'Current directory: {os.getcwd()}')
print(f'script_root: {script_root}')
print(f'tenant_id: {tenant_id}')
print(f'application_id: {application_id}')
print(f'subscription_id: {subscription_id}')
print(f'app_secret: {app_secret}')
print(f'resource_group: {resource_group}')
print(f'workspace_name: {workspace_name}')
print(f'workspace_region: {workspace_region}')
print(f'model_name: {model_name}')
print(f'image_name: {image_name}')
print(f'aks_name: {aks_name}')
print(f'aks_service_name: {aks_service_name}')

os.chdir(script_root)

print("Pipeline SDK-specific imports completed")

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

from azureml.core.model import Model

model_root = Model(ws, name=model_name) # Model.get_model_path(model_name, _workspace=ws)

from azureml.core.runconfig import CondaDependencies

cd = CondaDependencies.create()
cd.add_conda_package('numpy')

# Adds all you need for TF
cd.add_tensorflow_conda_package() # core_type='cpu', version='1.13')
cd.save_to_file(base_directory='./score', conda_file_path='myenv.yml')

print(cd.serialize_to_string())
print(os.getcwd())
os.chdir('./score')
image_config = ContainerImage.image_configuration(execution_script = "score.py",
                                                  runtime = "python",
                                                  conda_file = "myenv.yml",
                                                  description = "MNIST TF Model",
                                                  tags = {'--release-id': "0", 'type': "TF deployment"})

image = ContainerImage.create(name = image_name,
                              # this is the model object
                              models = [model_root],
                              image_config = image_config,
                              workspace = ws)

image.wait_for_creation(show_output = True)

from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException

try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print('Found existing AKS service')
except ComputeTargetException:
    prov_config = AksCompute.provisioning_configuration()

    print('Creating a new AKS service...')
    aks_target = ComputeTarget.create(workspace = ws, 
                                  name = aks_name, 
                                  provisioning_configuration = prov_config)

    aks_target.wait_for_completion(show_output = True)
    print(aks_target.provisioning_state)
    print(aks_target.provisioning_errors)

# use get_status() to get a detailed status for the current cluster. 
print(aks_target.get_status())

#Set the web service configuration (using default here)
aks_config = AksWebservice.deploy_configuration(autoscale_enabled=True,
                                                autoscale_min_replicas=1,
                                                autoscale_max_replicas=2,
                                                collect_model_data=True,
                                                enable_app_insights=True)

try:
    aks_service = Webservice(name = aks_service_name, workspace = ws)
    print('Found the webservice, deleting the service to add a new one')
    aks_service.delete()
    print('Old webservice is deleted')
except Exception:
    print("This webservice doesn't exist")
finally:
    print('Deploying the new web service')
    aks_service = Webservice.deploy_from_image(workspace = ws, 
                                           name = aks_service_name,
                                           image = image,
                                           deployment_config = aks_config,
                                           deployment_target = aks_target)

    aks_service.wait_for_deployment(show_output = True)
    print('This webservice is deployed')
