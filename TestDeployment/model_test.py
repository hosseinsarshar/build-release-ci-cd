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
import numpy as np
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--script-root', type=str, dest='script_root', help='Root script')
parser.add_argument('--tenant_id', type=str, dest='tenant_id', help="Provide tenant_id")
parser.add_argument('--application_id', type=str, dest='application_id', help="Provide application_id")
parser.add_argument('--subscription_id', type=str, dest='subscription_id', help="Provide subscription_id")
parser.add_argument('--app_secret', type=str, dest='app_secret', help="Provide app_secret")
parser.add_argument('--resource_group', type=str, dest='resource_group', help="Provide resource_group")
parser.add_argument('--workspace_name', type=str, dest='workspace_name', help="Provide workspace_name")
parser.add_argument('--workspace_region', type=str, dest='workspace_region', help="Provide workspace_region")
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
aks_service_name = args.aks_service_name

os.chdir(script_root)

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

aks_service = Webservice(name = aks_service_name, workspace = ws)

test_data = np.load('./test_data.npy')

test_samples = json.dumps({"data": test_data.tolist()})
test_samples = bytes(test_samples, encoding='utf8')

# predict using the deployed model
result = aks_service.run(input_data=test_samples)

assert len(result) == len(test_data)

print(f'Model is working properly at the {aks_service_name} environment')
