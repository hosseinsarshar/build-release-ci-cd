# build-release-ci-cd

This repo is built by Hossein Sarshar inspired from [Microsfot MLOps](https://github.com/Microsoft/MLOps) repo. Should you have any questions, feel free to email me at hossein.sarshar@gmail.com or tweet me at [classicboyir](https://twitter.com/classicboyir).

This repo is backed by Build Pipeline:

[![Build status](https://dev.azure.com/hosarsha/build-release-pipeline/_apis/build/status/build-release-pipeline-CI)](https://dev.azure.com/hosarsha/build-release-pipeline/_build/latest?definitionId=2)

This repo contains a set of procedures that helps you build the full CI/CD pipeline based on MLOps principles. You should follow the notebooks in this order:

1. To become familiar with the ML Pipelines in Azure ML read this notebook: [MLPipeline_MNIST.ipynb](https://github.com/classicboyir/build-release-ci-cd/blob/master/MLPipeline_MNIST.ipynb)
2. Then you need to integrate the ML Pipeline within your DevOps Build Pipeline. For that, check out this notebook: [Service-Principle-Login.ipynb](https://github.com/classicboyir/build-release-ci-cd/blob/master/Service-Principle-Login.ipynb)
3. To find out the steps required in the Deployment time, check out this notebook: [DeployModel.ipynb](https://github.com/classicboyir/build-release-ci-cd/blob/master/DeployModel.ipynb)
4. Then, it's time to work on your entire build and release Pipeline, for the instructions, read: [Create_Release_Pipeline_Instructions.ipynb](https://github.com/classicboyir/build-release-ci-cd/blob/master/Create_Release_Pipeline_Instructions.ipynb)
5. Now, have everything you needed. Check your release pipeline and enjoy it.
