import os
from azureml.core import Model, Run
import argparse
import numpy as np

def get_run():
    global run
    global exp
    global ws

    run = Run.get_context()
    exp = run.experiment
    ws = run.experiment.workspace

def get_args():
    global model_name
    global release_id

    parser = argparse.ArgumentParser("evaluate_model")
    parser.add_argument('--release_id', type=int, dest='release_id', default=0, help='Release ID')
    parser.add_argument('--model_name', type=str, dest='model_name', default='tf_mnist_pipeline.model', help='Model Name')

    args = parser.parse_args()
    model_name = args.model_name
    release_id = args.release_id

    print(f"Argument release_id: {model_name}")
    print(f"Argument model_name: {release_id}")
   
def evaluate_model():
    all_runs = exp.get_runs(properties={"release_id": release_id, "run_type": "train"}, include_children=True)

    new_model_run = next(all_runs)
    new_model_run_id = new_model_run.id
    print(f'New Run found with Run ID of: {new_model_run_id}')

    new_model_run = Run(exp, run_id=new_model_run_id)
    new_model_acc = new_model_run.get_metrics().get("final-accuracy")

    try:
        # Get most recently registered model, we assume that
        # is the model in production.
        # Download this model and compare it with the recently
        # trained model by running test with same data set.
        model_list = Model.list(ws)
        production_model = next(
            filter(
                lambda x: x.created_time == max(
                    model.created_time for model in model_list),
                model_list,
            )
        )
        production_model_run_id = production_model.tags.get("run_id")
        run_list = exp.get_runs()

        # Get the run history for both production model and
        # newly trained model and compare final-accuracy
        production_model_run = Run(exp, run_id=production_model_run_id)

        production_model_acc = production_model_run.get_metrics().get("final-accuracy")
        
        print("Current Production model accuracy: {}, New trained model accuracy: {}".format(production_model_acc, new_model_acc))

        promote_new_model = False
        if new_model_acc < production_model_acc:
            promote_new_model = True
            print("New trained model performs better, thus it will be registered")
    except Exception:
        promote_new_model = True
        print("This is the first model to be trained, \
            thus nothing to evaluate for now")
    
    return promote_new_model, new_model_run, new_model_acc

def register_model(promote_new_model, new_model_run, new_model_acc,):
    tags = {}
    tags['run_id'] = new_model_run.id
    tags['final_accuracy'] = np.float(new_model_acc)
    # Un-comment this if you like to register the model with the highest accuracy.
    # if promote_new_model:
    if True:
        model_path = os.path.join('outputs/model', model_name)
        new_model_run.register_model(
            model_name=model_name,
            model_path='outputs/model',
            properties={"release_id": release_id},
            tags=tags)
        print("Registered new model!")
    else:
        print("The model in production has higher accuracy!")

if __name__ == '__main__':
    get_run()
    get_args()
    promote_new_model, new_model_run, new_model_acc = evaluate_model()
    register_model(promote_new_model, new_model_run, new_model_acc)
