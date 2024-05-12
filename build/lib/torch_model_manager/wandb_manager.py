import wandb
from typing import Union
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import helpers
import json

class WandbManager:
    project = None
    run_ids_path = None
    
    def __init__(self, 
                 project: str, 
                 run_ids_path: str, key: str=None, **kwargs):
        
        WandbManager.project = project
        WandbManager.run_ids_path = run_ids_path
        
        if not os.path.exists(WandbManager.run_ids_path):
           with open(WandbManager.run_ids_path, 'w') as json_file:
                json.dump({}, json_file)
        
        wandb.login(key=key)
        
    class Run:
        def __init__(self, run):
            pass
    # def connect_to_wandb(project: str, 
    #                     run_id_path: str,
    #                     run_name: str) -> None:
        
    #     # Additional runtime checks if needed
    #     assert isinstance(project, str), "project_name should be a string"
    #     assert isinstance(run_id_path, Union[str, None]), "run_id_path should be a string"
    #     assert isinstance(run_name, Union[str, None]), "run_name should be a string"

    #     run_id = None
    #     resume = None 
        
    #     try:
    #         run_ids = helpers.load_data_from_path(run_id_path)
            
    #         if run_ids is None:
    #             run_ids = dict()

    #         if run_name in run_ids.keys():
    #             run_id = run_ids[run_name]
    #             resume = "must"
    #     except:
    #         print(f"{run_id_path} has been created")

    #     finally:
        
    #         wandb.init(project = project, name = run_name, id = run_id, resume = resume)

    #         # If the run is created for the first time, we will associate the run id to the run name
    #         # We want that this file could not be directly accessed by the user
    #         not_exist_run_name = (os.path.exists(run_id_path)) \
    #                         and (run_name not in run_ids.keys()) \
    #                         or (not os.path.exists(run_id_path))
            
    #         if not_exist_run_name:
    #             run_ids[run_name] = wandb.run.id
    #             helpers.dump_data(run_ids, run_id_path)


    # def log(self, data):
    #     wandb.log(data)

    # def finish(self):
    #     wandb.finish()
    
# wandb_manager = WandbManager(project='test')