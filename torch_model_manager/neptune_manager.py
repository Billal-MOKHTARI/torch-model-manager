import neptune
from neptune import management
from neptune.types import File
import json

from torchvision import transforms
import pickle as pkl

class NeptuneManager:
    def __init__(self, project, api_token, run_ids_path, visibility = None, workspace=None, description=None, key=None):
        self.project = project
        self.api_token = api_token
        self.run_ids_path = run_ids_path

        if project not in management.get_project_list(api_token=self.api_token):
            management.create_project(name=project, 
                                workspace=workspace, 
                                key = key,
                                visibility=visibility, 
                                description=description, 
                                api_token=api_token)
            # Create a JSON file
            with open(self.run_ids_path, 'w') as json_file:
                json.dump({}, json_file)
            
    def add_to_json_file(self, file_path, key, value):
        data = self.read_json_file(file_path)
        data[key] = value
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
            
    def read_json_file(self, file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data
        
    def get_project_list(self):
        return management.get_project_list()
    
    def delete_project(self):
        management.delete_project(project=self.project)
        
    def create_run(self, 
                name, 
                description = None, 
                tags = None, 
                source_files = None, 
                dependencies = None,
                capture_strerr = True,
                git_ref = None,
                **kwargs):

        run_ids = self.read_json_file(self.run_ids_path)
        if name in run_ids: 
            if run_ids is not None:
                run = neptune.init_run(
                    project=self.project,
                    api_token=self.api_token,
                    with_id=run_ids[name],
                    **kwargs
                )
        else:
            assert name not in run_ids.keys(), "Run with the same name already exists. Please choose a different name."
            
            run = neptune.init_run(
                project=self.project,
                api_token=self.api_token,
                name = name,
                description=description,
                tags=tags,
                source_files=source_files,
                dependencies=dependencies,
                capture_stderr=capture_strerr,
                git_ref=git_ref,
                **kwargs
            ) 
            run_id = run["sys/id"].fetch()
            self.add_to_json_file(self.run_ids_path, name, run_id)
            
        return run
    
    def log_files_by_paths(self, run, neptune_paths, file_paths):
        for neptune_path, file_path in zip(neptune_paths, file_paths):
            run[neptune_path].upload(file_path)
    
    def log_tensors(self, run, tensors, descriptions = None, names = None, paths = None, workspace = None, on_series = False):
        to_pil = transforms.ToPILImage()
        if not on_series:
            for path, tensor in zip(paths, tensors) :
                run[path].upload(File.as_image(to_pil(tensor)))
        else:
            if descriptions is not None:
                for name, description, tensor in zip(names, descriptions, tensors):
                    run[workspace].append(File.as_image(to_pil(tensor)), name = name, description = description)
            else:
                for name, tensor in zip(names, tensors):
                    run[workspace].append(File.as_image(to_pil(tensor)), name = name)

    def delete_data(self, run, workspaces):
        for workspace in workspaces:
            run.pop(workspace)
            
    def log_artifacts(self, run, data, path, workspace):
        with open(path, 'wb') as f:
            pkl.dump(data, f)
        run[workspace].track_files(path)

        