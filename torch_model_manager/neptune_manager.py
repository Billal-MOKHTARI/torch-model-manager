import neptune
from neptune import management
from neptune.types import File
import json
from colorama import Fore
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import torch
from typing import Any, List, Union
from neptune_pytorch import NeptuneLogger
from neptune.utils import stringify_unsupported
from torch import nn
from io import StringIO
from ydata_profiling import ProfileReport
from neptune.integrations.python_logger import NeptuneHandler
import logging
from torchviz import make_dot
from torchcam.methods import LayerCAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISCAM, XGradCAM 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import helpers
import torch_model_manager as tmm
from torchcam.utils import overlay_mask

def add_to_json_file(file_path: str, key, value):
    data = read_json_file(file_path)
    data[key] = value
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)
        
def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

class NeptuneManager:
    # Define static attributes
    run_ids_path = None
    project_name = None
    api_token = None
    project = None
     
    def __init__(self, project_name, api_token, run_ids_path_var, visibility = None, namespace=None, description=None, key=None):
        NeptuneManager.project_name = project_name
        NeptuneManager.api_token = api_token
        NeptuneManager.run_ids_path = run_ids_path_var

        if NeptuneManager.project_name not in management.get_project_list(api_token=NeptuneManager.api_token):
            NeptuneManager.project = management.create_project(name=NeptuneManager.project_name, 
                                namespace=namespace, 
                                key = key,
                                visibility=visibility, 
                                description=description, 
                                api_token=api_token)
        if not os.path.exists(NeptuneManager.run_ids_path):
            # Create a JSON file
            with open(NeptuneManager.run_ids_path, 'w') as json_file:
                json.dump({}, json_file)
            

    def get_project_list(self):
        return management.get_project_list()
    
    def delete_project(self):
        management.delete_project(project=NeptuneManager.project_name)
    
    def fetch_runs_table(self):
        runs_table_df = NeptuneManager.project.fetch_runs_table().to_pandas()
        return runs_table_df
    
    class Run:
        def __init__(self, name, 
                    description = None, 
                    tags = None, 
                    source_files = None, 
                    capture_strerr = True,
                    git_ref = None,
                    **kwargs):
            self.run = None
            # Try to read the JSON file
            try:

                run_ids = read_json_file(NeptuneManager.run_ids_path)

                # If the name of the run already exists in the JSON file, load the run with the same name.
                if name in run_ids.keys(): 
                    
                    if run_ids is not None:
                        self.run = neptune.init_run(
                            project=NeptuneManager.project_name,
                            api_token=NeptuneManager.api_token,
                            with_id=run_ids[name],
                            **kwargs
                        )
                        
                else:
                    # Create a new run
                    # The names of the runs should all be different
                    assert name not in run_ids.keys(), "Run with the same name already exists. Please choose a different name."    
                    
                    self.run = neptune.init_run(
                        project=NeptuneManager.project_name,
                        api_token=NeptuneManager.api_token,
                        name = name,
                        description=description,
                        tags=tags,
                        source_files=source_files,
                        capture_stderr=capture_strerr,
                        git_ref=git_ref,
                        **kwargs
                    ) 
                    # Retrieve the id of the run and store it in the file with its associated name
                    self.run_id = self.run["sys/id"].fetch()
                    add_to_json_file(NeptuneManager.run_ids_path, name, self.run_id)
            except:
                print(Fore.RED+"The JSON file is not found. Please check the path."+Fore.WHITE)
            self.npt_logger = None
            
        # Log the tensors to Neptune
        def log_tensors(self, 
                        tensors, 
                        descriptions: List[str] = None, 
                        names: List[str] = None, 
                        paths: List[str] = None, 
                        namespace: str = None, 
                        on_series: bool = False):
            # Transform the tensor to PIL image
            to_pil = transforms.ToPILImage()
            
            # If the images are not uploaded in series, log them separately each one with its path
            if not on_series:
            
                for path, tensor in zip(paths, tensors) :
                    self.run[path].upload(File.as_image(to_pil(tensor)))
            
            # If the images are uploaded in series, log them in the same namespace
            else:
                if descriptions is not None:
                    for name, description, tensor in zip(names, descriptions, tensors):
                        self.run[namespace].append(File.as_image(to_pil(tensor)), name = name, description = description)
                else:
                    for name, tensor in zip(names, tensors):
                        self.run[namespace].append(File.as_image(to_pil(tensor)), name = name)            

            print(Fore.GREEN+"The tensors are successfully uploaded to Neptune.", Fore.WHITE)
        def log_files(self, data, namespace, from_path=None, extension=None):
            try:
                if from_path is not None:
                    self.run[namespace].upload(File.from_path(from_path, extension=extension))
                else:
                    self.run[namespace].upload(File.as_pickle(data))
                print(Fore.GREEN+"The data are successfully loaded to Neptune.")
            except:
                print(Fore.RED+"The data are not loaded to Neptune. Please check the path or the data format.\
                    This also might due to the existence of the same path in the namespace which risks to be overweighted."+Fore.WHITE)
        
        def log_dataframe(self, 
                          dataframe: pd.DataFrame, 
                          namespace: str, 
                          df_name: str,
                          df_format: bool= True,
                          csv_format: bool = False, 
                          profile_report_title: str = None,
                          profile_report_name: str = None,
                          **kwargs):
            assert df_format or csv_format, "At least one format should be chosen."
            to_csv_kwargs = kwargs.get("csv_kwargs", {})
            profile_report_kwargs = kwargs.get("profile_report_kwargs", {"dark_mode": True})
            if df_format:
                self.run[namespace][df_name].upload(File.as_html(dataframe))
            if csv_format:
                # create the temporary folder if it doesn't exist
                csv_buffer = StringIO()
                dataframe.to_csv(csv_buffer, **to_csv_kwargs)
                self.run[namespace][df_name].upload(File.from_stream(csv_buffer, extension="csv"))
                
            if profile_report_name is not None and profile_report_title is not None:
                profile = ProfileReport(dataframe, title=profile_report_title, **profile_report_kwargs)
                
                self.run[namespace][profile_report_name].upload(
                    File.from_content(profile.to_html(), extension="html")
                )
            
        
        def init_npt_logger(self, 
                            model: nn.Module,
                            base_namespace: str = 'training', 
                            log_model_diagram: bool = True, 
                            log_gradients: bool = True, 
                            log_parameters: bool = True, 
                            log_freq: int = 1,
                            **kwargs):
            self.npt_logger = NeptuneLogger(
                run=self.run,
                base_namespace=base_namespace,
                model=model,
                log_model_diagram=log_model_diagram,
                log_gradients=log_gradients,
                log_parameters=log_parameters,
                log_freq=log_freq,

            )
            
            output = kwargs.get("output", None)
            show_attrs = kwargs.get("show_attrs", True)
            show_saved = kwargs.get("show_saved", True)
            render_args = kwargs.get("render_args", {"filename": "model", "format": "png"})
            
            if output is not None:
                make_dot(output, 
                    params=dict(model.named_parameters()), 
                    show_attrs=show_attrs, show_saved=show_saved).render(**render_args)
                self.run[self.npt_logger.base_namespace]["model"][render_args["filename"]].upload(File.from_path(render_args["filename"]+"."+render_args["format"]), wait=True)
                
            print(Fore.GREEN+"The Neptune Logger is successfully initialized.", Fore.WHITE)
        
        def track_metric(self, 
                         model: nn.Module, 
                         metric: Union[float, int],
                         namespace: str, 
                         **kwargs):
            self.init_npt_logger(model, **kwargs)
            self.run[self.npt_logger.base_namespace][namespace].append(metric)

        def log_checkpoint(self, model, checkpoint_name, **kwargs):
            self.init_npt_logger(model, **kwargs)
            self.npt_logger.log_checkpoint(checkpoint_name)
            
            print(Fore.GREEN+"The checkpoint is successfully logged to Neptune.", Fore.WHITE)

        def hyperparams_logger(self, 
                               model: nn.Module, 
                               hyperparams: dict, 
                               namespace: str ="hyperparams",
                               **kwargs):  
            self.init_npt_logger(model, kwargs)
            self.run[self.npt_logger.base_namespace][namespace] = stringify_unsupported( 
                hyperparams
            )
            
            print(Fore.GREEN+"The hyperparameters are successfully logged to Neptune.", Fore.WHITE)
            
        def log_figure(self, 
                       figure, 
                       namespace: str):
            self.run[namespace].upload(File.as_html(figure))
            print(Fore.GREEN+"The figure is successfully uploaded to Neptune.", Fore.WHITE)
        
        def log_text(self, text, namespace, logger_name, level):
            assert level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "The level should be one of the following: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'"
            level = eval(f"logging.{level}")
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            npt_handler = NeptuneHandler(run=self.run)
            
            logger.addHandler(npt_handler)
            self.run[namespace].log(text)
            print(Fore.GREEN+"The text is successfully logged to Neptune.", Fore.WHITE)
        
        def log_args(self, args, namespace):
            self.run[namespace] = args
            
            print(Fore.GREEN+"The arguments are successfully logged to Neptune.", Fore.WHITE)
        
        def delete_data(self, namespaces):
            for namespace in namespaces:
                self.run.pop(namespace)

        def stop_run(self):
            self.run.stop()
                
        def log_hidden_conv2d(self, 
                                model: nn.Module,
                                input_data: torch.Tensor, 
                                indexes, 
                                method="layercam",
                                row_index: List[str] = None, 
                                save_path = None,
                                namespace = None,
                                **kwargs
                                ) :
                assert namespace is not None, "Please provide an image namespace."
                explainers = {
                    "layercam": LayerCAM,
                    "gradcam": GradCAM,
                    "smooth_gradcampp": SmoothGradCAMpp,
                    "gradcampp": GradCAMpp,
                    "scorecam": ScoreCAM,
                    "sscam": SSCAM,
                    "iscam": ISCAM,
                    "xgradcam": XGradCAM
                    
                }
                explainer = explainers[method]
                
                # Define the model mnager
                model_manager= tmm.TorchModelManager(model)
                
                # Get the needed layers
                layers = model_manager.get_layers_by_indexes(indexes)
                
                result = []
                for im in input_data:
                    # Set the model to evaluation mode
                    tmp_model = model.eval()

                    # Extract the CAMs
                    layer_extractor = explainer(tmp_model, layers)
                    out = tmp_model(im.unsqueeze(0))
                    cams = layer_extractor(out.squeeze(0).argmax().item(), out)
                    
                    
                    if method != "layercam":
                        alpha = kwargs.get("alpha", 0.5)
                        to_pil_image = transforms.ToPILImage()
                        
                        row_imgs = []
                        # Resize the CAM and overlay it
                        for cam in cams:
                            cams = overlay_mask(to_pil_image(im), to_pil_image(cam.squeeze(0)), alpha=alpha)
                            cams = transforms.ToTensor()(cams)
                            row_imgs.append(cams)
                            cams = row_imgs
                        # Display it
                        
                    
                    result.append(cams)


                if row_index is None:
                    row_index = np.arange(input_data.shape[0])
                
                # Parse the indexes
                col_index = [helpers.parse_list(index, joiner='->') for index in indexes]

                # Concatenate the images
                figure = helpers.resize_and_concat_images(result)
                
                # Save the figure to disk if save_path is provided
                if save_path is not None:
                    to_pil = transforms.ToPILImage()
                    pil_image = to_pil(figure)

                    # Save the PIL image to disk
                    pil_image.save(save_path)
                
                # Log the figure to Neptune
                for res_row, row_ind in zip(result, row_index):
                    
                    self.log_tensors(tensors=res_row, 
                                    names = helpers.concatenate_with_character(col_index, f"{row_ind} -> ", mode='pre'), 
                                    namespace=namespace,
                                    on_series=True)
                        
                print(Fore.GREEN+"The hidden conv2d layer outputs are successfully logged to Neptune.", Fore.WHITE)
                return result, row_index, col_index
            
# nm = NeptuneManager(project_name="Billal-MOKHTARI/Image-Clustering-based-on-Dual-Message-Passing",
#                     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NGRlOTNiZC0zNGZlLTRjNWUtYWEyMC00NzEwOWJkOTRhODgifQ==",
#                     run_ids_path_var="run_ids.json")

# from torchvision import models
# parameters = {
#     "lr": 1e-2,
#     "bs": 128,
#     "input_sz": 32 * 32 * 3,
#     "n_classes": 10,
#     "model_filename": "basemodel",
#     "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     "epochs": 2,
# }


# run = nm.Run(name="Test2")

# model= models.vgg16()
# run.log_hidden_conv2d(model, torch.randn(1, 3, 224, 224), indexes=[["features", 0], ["features", 1]], namespace="hidden_conv2d", method="scam")