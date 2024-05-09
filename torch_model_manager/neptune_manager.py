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
import pickle
import tempfile


class NeptuneManager:
    """
    A class for managing Neptune projects and runs.
    """
    # Define static attributes
    run_ids_path = None
    project_name = None
    api_token = None
    project = None
     
    def __init__(self, 
                project_name: str, 
                api_token: str, 
                run_ids_path: str, 
                visibility: str = None, 
                namespace: str = None, 
                description: str = None, 
                key = None):
        """
        Initialize the NeptuneManager class.

        Args:
            project_name (str): The name of the project.
            api_token (str): The API token for Neptune.
            run_ids_path (str): The path to the JSON file for storing run IDs.
            visibility (str, optional): The visibility of the project. Defaults to None.
            namespace (str, optional): The namespace of the project. Defaults to None.
            description (str, optional): The description of the project. Defaults to None.
            key (str, optional): The key of the project. Defaults to None.
        """
        NeptuneManager.project_name = project_name
        NeptuneManager.api_token = api_token
        NeptuneManager.run_ids_path = run_ids_path

        if NeptuneManager.project_name not in management.get_project_list(api_token=NeptuneManager.api_token):
            NeptuneManager.project = management.create_project(name=NeptuneManager.project_name, 
                                namespace=namespace, 
                                key=key,
                                visibility=visibility, 
                                description=description, 
                                api_token=api_token)
        else:
            NeptuneManager.project = neptune.init_project(project=NeptuneManager.project_name, api_token=NeptuneManager.api_token)
        if not os.path.exists(NeptuneManager.run_ids_path):
            # Create a JSON file
            with open(NeptuneManager.run_ids_path, 'w') as json_file:
                json.dump({}, json_file)
            

    def get_project_list(self):
        """
        Get the list of projects.

        Returns:
            The list of projects.
        """
        return management.get_project_list()
    
    def delete_project(self):
        """
        Delete the project.
        """
        management.delete_project(project=NeptuneManager.project_name)
    
    def fetch_runs_table(self):
        """
        Fetch the runs table.

        Returns:
            The runs table as a pandas DataFrame.
        """
        runs_table_df = NeptuneManager.project.fetch_runs_table().to_pandas()
        return runs_table_df
    
    def log_files(self, data, namespace, from_path=None, extension=None, wait = False):
 
        try:
            if from_path is not None:
                NeptuneManager.project[namespace].upload(File.from_path(from_path, extension=extension), wait=wait)
            else:
                print(NeptuneManager.project)
                NeptuneManager.project[namespace].upload(File.as_pickle(data), wait=wait)
            print(Fore.GREEN+"The data are successfully loaded to the project."+Fore.WHITE)
        except:
            print(Fore.RED+"The data are not loaded to the project. Please check the path or the data format.\
                This also might due to the existence of the same path in the namespace which risks to be overweighted."+Fore.WHITE)  
        
    def delete_data(self, namespaces, wait = False):
        """
        Delete data from the run.

        Args:
            namespaces: The namespaces to delete.
        """
        for namespace in namespaces:
            NeptuneManager.project.pop(namespace, wait = wait)

    def fetch_files(self, namespace):
        ns = namespace.split("/")
        
        struct = NeptuneManager.project.get_structure()
        for elem in ns :
            struct = struct[elem]
        
        return list(struct.keys())

    def log_tensors(self, 
                    tensors, 
                    descriptions: List[str] = None, 
                    names: List[str] = None, 
                    paths: List[str] = None, 
                    namespace: str = None, 
                    on_series: bool = False):
        """
        Log tensors to Neptune.

        Args:
            tensors: The tensors to log.
            descriptions (List[str], optional): The descriptions of the tensors. Defaults to None.
            names (List[str], optional): The names of the tensors. Defaults to None.
            paths (List[str], optional): The paths of the tensors. Defaults to None.
            namespace (str, optional): The namespace to log the tensors. Defaults to None.
            on_series (bool, optional): Whether to log the tensors in series. Defaults to False.
        """
        # Transform the tensor to PIL image
        to_pil = transforms.ToPILImage()
        
        # If the images are not uploaded in series, log them separately each one with its path
        if not on_series:
            for path, tensor in zip(paths, tensors) :
                NeptuneManager.project[path].upload(File.as_image(to_pil(tensor)))
        
        # If the images are uploaded in series, log them in the same namespace
        else:
            if descriptions is not None:
                for name, description, tensor in zip(names, descriptions, tensors):
                    NeptuneManager.project[namespace].append(File.as_image(to_pil(tensor)), name=name, description=description)
            else:
                for name, tensor in zip(names, tensors):
                    NeptuneManager.project[namespace].append(File.as_image(to_pil(tensor)), name=name)            

        print(Fore.GREEN+"The tensors are successfully uploaded to Neptune.", Fore.WHITE)

    def log_hidden_conv2d(self, 
                          model: nn.Module,
                          input_data: torch.Tensor, 
                          indexes, 
                          method="layercam",
                          row_index: List[str] = None, 
                          save_path=None,
                          namespace=None,
                          **kwargs) :
        """
        Log hidden conv2d layer outputs.

        Args:
            model (nn.Module): The model to log the hidden conv2d layer outputs for.
            input_data (torch.Tensor): The input data.
            indexes: The indexes of the layers.
            method (str, optional): The method to use for generating the CAMs. Defaults to "layercam".
            row_index (List[str], optional): The row indexes. Defaults to None.
            save_path (str, optional): The path to save the figure. Defaults to None.
            namespace (str, optional): The namespace to log the hidden conv2d layer outputs. Defaults to None.
        """
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
    
    def fetch_pkl_data(self, namespace: str):        
        tmp_file = tempfile.NamedTemporaryFile(delete=True)
        try :
            NeptuneManager.project[namespace].download(tmp_file.name)
            with open(tmp_file.name, 'rb') as f:
                data = pickle.load(f)
        
            print(Fore.GREEN+"The data is successfully fetched from Neptune.", Fore.WHITE)
            return data
            
        except:
            print(Fore.RED+"The data is not fetched from Neptune. Please check the namespace."+Fore.WHITE)

    class Run:
        """
        A class representing a Neptune run.
        """
        def __init__(self, 
                    name: str, 
                    description: str = None, 
                    tags: List[str] = None, 
                    source_files = None, 
                    capture_strerr: bool = True,
                    git_ref=None,
                    **kwargs):
            """
            Initialize the Run class.

            Args:
                name (str): The name of the run.
                description (str, optional): The description of the run. Defaults to None.
                tags (list, optional): The tags of the run. Defaults to None.
                source_files (list, optional): The source files of the run. Defaults to None.
                capture_strerr (bool, optional): Whether to capture stderr. Defaults to True.
                git_ref (str, optional): The git reference of the run. Defaults to None.
            """
            self.run = None
            self.name = name
            # Try to read the JSON file
            try:
                run_ids = helpers.read_json_file(NeptuneManager.run_ids_path)
                # If the name of the run already exists in the JSON file, load the run with the same name.
                if self.name in list(run_ids.keys()): 
                    if run_ids is not None:
                        self.run = neptune.init_run(
                            project=NeptuneManager.project_name,
                            api_token=NeptuneManager.api_token,
                            with_id=run_ids[self.name],
                            **kwargs
                        )
                        
                else:
                    # Create a new run
                    # The names of the runs should all be different
                    assert self.name not in list(run_ids.keys()), "Run with the same name already exists. Please choose a different name."    
                    
                    self.run = neptune.init_run(
                        project=NeptuneManager.project_name,
                        api_token=NeptuneManager.api_token,
                        name=self.name,
                        description=description,
                        tags=tags,
                        source_files=source_files,
                        capture_stderr=capture_strerr,
                        git_ref=git_ref,
                        **kwargs
                    ) 
                    # Retrieve the id of the run and store it in the file with its associated name
                    self.run_id = self.run["sys/id"].fetch()
                    helpers.add_to_json_file(NeptuneManager.run_ids_path, self.name, self.run_id)
            except:
                print(Fore.RED+"The JSON file is not found. Please check the path."+Fore.WHITE)
            
        def log_tensors(self, 
                        tensors, 
                        descriptions: List[str] = None, 
                        names: List[str] = None, 
                        paths: List[str] = None, 
                        namespace: str = None, 
                        on_series: bool = False):
            """
            Log tensors to Neptune.

            Args:
                tensors: The tensors to log.
                descriptions (List[str], optional): The descriptions of the tensors. Defaults to None.
                names (List[str], optional): The names of the tensors. Defaults to None.
                paths (List[str], optional): The paths of the tensors. Defaults to None.
                namespace (str, optional): The namespace to log the tensors. Defaults to None.
                on_series (bool, optional): Whether to log the tensors in series. Defaults to False.
            """
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
                        self.run[namespace].append(File.as_image(to_pil(tensor)), name=name, description=description)
                else:
                    for name, tensor in zip(names, tensors):
                        self.run[namespace].append(File.as_image(to_pil(tensor)), name=name)            

            print(Fore.GREEN+"The tensors are successfully uploaded to Neptune.", Fore.WHITE)
        
        def log_files(self, data, namespace, from_path=None, extension=None, wait = False):
            """
            Log files to Neptune.

            Args:
                data: The data to log.
                namespace (str): The namespace to log the data.
                from_path (str, optional): The path of the file to log. Defaults to None.
                extension (str, optional): The extension of the file. Defaults to None.
            """
            try:
                if from_path is not None:
                    self.run[namespace].upload(File.from_path(from_path, extension=extension), wait=wait)
                else:
                    self.run[namespace].upload(File.as_pickle(data), wait=wait)
                print(Fore.GREEN+"The data are successfully loaded to Neptune."+Fore.WHITE)
            except:
                print(Fore.RED+"The data are not loaded to Neptune. Please check the path or the data format.\
                    This also might due to the existence of the same path in the namespace which risks to be overweighted."+Fore.WHITE)
        
        def log_dataframe(self, 
                          dataframe: pd.DataFrame, 
                          namespace: str, 
                          df_name: str,
                          df_format: bool=True,
                          csv_format: bool=False, 
                          profile_report_title: str=None,
                          profile_report_name: str=None,
                          **kwargs):
            """
            Log a dataframe to Neptune.

            Args:
                dataframe (pd.DataFrame): The dataframe to log.
                namespace (str): The namespace to log the dataframe.
                df_name (str): The name of the dataframe.
                df_format (bool, optional): Whether to log the dataframe in HTML format. Defaults to True.
                csv_format (bool, optional): Whether to log the dataframe in CSV format. Defaults to False.
                profile_report_title (str, optional): The title of the profile report. Defaults to None.
                profile_report_name (str, optional): The name of the profile report. Defaults to None.
            """
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
            
        
        def track_metric(self, 
                         metric: Union[float, int],
                         namespace: str,
                         step = None,
                         timestamp = None, 
                         wait = False):
            """
            Track a metric.

            Args:
                model (nn.Module): The model to track the metric for.
                metric (Union[float, int]): The metric value.
                namespace (str): The namespace to log the metric.
            """
            self.run[namespace].append(metric, step=step, timestamp=timestamp, wait=wait)

        def log_checkpoint(self, namespace, model, optimizer, loss, epoch, keep=3, wait=False, **kwargs):
            parent_namespace = "/".join(namespace.split("/")[:-1])


            state_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "epoch": epoch,
                **kwargs
            }
            
            tmp_file = tempfile.NamedTemporaryFile(suffix = '.pth', delete=True)
            torch.save(state_dict, tmp_file.name)
                
            self.log_files(namespace=namespace, data= None, from_path=tmp_file.name, extension='pth', wait=wait)
            
            print(Fore.GREEN+"The checkpoint is successfully logged to Neptune.", Fore.WHITE)


            struct = helpers.sort_string_list(self.fetch_files(parent_namespace))

            if len(struct) > keep:
                for file in struct[:-keep]:
                    self.run.pop(os.path.join(parent_namespace, file), wait=wait)

        def log_hyperparameters(self, 
                               hyperparams: dict, 
                               namespace: str):  
            """
            Log hyperparameters.

            Args:
                model (nn.Module): The model to log the hyperparameters for.
                hyperparams (dict): The hyperparameters.
                namespace (str, optional): The namespace to log the hyperparameters. Defaults to "hyperparams".
            """
            self.run[namespace] = stringify_unsupported( 
                hyperparams
            )
            
            print(Fore.GREEN+"The hyperparameters are successfully logged to Neptune.", Fore.WHITE)
            
        def log_figure(self, 
                       figure, 
                       namespace: str):
            """
            Log a figure.

            Args:
                figure: The figure to log.
                namespace (str): The namespace to log the figure.
            """
            self.run[namespace].upload(File.as_html(figure))
            print(Fore.GREEN+"The figure is successfully uploaded to Neptune.", Fore.WHITE)
        
        def log_text(self, text, namespace, logger_name, level):
            """
            Log text.

            Args:
                text: The text to log.
                namespace (str): The namespace to log the text.
                logger_name (str): The name of the logger.
                level (str): The level of the log.
            """
            assert level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "The level should be one of the following: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'"
            level = eval(f"logging.{level}")
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            npt_handler = NeptuneHandler(run=self.run)
            
            logger.addHandler(npt_handler)
            self.run[namespace].log(text)
            print(Fore.GREEN+"The text is successfully logged to Neptune.", Fore.WHITE)
        
        def log_args(self, args, namespace):
            """
            Log arguments.

            Args:
                args: The arguments to log.
                namespace (str): The namespace to log the arguments.
            """
            self.run[namespace] = args
            
            print(Fore.GREEN+"The arguments are successfully logged to Neptune.", Fore.WHITE)
        
        def delete_data(self, namespaces, wait = False):
            """
            Delete data from the run.

            Args:
                namespaces: The namespaces to delete.
            """
            for namespace in namespaces:
                self.run.pop(namespace, wait = wait)

        def stop_run(self):
            """
            Stop the run.
            """
            self.run.stop()
                
        def log_hidden_conv2d(self, 
                                model: nn.Module,
                                input_data: torch.Tensor, 
                                indexes, 
                                method="layercam",
                                row_index: List[str] = None, 
                                save_path=None,
                                namespace=None,
                                **kwargs
                                ) :
            """
            Log hidden conv2d layer outputs.

            Args:
                model (nn.Module): The model to log the hidden conv2d layer outputs for.
                input_data (torch.Tensor): The input data.
                indexes: The indexes of the layers.
                method (str, optional): The method to use for generating the CAMs. Defaults to "layercam".
                row_index (List[str], optional): The row indexes. Defaults to None.
                save_path (str, optional): The path to save the figure. Defaults to None.
                namespace (str, optional): The namespace to log the hidden conv2d layer outputs. Defaults to None.
            """
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
        
        def fetch_pkl_data(self, namespace: str):
            """
            Fetches pickle data from Neptune.

            Args:
                namespace (str): The namespace where the data is stored.

            Returns:
                data: The fetched data.
            """
            tmp_file = tempfile.NamedTemporaryFile(delete=True)
            try :
                self.run[namespace].download(tmp_file.name)
                with open(tmp_file.name, 'rb') as f:
                    data = pickle.load(f)
            
                print(Fore.GREEN+"The data is successfully fetched from Neptune.", Fore.WHITE)
                return data
                
            except:
                print(Fore.RED+"The data is not fetched from Neptune. Please check the namespace."+Fore.WHITE)

        def load_model_checkpoint(self, namespace, **kwargs):
            """
            Loads a model checkpoint from Neptune.

            Args:
                namespace (str): The namespace where the checkpoint is stored.
                **kwargs: Additional keyword arguments for torch.load().

            Returns:
                state_dict: The loaded model state dictionary.
            """
            tmp_file = tempfile.NamedTemporaryFile(delete=True)
            try:
                self.run[namespace].download(tmp_file.name)
                state_dict = torch.load(tmp_file.name, **kwargs)
                
                return state_dict
            except:
                print(Fore.RED+"The checkpoint is not fetched from Neptune. Please check the namespace."+Fore.WHITE)

        def fetch_files(self, namespace):
            """
            Fetches files from a specified namespace in Neptune.

            Args:
                namespace (str): The namespace containing the files.

            Returns:
                list: A list of filenames in the specified namespace.
            """
            ns = namespace.split("/")
            
            struct = self.run.get_structure()
            for elem in ns :
                struct = struct[elem]
            
            return list(struct.keys())
        
        def fetch_data(self, namespace):
            """
            Fetches data from a specified namespace in Neptune.

            Args:
                namespace (str): The namespace containing the data.

            Returns:
                list: A list of fetched data values.
            """
            return self.run[namespace].fetch_values()