
from torch import nn
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import helpers
from typing import List
from torchviz import make_dot
import torch
from torchvision import transforms
from torchcam.methods import LayerCAM
from torchvision import models
from torch.nn.init import xavier_uniform_, kaiming_uniform_, constant_, dirac_, kaiming_normal_, \
                            xavier_normal_, uniform_, eye_, normal_, sparse_, ones_, orthogonal_, \
                            zeros_, trunc_normal_
                                

class TorchModelManager:
    """
    A class for managing PyTorch models.

    Attributes:
        model: The PyTorch model to manage.
        named_layers: A dictionary containing the named layers of the model.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize PyModelManager with a given PyTorch model.

        Args:
            model (nn.Module): The PyTorch model to manage.
        """
        self.model = model
        self.named_layers = self.get_named_layers()
        self.model_depth = self.get_model_depth()

    def get_named_layers(self) -> dict:
        """
        Recursively fetch and store all named layers of the model.

        Returns:
            dict: A dictionary containing the named layers of the model.
        """
        def get_layers_recursive(model_children: dict) -> dict:
            """
            Recursively retrieves all layers in a model.

            Args:
                model_children (dict): A dictionary containing the children layers of a model.

            Returns:
                dict: A dictionary containing all the named layers in the model.
            """
            for name, layer in model_children.items():
                if dict(layer.named_children()) != {}:
                    model_children[name] = get_layers_recursive(dict(layer.named_children()))
                else:
                    model_children[name] = layer
            self.named_layers = model_children
            return self.named_layers
        
        return get_layers_recursive(dict(self.model.named_children()))

    def get_model_depth(self):

        def dict_depth(dictionary):
            if not isinstance(dictionary, dict) or not dictionary:
                return 0
            return 1 + max(dict_depth(value) for value in dictionary.values())
        
        return dict_depth(self.named_layers)

    def get_attribute(self, layer: nn.Module, attribute: str) -> any:
        """
        Get a specific attribute of a layer.

        Args:
            layer (nn.Module): The layer.
            attribute (str): The attribute name.

        Returns:
            Any: The value of the attribute.
        """
        assert hasattr(layer, attribute), f'{layer} does not have the attribute {attribute}'
        return getattr(layer, attribute)
    
    def get_layer_by_index(self, index: list) -> nn.Module:
        """
        Get a layer from the model using its index.

        Args:
            index (list): The index path of the layer in the model.

        Returns:
            nn.Module: The layer from the model.
        """
        trace = ['self.model']
        try:
            for ind in index[:-1]:
                
                if isinstance(ind, int):
                    trace.append(f'[{ind}]')
                else:
                    trace.append(f'.{ind}')
            
            if isinstance(index[-1], int):
                layers = eval(''.join(trace))
                return layers[index[-1]]
            else:
                return getattr(eval(''.join(trace)), index[-1])
        except:
            print(f'Layer with index {index} not found')

    def get_layer_by_attribute(self, property: str, value: any, operator: str) -> dict:
        """
        Retrieves layers from the model that have a specific attribute value based on the given property and operator.

        Args:
            property (str): The name of the attribute to check in each layer.
            value (any): The value to compare against the attribute.
            operator (str): The operator to use for comparison. Supported operators are '==', '!=', '>', '<', '>=', and '<='.

        Returns:
            dict: A dictionary mapping the indexes of the layers to the corresponding layer objects.

        Example:
            >>> model_manager = ModelManager()
            >>> layers = model_manager.get_layer_by_attribute('activation', 'relu', '==')
            >>> print(layers)
            {0: <torch.nn.ReLU object at 0x7f9a2e3a8a90>, 2: <torch.nn.ReLU object at 0x7f9a2e3a8b50>}
        """
        def dfs(self, model, property, value, operator, layers, indexes, tmp=None, depth=0):
            if tmp is None:
                tmp = []

            for name, layer in model.named_children():
                tmp.append(name)
                if hasattr(layer, property) and helpers.bi_operator(operator, self.get_attribute(layer, property), value):
                    layers.append(layer)
                    indexes.append(tmp.copy())

                # Recursive call
                dfs(self, layer, property, value, operator, layers, indexes, tmp, depth+1)

                # Pop the last element to backtrack
                tmp.pop()

        layers = []
        indexes = []

        dfs(self, self.model, property, value, operator, layers, indexes)

        return helpers.create_dictionary(helpers.convert_to_int(indexes), layers)
    

    def get_layer_by_attributes(self, conditions: dict) -> dict:
        """
        Retrieves layers that satisfy the given conditions.

        Args:
            conditions (dict): A dictionary containing the conditions for layer retrieval.
                The dictionary should have the following structure:
                {
                    'and': [
                        {'property': ['attribute_name', 'attribute_value'], 'operator': 'comparison_operator'},
                        ...
                    ],
                    'or': [
                        {'property': ['attribute_name', 'attribute_value'], 'operator': 'comparison_operator'},
                        ...
                    ]
                }

                - 'and' key represents the conditions that must be satisfied simultaneously.
                - 'or' key represents the conditions where at least one of them must be satisfied.

                Each condition is a dictionary with the following keys:
                - 'property': A list containing the attribute name and attribute value to be compared.
                - 'operator': The comparison operator to be used for the attribute comparison.

        Returns:
            dict: A dictionary containing the layers that satisfy the given conditions.

        Example:
            >>> conditions = {
            >>>     'and': [
            >>>         {'property': ['name', 'layer1'], 'operator': '=='},
            >>>         {'property': ['type', 'convolutional'], 'operator': '=='}
            >>>     ],
            >>>     'or': [
            >>>         {'property': ['name', 'layer2'], 'operator': '=='},
            >>>         {'property': ['type', 'pooling'], 'operator': '=='}
            >>>     ]
            >>> }
            >>> result = get_layer_by_attributes(conditions)
            >>> print(result)
            {'layer1': {'name': 'layer1', 'type': 'convolutional'}, 'layer2': {'name': 'layer2', 'type': 'pooling'}}
        """
        result = dict()
        first_iter = True
        for operator, operands in conditions.items():
            tmp_statement_result = dict()
            if operator == 'and':
                for stat in operands:
                    prop = list(stat.values())[0][0]
                    value = list(stat.values())[0][1]
                    op = list(stat.keys())[0]
                    search_result = self.get_layer_by_attribute(prop, value, op)
                    if first_iter:
                        tmp_statement_result = search_result
                        first_iter = False
                    else:
                        tmp_statement_result = helpers.intersect_dicts(tmp_statement_result, search_result)
            elif operator == 'or':
                for stat in operands:
                    prop = list(stat.values())[0][0]
                    value = list(stat.values())[0][1]
                    op = list(stat.keys())[0]
                    search_result = self.get_layer_by_attribute(prop, value, op)
                    if first_iter:
                        tmp_statement_result = search_result
                        first_iter = False
                    else:
                        tmp_statement_result = helpers.union_dicts(tmp_statement_result, search_result)
            result = helpers.union_dicts(result, tmp_statement_result)
        return result
                    
    def get_layer_by_instance(self, instance_type: List[type]) -> dict:
        """
        Search for layers in the model by their instance type.

        Args:
            instance_type (type): The instance type of the layers to search for.

        Returns:
            dict: A dictionary mapping the indexes of found layers to the layers themselves.
        """
        def dfs(self, model, instance_type, layers, indexes, tmp=None, depth=0):
            if tmp is None:
                tmp = []

            for name, layer in model.named_children():
                tmp.append(name)
                if helpers.is_instance_of(layer, instance_type):
                    layers.append(layer)
                    indexes.append(tmp.copy())

                # Recursive call
                dfs(self, layer, instance_type, layers, indexes, tmp, depth + 1)

                # Pop the last element to backtrack
                tmp.pop()

        layers = []
        indexes = []

        dfs(self, self.model, instance_type, layers, indexes)

        return helpers.create_dictionary(helpers.convert_to_int(indexes), layers)    

    def get_layer_by_depth(self, depth: int) -> dict:
        """
        Retrieves layers from the model that are at a specific depth.

        Args:
            depth (int): The depth at which the layers are located.

        Returns:
            dict: A dictionary mapping the indexes of the layers to the corresponding layer objects.
        """
        def dfs(self, model, depth, layers, indexes, tmp=None, current_depth=0):
            if tmp is None:
                tmp = []

            for name, layer in model.named_children():
                tmp.append(name)
                if current_depth == depth:
                    layers.append(layer)
                    indexes.append(tmp.copy())
                else:
                    dfs(self, layer, depth, layers, indexes, tmp, current_depth + 1)

                # Pop the last element to backtrack
                tmp.pop()

        layers = []
        indexes = []

        dfs(self, self.model, depth, layers, indexes)

        return helpers.create_dictionary(helpers.convert_to_int(indexes), layers)

    def get_indexes(self, indexes):
        def dfs(self, model, indexes, tmp=None, depth=0):
            if tmp is None:
                tmp = []

            for name, _ in model.named_children():
                tmp.append(name)
                indexes.append(tmp.copy())

                # Recursive call
                dfs(self, indexes, tmp, depth+1)

                # Pop the last element to backtrack
                tmp.pop()

        indexes = []
        dfs(self, self.model, indexes)
        
        return indexes
    
    
    def delete_layer_by_index(self, index: list) -> None:
        """
        Delete a layer from the model using its index.

        Args:
            index (list): The index path of the layer in the model.
        """
        trace = ['self.model']
        for ind in index[:-1]:
            
            if isinstance(ind, int):
                trace.append(f'[{ind}]')
            else:
                trace.append(f'.{ind}')
        
        if isinstance(index[-1], int):
            layers = eval(''.join(trace))
            del layers[index[-1]]
        else:
            delattr(eval(''.join(trace)), index[-1])

    def delete_layer_by_attribute(self, property: str, value: any, operator: str) -> None:
        """
        Deletes a layer from the model based on the specified attribute.

        Args:
            property (str): The attribute to search for.
            value: The value to compare against.
            operator (str): The operator to use for comparison.

        Returns:
            None

        Raises:
            None
        """
        search_res = self.get_layer_by_attribute(property, value, operator)

        while list(search_res.keys()) != []:
            self.delete_layer_by_index(list(search_res.keys())[0])
            search_res = self.get_layer_by_attribute(property, value, operator)

    def delete_layer_by_attributes(self, conditions: dict) -> None:
        """
        Deletes layers that match the given attributes.

        Parameters:
        - conditions (dict): A dictionary of attribute-value pairs to match the layers.

        Returns:
        None
        """
        search_res = self.get_layer_by_attributes(conditions)

        while list(search_res.keys()) != []:
            self.delete_layer_by_index(list(search_res.keys())[0])
            search_res = self.get_layer_by_attributes(conditions)


    def delete_layer_by_instance(self, instance_types: List[type]) -> None:
        """
        Delete layers from the model by their instance type.

        Args:
            instance_type (type): The instance type of the layers to delete.
        """
        search_res = self.get_layer_by_instance(instance_types)
 
        while list(search_res.keys()) != []:
            self.delete_layer_by_index(list(search_res.keys())[0])
            search_res = self.get_layer_by_instance(instance_types)

    def get_layers_by_indexes(self, indexes: List[list]) -> List:
        """
        Get layers from the model using their indexes.

        Args:
            indexes (List[list]): A list of indexes of the layers in the model.

        Returns:
            dict: A dictionary containing the layers from the model.
        """
        layers = []
        for index in indexes:
            layers.append(self.get_layer_by_index(index))
        return layers

    def delete_layers_by_indexes(self, indexes: List[list]) -> None:
        """
        Delete layers from the model using their indexes.

        Args:
            indexes (List[list]): A list of indexes of the layers in the model.
        """
        for index in indexes:
            self.delete_layer_by_index(index)

    def visualize(self, shape, show_attrs: bool = True, show_saved: bool = True, **kwargs) -> None:
        x = torch.randn(shape)
        make_dot(self.model(x), 
                 params=dict(self.model.named_parameters()), 
                 show_attrs=show_attrs, show_saved=show_saved).render(**kwargs)
        


    def show_hidden_conv2d(self, 
                           input_data: torch.Tensor, 
                           indexes, 
                           row_index: List[str] = None, 
                           show_figure: bool = True, 
                           figure_factor: float = 1.0,
                           save_path = None,
                           neptune_manager = None,
                           run = None,
                           image_workspace = None,
                           ) :
        result = []
        layers = self.get_layers_by_indexes(indexes)

        for im in input_data:
            tmp_model = self.model.eval()

            layer_extractor = LayerCAM(tmp_model, layers)
            out = tmp_model(im.unsqueeze(0))
 
            cams = layer_extractor(out.squeeze(0).argmax().item(), out)
            
            result.append(cams)

        if row_index is None:
            row_index = np.arange(input_data.shape[0])
        
        col_index = [helpers.parse_list(index, joiner='->') for index in indexes]


        if show_figure:
            helpers.show_images_with_indices(result, row_index, col_index, figure_factor=figure_factor)



        figure = helpers.resize_and_concat_images(result)
        if save_path is not None:
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(figure)

            # Save the PIL image to disk
            pil_image.save(save_path)

        if run is not None:
            assert image_workspace is not None, "Please provide a workspace name for the images"
            for res_row, row_ind in zip(result, row_index):
                neptune_manager.log_tensors(run, 
                                            tensors=res_row, 
                                            names = helpers.concatenate_with_character(col_index, f"{row_ind} -> ", mode='pre'), 
                                            workspace=image_workspace,
                                            on_series=True)
                

        return result, row_index, col_index
    
    def init_model_parameters(self, method_weight, method_bias, **kwargs):
        weight_args = kwargs.get('weight_args', None)
        bias_args = kwargs.get('bias_args', None)
        
        for layer in self.model.children():
            if isinstance(layer, (nn.Conv2d, 
                        nn.Linear, 
                        nn.ConvTranspose2d, 
                        nn.Conv3d, 
                        nn.ConvTranspose3d,
                        nn.Embedding)):
                
                weight_initializer = eval(method_weight)
                bias_initializer = eval(method_bias)
                
                weight_initializer(layer.weight, weight_args)
                if layer.bias is not None:
                    bias_initializer(layer.bias, bias_args)
                    
                
                            
                
                
                        
# model = nn.Embedding(12, 13)
# for layer in model.modules():
#     if isinstance(layer, (nn.Conv2d, nn.Linear, nn.Embedding)):
#         layer.weight.data.fill_(1)

        
# for layer in model.modules():
#     if isinstance(layer, (nn.Conv2d, nn.Linear, nn.Embedding)):
#         print(layer.weight.data)
  