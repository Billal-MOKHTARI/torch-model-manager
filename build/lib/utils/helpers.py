from typing import List, Any
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import json

def required_kernel(in_size: int, out_size:int, stride=1, padding=1):
    assert in_size > 0, "Input size must be greater than 0"
    assert out_size > 0, "Output size must be greater than 0"
    assert in_size >= out_size, "Input size must be greater than or equal to output size"
    assert stride > 0, "Stride must be greater than 0"
    assert padding >= 0, "Padding must be greater than or equal to 0"
    
    return (1-out_size)*stride+in_size+2*padding

def convert_to_int(input_list):
    result = []
    for item in input_list:
        if isinstance(item, list):
            result.append(convert_to_int(item))
        elif item.isdigit():  # Check if the string represents a number
            result.append(int(item))
        else:
            result.append(item)
    return result

def create_dictionary(keys, values):
    return dict(zip(map(tuple, keys), values))

def bi_operator(op, a, b):
    if op == '==':
        
        return a == b
    elif op == '!=':
        return a != b
    elif op == '>':
        return a > b
    elif op == '>=':
        return a >= b
    elif op == '<':
        return a < b
    elif op == '<=':
        return a <= b
    elif callable(op):
        return op(a, b)
    
def intersect_dicts(dict1, dict2):
    intersection_dict = {}
    for key in dict1.keys() & dict2.keys():  # Using set intersection for keys
        if dict1[key] == dict2[key]:  # Ensure values are the same for the common key
            intersection_dict[key] = dict1[key]
    return intersection_dict

def union_dicts(dict1, dict2):
    return {**dict1, **dict2}

def is_instance_of(obj, class_names: List[type]):
    return any(isinstance(obj, class_name) for class_name in class_names)

def parse_list(l: List[Any], joiner: str = '->'):
    return joiner.join(map(str, l))

def show_images_with_indices(list_of_lists, row_indices, col_indices, **kwargs):
    """
    Visualize a list of lists containing tensors of images along with row and column indices.

    Args:
        list_of_lists (list): List of lists containing tensors of images.
        row_indices (list): List of row indices.
        col_indices (list): List of column indices.
        figure_factor (float): Factor to control the size of the figure.

    Returns:
        None
    """
    figure_factor = kwargs.get("figure_factor", 1.0)
    font_size=kwargs.get("font_size", 12)

    num_rows = len(list_of_lists)
    num_cols = max(len(sublist) for sublist in list_of_lists)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figure_factor*num_cols, figure_factor*num_rows))

    for i, sublist in enumerate(list_of_lists):
        for j, tensor in enumerate(sublist):
            # Get the axes for the current subplot
            ax = axes[i, j] if num_rows > 1 else axes[j]

            # Plot the image tensor
            ax.imshow(tensor.squeeze().numpy(), cmap='gray')

            # Set title with indices
            ax.set_title(f'({row_indices[i]}, {col_indices[j]})', fontsize=font_size)

            # Remove axis ticks
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    """
    Visualize a list of lists containing tensors of images along with row and column indices.

    Args:
        list_of_lists (list): List of lists containing tensors of images.
        row_indices (list): List of row indices.
        col_indices (list): List of column indices.

    Returns:
        None
    """
    num_rows = len(list_of_lists)
    num_cols = max(len(sublist) for sublist in list_of_lists)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows))

    for i, sublist in enumerate(list_of_lists):
        for j, tensor in enumerate(sublist):
            # Get the axes for the current subplot
            ax = axes[i, j] if num_rows > 1 else axes[j]

            # Plot the image tensor
            ax.imshow(tensor.squeeze().numpy(), cmap='gray')

            # Set title with indices
            ax.set_title(f'({row_indices[i]}, {col_indices[j]})')

            # Remove axis ticks
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def resize_and_concat_images(list_of_lists):
    """
    Resize images in a list of lists to the same size and concatenate them.

    Args:
        list_of_lists (list): List of lists containing tensors of images.

    Returns:
        torch.Tensor: Concatenated tensor of resized images.
    """
    # Determine the maximum width and height among all images
    max_width = max(max(img.shape[2] for img in sublist) for sublist in list_of_lists)
    max_height = max(max(img.shape[1] for img in sublist) for sublist in list_of_lists)


    # Resize and concatenate images
    concatenated_images = []
    for sublist in list_of_lists:
        resized_images = []
        for img in sublist:
            # Resize image to the maximum width and height
            transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((max_width, max_height)), 
            transforms.ToTensor()           
            ])

            resized_img = transform(img)
            resized_images.append(resized_img)
        # Concatenate resized images horizontally
        concatenated_images.append(torch.cat(resized_images, dim=2))
    # Concatenate sublist images vertically
    final_image = torch.cat(concatenated_images, dim=1)
    
    return final_image

def concatenate_with_character(lst, char, mode='post'):
    if mode == 'pre':
        concatenated_list = [char + str(item) for item in lst]
    elif mode == 'post':
        concatenated_list = [str(item) + char for item in lst]
    else:
        raise ValueError("Invalid mode. Use 'pre' or 'post'.")

    return concatenated_list


def add_to_json_file(file_path: str, key, value):
    """
    Add a key-value pair to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        key: The key to add.
        value: The value to add.
    """
    data = read_json_file(file_path)
    data[key] = value
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)
        
def read_json_file(file_path):
    """
    Read a JSON file and return its content.

    Args:
        file_path: The path to the JSON file.

    Returns:
        The content of the JSON file.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data
