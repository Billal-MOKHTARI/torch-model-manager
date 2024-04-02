# Torch Model Manager


**Torch Model Manager** is an open-source python project designed for Deep Learning developpers that aims to make the use of pytorch library easy. The version ![version](https://img.shields.io/badge/version-1.0.0.dev1-gray?labelColor=blue&style=flat) is still under developpment. The package allows us to access, search and delete layers by index, attributes or instance.

### Examples of Use
1. **Initialization**
```python
from torchvision import

# Assume you have a PyTorch model 'model'
model = models.vgg16(pretrained=True)

model_manager = TorchModelManager(model)
```

2. **Get Named Layers**
```python
named_layers = model_manager.get_named_layers()
```

3. **Get Layer by Index**
```python
layer_index = ['classifier', 6]
layer = model_manager.get_layer_by_index(layer_index)
```

4. **Get Layer by Attribute**
```python
layers = model_manager.get_layer_by_attribute('activation', 'relu', '==')
```

5. **Get Layers by Conditions**
```python
# Retrieve layers that satisfy the given conditions
conditions = {
            'and': [{'==': ('kernel_size', (1, 1))}, {'==': ('stride', (1, 1))}],
            'or': [{'==': ('kernel_size', (3, 3))}]
            }
layers = model_manager.get_layer_by_attributes(conditions)

```

6. **Get Layer by Instance**
```python
# Search for layers in the model by their instance type
layers = model_manager.get_layer_by_instance(nn.Conv2d)

```

7. **Delete Layer by Index**
```python
# Delete a layer from the model using its index
model_manager.delete_layer_by_index(['features', 0])
```

8. **Delete Layer by Attribute**
```python
# Delete layers from the model based on a specific attribute
model_manager.delete_layer_by_attribute('activation', 'relu', '==')
```
9. **Delete Layers by Conditions**
```python
# Delete layers from the model based on multiple conditions
conditions = {
    'and': [{'==': ('kernel_size', (1, 1))}, {'==': ('stride', (1, 1))}],
    'or': [{'==': ('kernel_size', (3, 3))}]
}
model_manager.delete_layer_by_attributes(conditions)
```
10. **Delete Layer by Instance**

```python
# Delete layers from the model by their instance type
model_manager.delete_layer_by_instance(nn.Conv2d)
```