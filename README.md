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
    'and': [
        {'==': ['name', 'layer1']},
        {'property': ['type', 'convolutional'], 'operator': '=='}
    ],
    'or': [
        {'property': ['name', 'layer2'], 'operator': '=='},
        {'property': ['type', 'pooling'], 'operator': '=='}
    ]
}
layers = model_manager.get_layer_by_attributes(conditions)

```