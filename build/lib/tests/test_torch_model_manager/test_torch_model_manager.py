import unittest
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from torch_model_manager import torch_model_manager

from torchvision import models
import torch.nn as nn

class TestTorchModelManager(unittest.TestCase):
    def setUp(self) -> None:
        self.model = models.vgg16(pretrained=True)
        self.manager = torch_model_manager.TorchModelManager(self.model)
    
    def test_get_layer(self):
        layer = self.manager.get_layer_by_index(['classifier'])
        self.assertEqual(layer, self.model.classifier)
    
    def test_get_layer_nested(self):
        layer = self.manager.get_layer_by_index(['features', 0])
        self.assertEqual(layer, self.model.features[0])
    
    def test_get_layer_by_attribute(self):
        layer = self.manager.get_layer_by_attribute('in_features', 4096, '==')
        self.assertEqual(list(layer.values())[1], self.model.classifier[-1])
        self.assertEqual(list(layer.keys())[1], ('classifier', 6))

    def test_get_layer_by_attribute_callable(self):
        layer = self.manager.get_layer_by_attribute('in_features', 4096, lambda x, y: x == y and x > 0)
        self.assertEqual(list(layer.values())[1], self.model.classifier[-1])
        self.assertEqual(list(layer.keys())[1], ('classifier', 6))

    def test_get_layer_by_instance(self):
        layer = self.manager.get_layer_by_instance(nn.Linear)
        self.assertEqual(list(layer.keys())[-1], ('classifier', 6))
    
    def test_get_layer_by_instance_not_found(self):
        layer = self.manager.get_layer_by_instance(nn.Tanh)
        self.assertEqual(layer, {})
    
    def test_get_layer_by_attribute_not_found(self):
        layer = self.manager.get_layer_by_attribute('in_features', 1000000, '>')
        self.assertEqual(layer, {})

    def test_get_layer_by_attributes(self):
        conditions = {
            'and': [{'==': ('kernel_size', (1, 1))}, {'==': ('stride', (1, 1))}],
            'or': [{'==': ('kernel_size', (3, 3))}]
            }
        layer = self.manager.get_layer_by_attributes(conditions)
        self.assertEqual(list(layer.keys())[0], ('features', 0))
        self.assertEqual(list(layer.keys())[1], ('features', 2))

    def test_delete_layer(self):
        self.manager.delete_layer_by_index(['classifier', 6])
        self.assertIsInstance(self.model.classifier[-1], nn.Dropout)
    


if __name__ == '__main__':
    unittest.main()
