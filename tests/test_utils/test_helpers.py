import unittest
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.helpers import required_kernel, convert_to_int, create_dictionary, bi_operator, evaluate_conditions, intersect_dicts, union_dicts

class TestHelpers(unittest.TestCase):
    def test_equal_sizes_without_padding(self):
        size = 100
        self.assertEqual(required_kernel(size, size, padding=0), 1)
    
    def test_equal_sizes_with_padding(self):
        size = 100
        self.assertEqual(required_kernel(size, size, padding=1), 3)
        
    def test_smaller_output_size_without_padding(self):
        self.assertEqual(required_kernel(100, 50, padding=0), 51)
    
    def test_smaller_output_size_with_padding(self):
        self.assertEqual(required_kernel(100, 50, padding=2), 55)
        
    def test_smaller_output_size_with_padding_and_stride(self):
        self.assertEqual(required_kernel(100, 50, padding=1, stride=2), 4)
    
    def test_convert_to_int(self):
        self.assertEqual(convert_to_int(['1', '2', '3']), [1, 2, 3])

    def test_convert_to_int_nested(self):
        self.assertEqual(convert_to_int(['1', ['2', '3']]), [1, [2, 3]])
    
    def test_create_dictionary(self):
        keys = [(1, 2), ('s', 't', 4)]
        values = ['a', 'b']
        dictionary = {(1, 2): 'a', ('s', 't', 4): 'b'}
        self.assertEqual(create_dictionary(keys, values), dictionary)
    
    def test_bi_operator_equal(self):
        self.assertTrue(bi_operator('==', 1, 1))
    
    def test_bi_operator_not_equal(self):
        self.assertTrue(bi_operator('!=', 1, 2))
    
    def test_bi_operator_greater(self):
        self.assertTrue(bi_operator('>', 2, 1))
    
    def test_bi_operator_greater_equal(self):
        self.assertTrue(bi_operator('>=', 2, 2))
    
    def test_bi_operator_less(self):
        self.assertTrue(bi_operator('<', 1, 2))
    
    def test_bi_operator_less_equal(self):
        self.assertTrue(bi_operator('<=', 1, 1))
    
    def test_bi_operator_callable(self):
        self.assertTrue(bi_operator(lambda x, y: (x == y) and (x > 0), 1, 1))
    
    def test_intersect_dicts(self):
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'b': 2, 'c': 4}
        self.assertEqual(intersect_dicts(dict1, dict2), {'b': 2})
    
    def test_union_dicts(self):
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'b': 3, 'c': 4}
        self.assertEqual(union_dicts(dict1, dict2), {'a': 1, 'b': 3, 'c': 4})
    
    def test_union_dicts_empty_1(self):
        dict1 = {}
        dict2 = {'b': 3, 'c': 4}
        self.assertEqual(union_dicts(dict1, dict2), {'b': 3, 'c': 4})
    
    def test_union_dicts_empty_2(self):
        dict1 = {'a': 1, 'b': 2}
        dict2 = {}
        self.assertEqual(union_dicts(dict1, dict2), {'a': 1, 'b': 2})
    
    def test_union_dicts_empty(self):
        dict1 = {}
        dict2 = {}
        self.assertEqual(union_dicts(dict1, dict2), {})
    

if __name__ == "__main__":
    unittest.main()