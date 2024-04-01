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
        op(a, b)

# Multiple conditions test
def evaluate_conditions(conditions):
    for operator, operands in conditions.items():
        if operator == 'and':
            result = True
            for condition in operands:
                key = list(condition.keys())[0]
                value = list(condition.values())[0]
                
                result = result and bi_operator(key, value[0], value[1])
                
            return result
        elif operator == 'or':
            result = False
            for condition in operands:
                key = list(condition.keys())[0]
                value = list(condition.values())[0]
                result = result or bi_operator(key, value[0], value[1])
            return result
             
       
    return False
    
def intersect_dicts(dict1, dict2):
    intersection_dict = {}
    for key in dict1.keys() & dict2.keys():  # Using set intersection for keys
        if dict1[key] == dict2[key]:  # Ensure values are the same for the common key
            intersection_dict[key] = dict1[key]
    return intersection_dict

def union_dicts(dict1, dict2):
    return {**dict1, **dict2}