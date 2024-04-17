import yaml
from argparse import Namespace

def parse_config_yaml_dict(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def convert_to_namespace(dict_in, args, dont_care):
    namespace = Namespace()
    
    for key, value in dict_in.items():
        setattr(namespace, key, value)
    
    # Assuming `args_list` is a list of arguments to ignore
    for arg in dont_care:
        delattr(namespace, arg)
    
    return namespace
