import json
from collections import OrderedDict

def load_category_names(file_path):
    with open(file_path, 'r') as file:
        category_mapping = json.load(file)

    category_mapping = {int(key): value for key, value in category_mapping.items()}
    sorted_category_mapping = OrderedDict(sorted(category_mapping.items()))

    return sorted_category_mapping