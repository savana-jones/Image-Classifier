import json
def get_label_map(cat_name_file):
    with open(cat_name_file, 'r') as file:
        data = json.load(file)
    
    category_mapping = {int(key): value for key, value in data.items()}
    sorted_category_mapping = dict(sorted(category_mapping.items(), key=lambda item: item[0]))

    return sorted_category_mapping