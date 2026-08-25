import json

class ExpParameterManager():
    def __init__(self, json_file="C:\\Users\\tweez\\ColdMatter\\pytweezer\\pytweezer\\experiment\\parameters.json"):
        self.json_file = json_file
        self.load_parameters()

    def load_parameters(self, category=None):
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        if category:
            return self.data.get(category, {})
        return self.data

    def get_parameter(self, param_name):
        for category in self.data.values():
            if param_name in category:
                return category[param_name]
        raise KeyError(f"Parameter '{param_name}' not found.")

    def set_parameter(self, param_name, value):
        for category in self.data.values():
            if param_name in category:
                category[param_name] = value
                print(f"Parameter '{param_name}' set to {value}.")
                return
        raise KeyError(f"Parameter '{param_name}' not found.")

    def save_parameters(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.data, f, indent=4)
            print("Parameters saved.")

    def add_parameter(self, param_name, value, category="misc_parameters"):
        if category not in self.data:
            self.data[category] = {}
        self.data[category][param_name] = value
        print(f"Parameter '{param_name}':{value} added to category '{category}'.")