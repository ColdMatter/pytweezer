import json

class DefaultExperimentalParameters:
    def __init__(self, filename="C:\\Users\\tweez\\ColdMatter\\pytweezer\\pytweezer\\experiment\\default_exp_parameters.json"):
        self.filename = filename
        self.parameters = self.load_parameters()

    def load_parameters(self):
        try:
            with open(self.filename, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_parameters(self):
        with open(self.filename, "w") as f:
            json.dump(self.parameters, f, indent=4)

    def add_parameter(self, key, value):
        self.parameters[key] = value
        self.save_parameters()

    def get_parameter(self, key):
        return self.parameters.get(key, None)