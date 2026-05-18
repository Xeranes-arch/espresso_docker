import json
import espressomd

features = espressomd.features()

filename = "compiled_features.json"

with open(filename, "w") as f:
    json.dump(features, f)