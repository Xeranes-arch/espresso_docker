import numpy as np

loaded = np.load(
    "/workspace/_data/mag_response/mag_response_data_ratio2.0_ani2.0.npz")
print(loaded["ani"])  # Access by the key you assigned
print(loaded["dipm_means"])
