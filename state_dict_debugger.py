import torch

# Load the weights
checkpoint = torch.load("params/cv_weights/new_best_targeting_tall_plastic.pth", map_location="cpu")

# If it's a full model or a state_dict dictionary
state_dict = checkpoint if not hasattr(checkpoint, "state_dict") else checkpoint.state_dict()

# Print the last 5 layers and their tensor shapes
layers = list(state_dict.keys())
for layer in layers[-5:]:
    print(f"{layer}: {state_dict[layer].shape}")