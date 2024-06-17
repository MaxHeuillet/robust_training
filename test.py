import torch 

def compare_state_dict_keys(model, saved_state_dict_path):
    model_state_dict = model.state_dict()
    saved_state_dict = torch.load(saved_state_dict_path, map_location='cpu')

    model_keys = set(model_state_dict.keys())
    saved_keys = set(saved_state_dict.keys())

    missing_keys = model_keys - saved_keys
    unexpected_keys = saved_keys - model_keys

    if missing_keys:
        print("Missing keys in the saved state dictionary:")
        for key in missing_keys:
            print(key)
    else:
        print("No missing keys in the saved state dictionary.")

    if unexpected_keys:
        print("Unexpected keys in the saved state dictionary:")
        for key in unexpected_keys:
            print(key)
    else:
        print("No unexpected keys in the saved state dictionary.")

# Use the same model and saved state dict path
model = ResNet50WithParametrizations().to(device)
compare_state_dict_keys(model, saved_state_dict_path)
