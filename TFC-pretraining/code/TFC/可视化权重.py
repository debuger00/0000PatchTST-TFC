import torch
from model import TFC
from PatchTST import PatchTSTNet
from config_files.AAR_Configs import Config

def print_model_weights_shape(checkpoint_path):
    # Load the saved model weights
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Check if 'model_state_dict' is in the checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Initialize the model
    configs = Config()
    model = TFC(configs)
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    
    # Print the shape of encoder_t and encoder_f weights
    print("Encoder T weights shape:")
    for name, param in model.encoder_t.named_parameters():
        print(f"{name}: {param.shape}")
    
    print("\nEncoder F weights shape:")
    for name, param in model.encoder_f.named_parameters():
        print(f"{name}: {param.shape}")

def export_encoder_t_weights(checkpoint_path, export_path):
    # Load the saved model weights
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Check if 'model_state_dict' is in the checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Initialize the model
    configs = Config()
    model = TFC(configs)
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    
    # Extract encoder_t weights, excluding classifier parameters
    encoder_t_weights = {name: param for name, param in model.encoder_t.named_parameters() if 'classifier' not in name}
    
    # Save encoder_t weights to a file
    torch.save(encoder_t_weights, export_path)
    print(f"Encoder T weights have been exported to {export_path}")

def print_weights_shapes(weights_path):
    # Load the encoder_t weights
    encoder_t_weights = torch.load(weights_path)
    
    # Print the shape of each parameter
    for name, param in encoder_t_weights.items():
        print(f"{name}: {param.shape}")



if __name__ == "__main__":
    checkpoint_path = "E:/program/aaa_DL_project/0000PatchTST-TFC/TFC-pretraining/code/experiments_logs/AAR_2_AAR/run1/pre_train_seed_42_2layertransformer/saved_models/ckp_last.pt"
    print_model_weights_shape(checkpoint_path)

    export_path = "E:/program/aaa_DL_project/0000PatchTST-TFC/TFC-pretraining/code/experiments_logs/AAR_2_AAR/run1/pre_train_seed_42_2layertransformer/saved_models/encoder_t_weights.pt"
    export_encoder_t_weights(checkpoint_path, export_path)
    print("#"*100)
    weights_path = "E:/program/aaa_DL_project/0000PatchTST-TFC/TFC-pretraining/code/experiments_logs/AAR_2_AAR/run1/pre_train_seed_42_2layertransformer/saved_models/encoder_t_weights.pt"
    print_weights_shapes(weights_path)