import torch
import torch.nn as nn
import torch.nn.init as init

def init_weights(model, depth=1):
    """
    Initializes the weights of all linear layers in the model using Kaiming normal initialization,
    with scaling based on the layer's depth to stabilize gradients.

    Args:
        model (nn.Module): The PyTorch model to initialize.
        depth (int): The current depth of the layer in the network.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Scale initialization based on layer depth
            scale = 1.0 / (depth ** 0.5)
            # Apply Kaiming normal initialization to the weight matrix
            init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            # Scale the weights
            with torch.no_grad():
                module.weight.mul_(scale)
            # Initialize the bias to zero if it exists
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, (nn.Sequential, nn.ModuleList, nn.Module)):
            # Recursively apply to nested modules, increasing depth
            for child in module.children():
                init_weights(child, depth + 1)