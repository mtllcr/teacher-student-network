
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from time import time
# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.optim as optim

def train_first_stage(dataloader, student, teacher, criterion_hint, epochs, learning_rate):
    """ Train only up to the intermediate layer """
    start = time()
    freeze_after_intermediate_layer2(student.features[4])
    regressor = ConvolutionalRegressor().to(device)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            teacher_layer_2_2 = Hook(teacher.layer2[2])
            student_layer_seq_4 = Hook(student.features[4])
            # intermediate outputs
            student.forward(inputs)
            with torch.no_grad():
                teacher_middle = teacher.forward(inputs)
            
            student_activations = student_layer_seq_4.output
            teacher_activations = teacher_layer_2_2.output
            teacher_activations_trans = regressor.forward(teacher_activations)
            #  the hint loss
            loss = criterion_hint(student_activations, teacher_activations_trans)
            loss.backward()
            optimizer.step()
          
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

    end = time()
    unfreeze_all_layers(student)
    print(f"First Stage Training Time: {end - start}")


def freeze_after_intermediate_layer2( layer):
        layer.requires_grad = False

def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True

      

class Hook():
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
       
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

class ConvolutionalRegressor(nn.Module):
    def __init__(self):
        super(ConvolutionalRegressor, self).__init__()
        # This regressor maps 32 channels to 16 channels
        self.regressor = nn.Conv2d(32, 16, kernel_size=(1, 1))  # 1x1 convolution

    def forward(self, x):
        return self.regressor(x)
    
def hint_loss(student_activations, teacher_activations):
    return torch.nn.functional.mse_loss(student_activations, teacher_activations).to(device)