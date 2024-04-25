
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from captum.attr import Saliency, IntegratedGradients, NoiseTunnel, DeepLift
from captum.attr import visualization as viz
from time import time
import gc
# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_first_stage(dataloader, student, teacher, criterion_hint, epochs, learning_rate, student_layer, teacher_layer):
    """ Train only up to the intermediate layer """
    start = time()
    regressor = ConvolutionalRegressor2().to(device)
    student.train()
    teacher.eval()
    regressor.train()
    freeze_after_intermediate_layer2(student_layer)
    
    #optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    optimizer = optim.Adam(
    list(filter(lambda p: p.requires_grad, student.parameters())) + list(regressor.parameters()),
    lr=learning_rate
                        )
    losses_epoch = []
    teacher_layer_hook= Hook(teacher_layer)
    student_layer_hook= Hook(student_layer)
    for epoch in range(epochs):
        losses = []
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            #student_layer_hook = Hook(student_layer)
            # intermediate outputs
            logits= student(inputs)
            with torch.no_grad():
                teacher_middle = teacher(inputs)
            
            student_activations = student_layer_hook.output
            teacher_activations = teacher_layer_hook.output
            teacher_activations_trans = regressor.forward(teacher_activations)
            #print(f"teacher inter shape {teacher_activations_trans.shape}")
            #print(f'student inter shape {student_activations.shape}')
            #  the hint loss
            loss = criterion_hint(student_activations, teacher_activations_trans)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
        losses_epoch.append(np.mean(losses))

    end = time()
    unfreeze_all_layers(student)
    print(f"First Stage Training Time: {end - start}")
    student_layer_hook.close()
    teacher_layer_hook.close()
    regressor.to('cpu')
    optimizer.state.clear()
    del optimizer
    torch.cuda.empty_cache() 
    return losses_epoch

def train_kd_intermediate_combined(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, 
                          device, criterion_hint, student_layer, teacher_layer, early_stop = False):
    print('Knowledge distillation training')
    start = time()
    early_stopper = EarlyStopper(patience=3, min_improv=0.1)
    ce_loss_weight= 1 - soft_target_loss_weight
    ce_loss = nn.CrossEntropyLoss().to(device)
    regressor = ConvolutionalRegressor2().to(device)
    optimizer = optim.Adam(
    list(filter(lambda p: p.requires_grad, student.parameters())) + list(regressor.parameters()),
    lr=learning_rate
                        )
    teacher_layer_hook = Hook(teacher_layer)
    student_layer_hook = Hook(student_layer)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    regressor.train()
    losses_epoch = []

    for epoch in range(epochs):
        running_loss = 0.0
        losses = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            
            # Forward pass with the student model
            student_logits = student(inputs)

            student_activations = student_layer_hook.output
            teacher_activations = teacher_layer_hook.output
            teacher_activations_trans = regressor.forward(teacher_activations)
            #  the hint loss
            mse_loss = criterion_hint(student_activations, teacher_activations_trans)
            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft target loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss + mse_loss
            
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            losses.append(loss.item())
        losses_epoch.append(np.mean(losses))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        if early_stop and early_stopper.check_loss(running_loss):
          break
    end = time()
    runtime = end - start
    print(f"Training Time: {runtime:.3f}")
    regressor.to('cpu')
    student_layer_hook.close()
    teacher_layer_hook.close()
    optimizer.state.clear()
    del optimizer
    torch.cuda.empty_cache() 
    inputs, labels = inputs.to('cpu'), labels.to('cpu')
    return losses_epoch

def freeze_after_intermediate_layer2( layer):
        layer.requires_grad = False

def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True

      

class Hook():
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        print(f"Hook Set: {layer}")
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()



class ConvolutionalRegressor(nn.Module):
    def __init__(self):
        super(ConvolutionalRegressor, self).__init__()
        # This regressor maps 1024 channels to 64 channels
        self.regressor = nn.Conv2d(1024, 128, kernel_size=(1, 1))  # 1x1 convolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.regressor(x)
        x = self.adaptive_pool(x)
        return x    
    

class ConvolutionalRegressor2(nn.Module):
    def __init__(self, in_channels=1024, out_channels=128, kernel_size=4, stride=2, padding=1):
        super(ConvolutionalRegressor2, self).__init__()
        self.regressor = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.init_weights()
        
    def init_weights(self):
        init.kaiming_uniform_(self.regressor[0].weight, mode='fan_in', nonlinearity='relu')
        if self.regressor[0].bias is not None:
            init.constant_(self.regressor[0].bias, 0)

    def forward(self, x):
        return self.regressor(x)

    
def hint_loss(student_activations, teacher_activations):
    return torch.nn.functional.mse_loss(student_activations, teacher_activations).to(device)

class EarlyStopper:
    def __init__(self, patience=3, min_improv=0.01):
        self.patience = patience
        self.min_improv = min_improv
        self.counter = 0
        self.min_validation_loss = float('inf')

    def check_loss(self, validation_loss):
      # if validation loss improve by at least min_improv percentage, then
      # set min to current loss and reset the counter
        if validation_loss <  (self.min_validation_loss * (1 - self.min_improv)):
            self.min_validation_loss = validation_loss
            self.counter = 0
      # else if validation loss exceeds previous loss by the min_improv percentage,
      # start counter until hit patience
        elif validation_loss > (self.min_validation_loss * (1 + self.min_improv)):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

import matplotlib.pyplot as plt

def normalize_losses(loss_lists):
    """
    Normalize a list of loss lists using min-max scaling.
    """
    normalized_losses = []
    for losses in loss_lists:
        min_loss = min(losses)
        max_loss = max(losses)
        normalized = [(loss - min_loss) / (max_loss - min_loss) for loss in losses]
        normalized_losses.append(normalized)
    return normalized_losses

def plot_normalized_losses(loss_lists, labels, title="Normalized Loss over Epochs", xlabel="Epochs", ylabel="Normalized Loss"):
    """
    Plots normalized losses over epochs for multiple models on the same graph.

    Parameters:
    loss_lists (list of lists): A list where each element is a list of loss values for a model.
    labels (list of str): A list of labels for each model.
    title, xlabel, ylabel: Plot formatting parameters.
    """
    normalized_losses = normalize_losses(loss_lists)
    epochs = range(1, len(normalized_losses[0]) + 1)
    
    plt.figure(figsize=(10, 6))
    for losses, label in zip(normalized_losses, labels):
        plt.plot(epochs, losses, marker='o', linestyle='-', label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_model_accuracies(model_names, accuracies, title="Model Accuracies", xlabel="Models", ylabel="Accuracy"):
    """
    Plots a bar chart of model accuracies.

    Parameters:
    - model_names: A list of names of the models (str).
    - accuracies: A list of accuracies corresponding to the models (float).
    - title: Title of the plot (str).
    - xlabel: Label for the x-axis (str).
    - ylabel: Label for the y-axis (str).
    """
    # Set the positions and width for the bars
    positions = range(len(model_names))
    width = 0.5  # the width of the bars

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(positions, accuracies, width, align='center', alpha=0.7, color='b')

    # Adding the aesthetics
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(positions, model_names)  # Replace default x-ticks with models' names
    plt.ylim([0, 100])  # Assuming accuracy is between 0 and 1

    # Adding the accuracy values on top of the bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc, f"{acc}", ha='center')

    # Show the plot
    plt.tight_layout()
    plt.show()


