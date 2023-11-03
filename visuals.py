import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Create a function to display images and labels
def show_images_with_labels(image_dir):
    labels = [f for f in os.listdir(image_dir)]
    st = image_dir + "/" + labels[2] + "/" + os.listdir(image_dir + "/" + labels[2] + "/")[0]
    print(st)
    if st.endswith(".tif"):
        labels = [label for label in labels[1:] if not label.endswith("labels")]
    if st.endswith(".jpg"):
        labels = labels[1:] 

    for label in labels:
        image_files = [image_dir + "/" + label + "/" + f for f in os.listdir(image_dir + "/" + label + "/")]
        fig, axes = plt.subplots(5, 4, figsize=(10, 10))

        for i, image_file in enumerate(image_files[:20]):

            # Create the full path to the image
            image_path = os.path.join(image_dir, image_file)
        
            # Open the image using Pillow
            image = Image.open(image_path)

            row_idx = i // 4
            col_idx = i % 4

            # Display the image
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].set_title(label + " " + str(i + 1))  # Use the filename as the label
            axes[row_idx, col_idx].axis('off')

            for i in range(20, 4 * 5):
                row_idx = i // 4
                col_idx = i % 4
                fig.delaxes(axes[row_idx, col_idx])

        plt.tight_layout()  # Adjust the layout to prevent overlapping
        plt.show()


def gradcam(model, img):
    # Set the model in evaluation mode
    model.eval()

    # Convert the image to a tensor and perform necessary preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),      
        transforms.ToTensor(),            
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = preprocess(img).unsqueeze(0)

    # Enable gradient computation for the input image
    img.requires_grad = True

    # Forward pass to get the model's prediction
    output = model(img)
    _, class_index = output.max(1)
    class_index = class_index.item()

    # Perform a backward pass
    model.zero_grad()
    output[0, class_index].backward()

    # Get the gradient of the target layer
    # gradients = img.grad.data
    if str(type(model)).startswith("<class 'efficientnet"):
        gradients = model._conv_head.weight.grad
        pooled_gradients = torch.mean(gradients, dim=(1, 2, 3))

        # Get the activations of the target layer
        target = model.extract_features(img).detach()

    elif str(type(model)).startswith("<class 'torchvision"):
        gradients = model.layer4[2].conv3.weight.grad
        pooled_gradients = torch.mean(gradients, dim=(1, 2, 3))

        # Get the activations of the target layer
        feature_extractor = nn.Sequential(*list(model.children())[:-2])
        target = feature_extractor(img).detach()
        

    # Weight the activations by the gradients
    for i in range(target.shape[1]):
        target[0, i] *= pooled_gradients[i]

    # Calculate the heatmap
    heatmap = target[0].sum(dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def learning_curves(history):
    
    plt.clf()

    # Plot Output Loss
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
                        y=history['train_loss'],
                        name='Train'))
    
    fig.add_trace(go.Scattergl(
                        y=history['val_loss'],
                        name='Valid'))
    
    fig.update_layout(height=500,
                    width=700,
                    title='Output loss',
                    xaxis_title='Epoch',
                    yaxis_title='Loss')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
                        y=history['train_acc'],
                        name='Train'))
    
    fig.add_trace(go.Scattergl(
                        y=history['val_acc'],
                        name='Valid'))
    
    fig.update_layout(height=500,
                    width=700,
                    title='Accuracy',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy (%)')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
                        y=history['train_f1'],
                        name='Train'))
    
    fig.add_trace(go.Scattergl(
                        y=history['val_f1'],
                        name='Valid'))
    
    fig.update_layout(height=500,
                    width=700,
                    title='F1-Score',
                    xaxis_title='Epoch',
                    yaxis_title='F1-score')
    fig.show()


def confusion(y_true, y_pred):

    cf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues')

