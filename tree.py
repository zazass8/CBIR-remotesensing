import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from transformations import augment
from sklearn.neighbors import KDTree

# KD-Tree
def kdtree(features):

    # reshape the features
    features = features.view(features.shape[0], -1)

    # Initialise the tree
    tree = KDTree(features, leaf_size = 40)

    # Save the KD-Tree using pickle
    with open('kd_tree.pkl', 'wb') as file:
        pickle.dump(tree, file)

def search(query_image_path, tree_name, cnn, test_loader):

    # convert image to tensor
    image = Image.open(query_image_path)
    _, transform_val = augment()
    image_tensor = transform_val(image)
    image_tensor = image_tensor.unsqueeze(0)

    # extract features from query image
    # EfficientNet-b0
    feature_vector = cnn.extract_features(image_tensor)


    # ResNet-50
    # feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
    # feature_vector = cnn.feature_extractor(image_tensor)

    feature_vector = feature_vector.view(feature_vector.shape[0], -1)

    # Load the KD-Tree from the saved file
    with open(tree_name, 'rb') as file:
        tree = pickle.load(file)
    
    output1, output2, output3 = tree.query(feature_vector.detach().numpy(), k = 3, return_distance = False)[0] 
    image1 = list(test_loader.dataset)[output1][0]
    image2 = list(test_loader.dataset)[output2][0]
    image3 = list(test_loader.dataset)[output3][0]

    # reverse normalising
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(3):
        image1[i] = image1[i]*std[i] + mean[i]
        image2[i] = image2[i]*std[i] + mean[i]
        image3[i] = image3[i]*std[i] + mean[i]

    # Reverse rescaling
    image1 = (image1 * 255).numpy().astype('uint8')
    image1 = image1.transpose(1, 2, 0)

    image2 = (image2 * 255).numpy().astype('uint8')
    image2 = image2.transpose(1, 2, 0)

    image3 = (image3 * 255).numpy().astype('uint8')
    image3 = image3.transpose(1, 2, 0)
 

    # Plot the top-k candidates
    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    plt.title("Query image")
    plt.axis("off")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    axes[0].imshow(image1)
    axes[0].set_title("Output 1")  
    axes[0].axis('off')

    axes[1].imshow(image2)
    axes[1].set_title("Output 2")  
    axes[1].axis('off')

    axes[2].imshow(image3)
    axes[2].set_title("Output 3")  
    axes[2].axis('off')

    plt.show()