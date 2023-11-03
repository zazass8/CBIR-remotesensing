import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

# Create a list of image transforms to apply
def augment():
    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),    # Random horizontal flip
        transforms.RandomVerticalFlip(),      # Random vertical flip
        transforms.ToTensor(),                # Convert to tensor, apply rescaling of pixels
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])            
    ])

    return transforms_train, transforms_val


def augment_fair1m(desired_size = (1000, 1000, 3)):
    transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),    
        transforms.RandomVerticalFlip(),
        ResizeIfNecessary(desired_size),   
        AddRGB(),
        transforms.ToTensor(),    
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    ])

    transforms_val = transforms.Compose([
        ResizeIfNecessary(desired_size),   
        AddRGB(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transforms_train, transforms_val

def reshape_features(feature_vectors):

    # Find the maximum dimensionality among all feature vectors
    max_dimension = max(len(vector) for vector in feature_vectors)

    # Pad feature vectors to the same dimension with zeros
    padded_feature_vectors = [vector + [0] * (max_dimension - len(vector)) for vector in feature_vectors]

    padded_feature_vectors = [features.view(features.shape[0], -1) for features in padded_feature_vectors]

    return padded_feature_vectors




# Create a custom transformation to add an alpha channel
class AddRGB:
    def __call__(self, img):
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return img


# Define your custom transformation for resizing
class ResizeIfNecessary:
    def __init__(self, desired_size):
        self.desired_size = desired_size

    def __call__(self, img):
        if img.size != self.desired_size[:2]:
            img = img.resize(self.desired_size[:2])
        return img
    
# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        self.data = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
        image = Image.open(image_path)
        label = self.data.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        return image, label
            
