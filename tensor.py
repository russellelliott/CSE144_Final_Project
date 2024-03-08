import os
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import torch

def process_test_images(directory_path, target_size):
    data = []

    for img_name in sorted(os.listdir(directory_path)):
        image_path = os.path.join(directory_path, img_name)
        
        # Open image using PIL
        img = Image.open(image_path)
        
        # Convert the image to RGB
        img = img.convert('RGB') # Resolves an error with inconsistent channels
        
        # Resize image to target size
        img = img.resize(target_size)
        
        # Convert image to NumPy array
        img_array = np.array(img)
        
        # Append data
        data.append(img_array)

    # Convert to NumPy array
    data = np.array(data)

    return data
    

def process_images(directory_path, target_size):
    data = []
    labels = []

    for class_folder in sorted(os.listdir(directory_path)):
        class_path = os.path.join(directory_path, class_folder)
        
        for img_name in sorted(os.listdir(class_path)):
            image_path = os.path.join(class_path, img_name)
            
            # Open image using PIL
            img = Image.open(image_path)
            
            # Convert the image to RGB
            img = img.convert('RGB') 
            
            # Resize image to target size
            img = img.resize(target_size)
            
            # Transformations
            img_inverted = ImageOps.invert(img)  # Invert the image
            img_rotate = img.rotate(135)  # Rotate 135 degrees
            img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip the image horizontally
            img_brightness = ImageEnhance.Brightness(img).enhance(1.5)  # Increase brightness
            
            # Convert image to NumPy array
            img_array = np.array(img)
            img_inverted_array = np.array(img_inverted) / 255.0
            img_rotate_array = np.array(img_rotate) / 255.0
            img_flip_array = np.array(img_flip) / 255.0
            img_brightness_array = np.array(img_brightness) / 255.0
            
            # Append data and label
            data.extend([img_array, img_inverted_array, img_rotate_array, img_flip_array, img_brightness_array])
            labels.extend([int(class_folder)] * 5)  # Assuming folder name is the class label

    # Convert to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

train_data, train_labels = process_images('train/train', target_size=(32, 32))
test_data = process_test_images('test/test', target_size=(32, 32))


# Convert NumPy arrays to torch tensors
train_data = torch.tensor(train_data)
train_labels = torch.tensor(train_labels)
test_data = torch.tensor(test_data)


# Convert data to float
train_data = train_data.float()
test_data = test_data.float()

# Save tensors
torch.save(train_data, 'train_data.pth')
torch.save(train_labels, 'train_labels.pth')
torch.save(test_data, 'test_data.pth')