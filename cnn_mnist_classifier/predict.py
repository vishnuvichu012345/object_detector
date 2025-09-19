# predict.py
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from model import SimpleCNN

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()

# Load and preprocess test image (you can replace this with your own)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load a sample image from file (you can use your own)
img_path = "/home/Vishnu/Documents/study_ml;/object_detection/cnn_mnist_classifier/test_digit.png"  # replace with path to your test image
img = Image.open(img_path)
img_tensor = transform(img).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output.data, 1)
    prediction = predicted.item()

# Show result
plt.imshow(np.array(img), cmap='gray')
plt.title(f"Predicted: {prediction}")
plt.axis('off')
plt.show()
