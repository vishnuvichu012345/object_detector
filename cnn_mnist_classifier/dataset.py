from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
image, label = testset[0]
image = transforms.ToPILImage()(image)
image.save("test_digit.png")
print("Saved digit image with label:", label)
