import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# The FashionMNIST features are in PIL Image format, and the labels are integers.
# For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors.
# To make these transformations, we use ToTensor and Lambda.

# ToTensor converts a PIL image or NumPy ndarray into a FloatTensor. and scales the image’s pixel intensity values in
# the range [0., 1.]

# Lambda transforms apply any user-defined lambda function. Here, we define a function to turn the integer into a
# one-hot encoded tensor. It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls
# scatter_ which assigns a value=1 on the index as given by the label y

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)