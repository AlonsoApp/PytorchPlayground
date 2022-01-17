import torch
x = torch.rand(5, 3)
print(x)

cuda = torch.cuda.is_available()

print("Is cuda available: {}".format(str(cuda)))