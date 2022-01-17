import torch
import torchvision.models as models

# Saving and Loading Model Weights
# Save
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# Load
model = models.vgg16()  # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()


# Saving and Loading Models with Shapes (Model must implement pickle protocol/interface)
# Save
torch.save(model, 'model.pth')

# Load
model = torch.load('model.pth')