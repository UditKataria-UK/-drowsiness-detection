import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# -------------------------
# 1. CNN Model
# -------------------------

class DrowsinessCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------
# 2. Load Model
# -------------------------

model = DrowsinessCNN()

model.load_state_dict(
    torch.load("drowsiness_model.pth")
)

model.eval()


# -------------------------
# 3. Image preprocessing
# -------------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# -------------------------
# 4. Load an image
# -------------------------

image_path = "/Users/uditkataria/Desktop/dl_projects/open_closed_eye/dataset/test/Open_Eyes/s0001_02336_0_0_1_0_0_01.png"

image = Image.open(image_path).convert("RGB")

image = transform(image)

# Add batch dimension
image = image.unsqueeze(0)


# -------------------------
# 5. Prediction
# -------------------------

with torch.no_grad():

    output = model(image)

    _, predicted = torch.max(output, 1)


# -------------------------
# 6. Result
# -------------------------

classes = ["Closed_Eyes", "Open_Eyes"]

prediction = classes[predicted.item()]

print("Prediction:", prediction)

if prediction == "Closed_Eyes":
    print("😴 DROWSY")

else:
    print("👀 ALERT")