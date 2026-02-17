import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

import os

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("images", transform = transform)
dataloader = DataLoader(dataset, batch_size = 20, shuffle = True)

class MonkeyNet(nn.Module):
    def __init__(self):
        super(MonkeyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 4)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = MonkeyNet().to(device)
    if os.path.exists("model_weights.pth"):
        print("Existing monkey model found, continuing training.")
        net.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    else:
        print("We are lowkey monkeyless. Creating new monkey model.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(15):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch: %d, total loss: %.3f" % (epoch, running_loss))

    print('Finished Training')

    torch.save(net.state_dict(), "model_weights.pth")