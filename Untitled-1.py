import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(num_inputs, num_hiddens)
        self.relu = nn.ReLU()
        self.output = nn.Linear(num_hiddens, num_outputs)
    
    def forward(self, X):
        X = self.flatten(X)
        X = self.hidden(X)
        X = self.relu(X)
        X = self.output(X)
        return X


def train_and_plot(model, train_loader, num_epochs=10, lr=0.1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')
    
   
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Curve')
    plt.legend()

    plt.show()
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

num_inputs = 784
num_outputs = 10

model_128 = MLP(num_inputs=num_inputs, num_outputs=num_outputs, num_hiddens=128)
train_and_plot(model_128, train_loader)

model_512 = MLP(num_inputs=num_inputs, num_outputs=num_outputs, num_hiddens=512)
train_and_plot(model_512, train_loader)

model_1024 = MLP(num_inputs=num_inputs, num_outputs=num_outputs, num_hiddens=1024)
train_and_plot(model_1024, train_loader)
