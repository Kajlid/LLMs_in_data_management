import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.mps.is_available() else 'cpu')    # because I have MacOS
print(f"{str(device)}")
batch_size = 100

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = next(examples)

# Show three examples
for k in range(3):
    example_tensor = example_data[k][0]
    print(f"\n\n EXAMPLE {k} -> {example_targets[k].item()}")
    for i in range(0,28):
        line = ""
        for j in range(0,28):
            if example_tensor[i,j].item() > 0.5:
                line += '*'
            else:
                line += ' '
        print(line)



# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, num_classes):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_size, hidden_size_1)   # Hidden layer 1
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size_1, hidden_size_2)  # Hidden layer 2
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size_2, hidden_size_3)  # Hidden layer 3
        self.relu = nn.ReLU()
        self.l4 = nn.Linear(hidden_size_3, hidden_size_4)  # Hidden layer 4
        self.relu = nn.ReLU()
        self.l5 = nn.Linear(hidden_size_4, num_classes) # Output layer

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        # no activation and no softmax at the end!
        return out

# Hyper-parameters
input_size = 784 # 28x28
# hidden_size = 500
hidden_size_1 = 100
hidden_size_2 = 75
hidden_size_3 = 50
hidden_size_4 = 25
num_classes = 10
num_epochs = 2
learning_rate = 0.001

model = NeuralNet(input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass and loss calculation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        # max returns (output_value ,index)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

    acc = n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {100 * acc} %')