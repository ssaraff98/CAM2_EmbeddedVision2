import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.convolutionLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),  # inplace means it doesn't allocate additional memory for output and just changes the input directly
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fullyConnectedLayer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.convolutionLayer(x)
        x = x.view(x.size(0),-1)
        x = self.fullyConnectedLayer(x)
        return x

"""
Each batch is (10000x3072) numpy array where 10000 is the number of sample data
and 3072 is 32x32 pixels x3 channels from the images in the sample training batch.
The tensor inputted into the CNN must be of form (wxhxc). Hence, the data is
transformed into a (10000x32x32x3) tensor and normalized for optimization of model
by reduction of vanishing gradient after activation functions are used.
"""

toTensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainingSet = datasets.CIFAR10(root="./CIFAR", train=True, download=True, transform=toTensor)
trainingLoad = torch.utils.data.DataLoader(trainingSet, batch_size=4, shuffle=True, num_workers=2)

testingSet = datasets.CIFAR10(root="./CIFAR", train=False, download=True, transform=toTensor)
testingLoad = torch.utils.data.DataLoader(testingSet, batch_size=4, shuffle=False, num_workers=2)

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = Classifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    runningLoss = 0.0
    for i, data in enumerate(trainingLoad, 0):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()

    runningLoss /= len(trainingLoad)
    print("Iteration: {0} | Loss: {1}".format(epoch+1, runningLoss))

print('Finished Training')

PATH = './classifier_model.pth'
torch.save(model.state_dict(), PATH)

model = Classifier()
model.load_state_dict(torch.load(PATH))

class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
with torch.no_grad():
    for data in testingLoad:
        testimages, labels = data
        outputs = model(testimages)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(len(classes)):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
