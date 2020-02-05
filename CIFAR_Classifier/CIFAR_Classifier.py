import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.maxpool(nn.functional.relu(self.conv1(x)))
        x = self.maxpool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
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
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(4):
    runningLoss = 0.0

    for i, data in enumerate(trainingLoad, 0):
        images, labels = data
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        runningLoss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, runningLoss / 2000))
            runningLoss = 0.0

print('Finished Training')

PATH = './cifar_model.pth'
torch.save(model.state_dict(), PATH)

model = Classifier()
model.load_state_dict(torch.load(PATH))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
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

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
