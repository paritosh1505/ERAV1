## Code WalkThrough

As expalined in the conclusion section of [First Neural Network Architecture](https://github.com/paritosh1505/ERAV1/blob/master/IntroToPyTorch/README.md) We can reduce the paramter and intact our accuracy by addingregularization,Architecture change etc. All the things which was concluded in [First Neural Network Architecture](https://github.com/paritosh1505/ERAV1/blob/master/IntroToPyTorch/README.md) will be explained in this code walkthrough
Objective of this code: To achieve an accuracy greater than 99.4% with parameters less than 20K (with less than 20 epochs)

__Why less paramter is good__

There are several reasons why using fewer parameters is often beneficial while training a neural network. Some of these reasons include:

- __Reduced Overfitting__: Neural networks with a large number of parameters have a higher risk of overfitting. Overfitting occurs when a model learns the training data too well, leading to poor generalization to unseen data. By constraining the number of parameters, the model becomes less complex and is less likely to overfit.

- __Improved Generalization__: With fewer parameters, the model tends to capture the essential features and patterns in the data rather than memorizing noise or irrelevant details. This can result in better generalization performance, where the model performs well on unseen data.

- __Faster Training__: Neural networks with fewer parameters often require less computational resources and time for training. This can be advantageous when dealing with large datasets or limited computing power.

- __Easy Interpretation__: A neural network with fewer parameters is often easier to interpret and analyze. Understanding the relationship between the model's parameters and the output becomes more feasible, which can aid in model debugging and improvement.

## Understanding Code 

__Importing Library__

For the following code Structure we are importing all the library which we are going to use in this code . At first step it is importing torch library then we are importing specific part of torch library which is called nn ( neural networks), F (functional), and optim (optimization) and if we want to perform certain operation related to dataset here are using two library called datasets and transforms from library called torchvision

```bash
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

```
__Defining model architecture__

This is the most important part of the whole code because here we are defining the model architecture. In this architecture, we are incorporating various techniques such as output channel selection, batch normalization, dropout, and max pooling in a way that the number of parameters remains below 20k while achieving an accuracy greater than 99.4%.

The model architecture is designed to form a virtual bi-directional funnel. Initially, the channel size is set to a higher value, gradually decreasing until the first max pooling operation is applied. After the max pooling, the channel size increases again. This design allows the model to capture relevant features effectively while reducing the computational complexity. Following is the model architecture

```bash
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 3, padding=1) #28
        self.bn1 = nn.BatchNorm2d(24)
        self.dp1 = nn.Dropout(0.05)
        self.conv2 = nn.Conv2d(24, 20, 3, padding=1)#28
        self.bn2 = nn.BatchNorm2d(20)
        self.dp2 = nn.Dropout(0.05)
        self.conv3 = nn.Conv2d(20, 15, 3, padding=1)#28
        self.bn3 = nn.BatchNorm2d(15)
        self.dp3 = nn.Dropout(0.05)
        self.pool1 = nn.MaxPool2d(2, 2)#14
        
        self.conv4 = nn.Conv2d(15, 24, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.dp4 = nn.Dropout(0.05)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(24, 20, 3)
        self.bn5 = nn.BatchNorm2d(20)
        self.dp5 = nn.Dropout(0.05)
        
        self.conv6 = nn.Conv2d(20, 14, 3)
        self.bn6 = nn.BatchNorm2d(14)
        self.dp6 = nn.Dropout(0.05)
        self.conv7 = nn.Conv2d(14, 10, 3)

    def forward(self, x):
        x = F.relu(self.dp2((self.bn2(self.conv2(F.relu(self.dp1(self.bn1(self.conv1(x)))))))))
        x=  self.pool1(F.relu(self.dp3(self.bn3(self.conv3(x)))))
        x = self.pool2(self.dp4(self.bn4(F.relu(self.conv4(x)))))
        x = F.relu(self.dp5(self.bn5(self.conv5(x))))
        x = F.relu(self.dp6(self.bn6(self.conv6(x))))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10)
        return F.log_softmax(x)
```

In the provided architecture, we have a class called "Net" which represents our neural network. Inside this class, we define different layers and operations that the network will perform.Each line starting with "self.conv" corresponds to a convolutional layer. These layers take an input, apply filters to it, and generate an output. They are responsible for extracting different features from the input data.The lines with "self.bn" indicate batch normalization. This technique normalizes the outputs of the convolutional layers, ensuring that the training process is more stable and efficient. It helps in improving the overall performance of the network.The lines containing "self.dp" represent dropout. During training, dropout randomly deactivates some neurons in the network. This helps prevent overfitting, where the network becomes too specialized to the training data, and improves the network's ability to generalize to new data.The lines with "self.pool" denote max pooling. Max pooling reduces the spatial dimensions of the input data, capturing the most important information while reducing computational complexity. It helps in summarizing the input data by retaining the most significant features.
The "forward" method is where the actual computations take place. Each line within this method corresponds to various operations performed on the data, such as convolutions, batch normalization, dropout, and activation functions.
The final line return F.log_softmax(x) applies a logarithmic transformation to the output and prepares it for classification tasks.

__Fetching total paramter used in the code__

Now we fetch total no of parameter is used to define the above model architecture . For that we can use folloing code
```bash
!pip install torchsummary
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```
Output of Above Code will be

```bash
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 24, 28, 28]             240
       BatchNorm2d-2           [-1, 24, 28, 28]              48
           Dropout-3           [-1, 24, 28, 28]               0
            Conv2d-4           [-1, 20, 28, 28]           4,340
       BatchNorm2d-5           [-1, 20, 28, 28]              40
           Dropout-6           [-1, 20, 28, 28]               0
            Conv2d-7           [-1, 15, 28, 28]           2,715
       BatchNorm2d-8           [-1, 15, 28, 28]              30
           Dropout-9           [-1, 15, 28, 28]               0
        MaxPool2d-10           [-1, 15, 14, 14]               0
           Conv2d-11           [-1, 24, 14, 14]           3,264
      BatchNorm2d-12           [-1, 24, 14, 14]              48
          Dropout-13           [-1, 24, 14, 14]               0
        MaxPool2d-14             [-1, 24, 7, 7]               0
           Conv2d-15             [-1, 20, 5, 5]           4,340
      BatchNorm2d-16             [-1, 20, 5, 5]              40
          Dropout-17             [-1, 20, 5, 5]               0
           Conv2d-18             [-1, 14, 3, 3]           2,534
      BatchNorm2d-19             [-1, 14, 3, 3]              28
          Dropout-20             [-1, 14, 3, 3]               0
           Conv2d-21             [-1, 10, 1, 1]           1,270
================================================================
Total params: 18,937
Trainable params: 18,937
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.21
Params size (MB): 0.07
Estimated Total Size (MB): 1.29
```

If we observe above code we can see that total number of paramter used in teh above is 18,937 which is less than our requirment i.e. 20k. Now our next task is to check if above architecture where paramter is less than 20k is accuracy is coming greater than 99.4% (which is our second requirment)

#### Train and test data trasnformation

__Training data Transfomration__

In the given transformations, the image is randomly cropped with a probability of 0.1. The cropping is performed with a size of 22 pixels. Following the cropping operation, the image data is resized to a size of 28x28 pixels. To introduce additional randomness, the image is rotated randomly between -15 degrees and +15 degrees. Since these operations will be performed in PyTorch, all the transformations need to be converted to tensors. Lastly, to make the data more stable and suitable for training, the image data is normalized by subtracting the mean of 0.1307 and dividing by the standard deviation of 0.3081.

```bash
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
```

__Test data Transformation__

To make our testing data simple and our training data more challenging, we refrain from introducing any randomness during the testing phase. This approach aims to enhance the model's efficiency when it is applied to the test data.

```bash
test_transforms = transforms.Compose([
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
```
#### Defining Train and Test data function

__Train Function__

in the following code we are defining train function and test function. We have a function called "train" that plays a crucial role in training our model. 
Inside this function, To keep track of progress, we create a visual indicator called a progress bar, which shows us how well the training is progressing. 
Next, we divide our training data into smaller groups called batches. We take these batches and present them to the model one by one. For each batch, we provide the model with input data, which in our case are MNIST image, and the corresponding labels or targets that we want the model to predict

After that we reset the optimizer. It's like giving the model a fresh start.

We feed the input data to our model, and it starts making predictions. The output it generates is then compared to the actual labels using a special loss function. This function helps us measure how well the model is performing compared to the correct answers.

To improve its performance, the model goes through a process called backpropagation which is explained in [Backporopogation](https://github.com/paritosh1505/ERAV1/blob/Backpropogation/BackPropogation/README.md). After that we update its parameters using the optimizer

```bash
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')

```

__Test Function__

We have a function called "test" that allows us to evaluate how well our trained model performs on a different set of data known as the test set.

Here model.eval() explains we're not making any adjustments or learning from the test data

Next, we iterate through each batch of the test data. We then pass the input data to our model and obtain the output. By comparing the output with the target labels using a loss function, we calculate the loss between the two.After that we are updating the "test_loss" variable by adding the loss of each batch.

To measure the model's accuracy, we compare its predicted labels (based on the output) with the actual target labels. When a predicted label matches the target label, we consider it a correct prediction and increment the "correct" variable accordingly.

After processing all the batches, we calculate the average test loss by dividing the total loss by the number of examples in the test set.

Finally, we print the average loss and accuracy of the model on the test set. This information provides us with insights into the model's performance, where a higher accuracy indicates better results.

```bash
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```
__Calculating model accuracy__

Now the final step is to calculate the accuracy and check if it is greater than 99.4%.Accuracy log is present in [Optimized Code](https://github.com/paritosh1505/ERAV1/blob/Backpropogation/BackPropogation/S6/S6.ipynb). 

```bash

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for epoch in range(1, 21):
    print(f'Epoch {epoch}')
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```

