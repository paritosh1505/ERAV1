
# Integration of PyTorch and First Nueral Network

In this Documentation, we are exploring PyTorch and how to start with Neural Networks.









## PyTorch

PyTorch is an open-source machine learning framework primarily used for building and training neural networks. It is an easy-to-use framework that offers computational efficiency, primarily due to the generation of dynamic computation graphs. These features make PyTorch the ideal choice for our project.

## Prerequisite

Before installing PyTorch, ensure that the following prerequisites are installed:

- Python: PyTorch is a Python library, so we must have Python installed on your system. Make sure to have Python version 3.6 or above installed.

- CUDA: If we intend to use PyTorch with GPU acceleration for improved performance, we will need to install CUDA. CUDA is a parallel computing platform provided by NVIDIA. The specific version of CUDA required depends on your PyTorch version and GPU.

- NumPy: PyTorch often utilizes NumPy for numerical operations. NumPy is a fundamental package for scientific computing in Python. We can install NumPy using the default Python package manager pip

```bash
  pip install numpy
```

## Pytorch Installtion

We can install pytorch using python default manager , pip

```bash
  pip install torch
```


## Project Description

Here we are using code "[Nueral netwrok Basic Code](https://colab.research.google.com/drive/1oVt7T6tb90Y1EXvFaIWgm72emqZZPXi4?usp=sharing)" which was discussed in the class. Here our objective is to divide the code in three files utils.py, model.py,S5.ipynb

## How we distributed Code in the files

Following Logic we used to distribiute our code "[Nueral netwrok Basic Code](https://colab.research.google.com/drive/1oVt7T6tb90Y1EXvFaIWgm72emqZZPXi4?usp=sharing)" in three different

- utils.py: This file contains a helper class and commonly used code snippets that are required across all models.
- model.py: This file contains the definition of the model architecture, layers, and related code.
- S5.ipynb: This file contains the remaining code that deals with model execution, and the results will be stored in this file.


### Usage

- ### 1. utils.py file
This file contains all the import , data preproecessing and graph plot. Library which we are importing here is 

####  Importing all important library
```bash
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
```
#### Checking if Cuda is avaliable or not

```bash
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
```
#### Train and test data trasnformation

__*1.1 Training data tranformation*__

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

__*1.2 Testing data tranformation*__

To make our testing data simple and our training data more challenging, we refrain from introducing any randomness during the testing phase. This approach aims to enhance the model's efficiency when it is applied to the test data.

```bash
test_transforms = transforms.Compose([
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
```

__*1.3 Train and test data load*__

Here we are downloading both the train and test data and applying the transformations train_transforms and test_transforms, which we defined in point 1.1 and point 1.2, respectively. The downloaded data is being stored in the '../data' directory. 
```bash
train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True,transform=test_transforms)
```

__*1.4 Setting up train and test data loader*__

This code snippet sets up the data loaders for both training and testing datasets, with the specified batch size and other desired configurations which is defined in **kwargs

```bash
batch_size = 512

kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
```

- ### 2. Model.py file

This file is used to define the model architecture. We import the nn class from the utils.py file, where it was already imported. By importing the util file, we can access the nn class and use it in this code.

```bash
import utils as utilval

class Net(utilval.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = utilval.nn.Conv2d(1, 32, kernel_size=3)#Input = 1*28*28 | output = 32 * 26 * 26 = 21632  | parameter = 1*32*9+32 = 320 | 
        self.conv2 = utilval.nn.Conv2d(32, 64, kernel_size=3)#Input = 32*26*26 | output = 64 * 24 * 24 = 36864 | parameter = 64*32*9+64 = 18496
        self.conv3 = utilval.nn.Conv2d(64, 128, kernel_size=3)#Input = 64*24*24 | output = 128 * 22 * 22 = 61952| parameter = 64*128*9+128 = 73856
        self.conv4 = utilval.nn.Conv2d(128, 256, kernel_size=3)#Input = 128*22*22 | output =256 * 20 * 20 = 102400 | parameter = 128*256*9+256 = 295168
        self.fc1 = utilval.nn.Linear(4096, 50)#parameter = 256*20 = 12800 +50 =12850
        self.fc2 = utilval.nn.Linear(50, 10)#paramtere = 50*10+10 = 510


def forward(self, x):
        x = utilval.F.relu(self.conv1(x), 2)
        x = utilval.F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = utilval.F.relu(self.conv3(x), 2)
        x = utilval.F.relu(F.max_pool2d(self.conv4(x), 2)) 
        x = x.view(-1, 4096)
        x = utilval.F.relu(self.fc1(x))
        x = self.fc2(x)
        return utilval.F.log_softmax(x, dim=1)

```
__*2.1 Why we added value 4096 during flattening*__

Lets deep dive into forward function 
 - for first layer initallaly input size is 28x28 which become 26*26 and receptive field become 1x1 to 3x3
 ```bash
         x = utilval.F.relu(self.conv1(x))

 ```
 - for second layer initallaly input size is 26x26 which become 24*24 and with max pooling it become 12x12 receptive field become 3x3 to 5x5 to 6x6
  ```bash
         x = utilval.F.relu(F.max_pool2d(self.conv2(x), 2)

 ```
 - for third layer ,image size become 10x10 from 12x12 and receptive field become 6x6 to 10x10

 ```bash
                 x = utilval.F.relu(self.conv3(x), 2)


 ```
 - for final input layer (before we are flattening the inut) become image size become 8x8 and after max pooling it become 4x4
 ```bash
         x = utilval.F.relu(F.max_pool2d(self.conv4(x), 2)) 

 ```
 so final image size is 4*4 and output channel is 256 hence input data will be 4x4x256= 4096

### 3. S5.ipynb file

This file contains the remaining code. In this file, we are basically dealing with model execution. Here, we will import the utils and model files and implement the functionality of the code present in utils.py and model.py.

Main code where we are doing execution is 

```bash

device = utils.torch.device('cuda:0' if utils.torch.cuda.is_available() else 'cpu')
print("device name",device)
model = Model.Net().to(device)
print("model val",model)
optimizer = utils.optim.SGD(model.parameters(), lr=0.0175, momentum=0.95)
scheduler = utils.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)

num_epochs = 20
for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  train(model, device, utils.train_loader, optimizer)
  scheduler.step()
  test(model, device, utils.test_loader)
```
In the above code we prepares the execution environment, initializes the model, optimizer, and learning rate scheduler, trains the model for a specific number of epochs, and assesses its performance using test data. The iterative training process, combined with the learning rate scheduler, aims to enhance the model's accuracy and generalization capabilities.

Finally after code getting executed the model summary is as Follow

```bash
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```

### Conclusion
The provided code achieves an accuracy of 99.45% on the MNIST dataset. However, it's worth noting that the total number of parameters in the model architecture is quite high. This suggests that the current architecture might not be the most suitable choice for training the MNIST dataset.

During a previous discussion or session, it was determined that an optimal architecture for the MNIST dataset can have as few as 3500 parameters. Therefore, significant modifications are required in the code to achieve this desired parameter count.

- Architecture Modification
- Hyper paramter Tunig
- Model Optimization
- Regularization
- Hyper Paramter Tuning
