import utils

class Net(utils.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = utils.nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = utils.nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = utils.nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = utils.nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = utils.nn.Linear(4096, 50)
        self.fc2 = utils.nn.Linear(50, 10)
    
    def forward(self, x):
        x = utils.F.relu(self.conv1(x))
        x = utils.F.relu(utils.F.max_pool2d(self.conv2(x), 2))
        x = utils.F.relu(self.conv3(x))
        x = utils.F.relu(utils.F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4096)
        x = utils.F.relu(self.fc1(x))
        x = self.fc2(x)
        return utils.F.log_softmax(x, dim=1)
