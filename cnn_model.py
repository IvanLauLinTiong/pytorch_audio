from torch import nn
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 4 CONV blocks -> Flatten -> Linear -> softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # (1, 64 , 44) --conv1--> (16, 66, 46) --MaxPool2d--> (16, 33, 23) -> ... ->(128, 5, 4) (filters, frequency, time)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128*5*4, 10)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear(x)
        # pred = self.softmax(x)
        return x


if __name__ == "__main__":
    model = CNN()
    summary(model, input_size=(1, 64, 44), device='cpu')
