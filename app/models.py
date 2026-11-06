"""Assignment 1 architectures, implemented directly with torch.nn."""
import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64*28*28,128),
                                        nn.ReLU(), nn.Dropout(.3), nn.Linear(128,2))
    @property
    def cam_layer(self): return self.features[6]
    def forward(self,x): return self.classifier(self.features(x))

class BasicBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(incoming,outgoing,3,stride,1,bias=False)
        self.bn1=nn.BatchNorm2d(outgoing)
        self.conv2=nn.Conv2d(outgoing,outgoing,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(outgoing)
        self.shortcut=nn.Identity() if stride==1 and incoming==outgoing else nn.Sequential(
            nn.Conv2d(incoming,outgoing,1,stride,bias=False),nn.BatchNorm2d(outgoing))
        self.relu=nn.ReLU()
    def forward(self,x):
        y=self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y))+self.shortcut(x))

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,64,7,2,3,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(3,2,1))
        self.layer1=nn.Sequential(BasicBlock(64,64),BasicBlock(64,64))
        self.layer2=nn.Sequential(BasicBlock(64,128,2),BasicBlock(128,128))
        self.layer3=nn.Sequential(BasicBlock(128,256,2),BasicBlock(256,256))
        self.layer4=nn.Sequential(BasicBlock(256,512,2),BasicBlock(512,512))
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Dropout(.3),nn.Linear(512,2))
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    @property
    def cam_layer(self): return self.layer4
    def forward(self,x):
        x=self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        return self.fc(self.pool(x).flatten(1))

def make_model(name):
    if name=='simple_cnn': return SimpleCNN()
    if name=='resnet18': return ResNet18()
    raise ValueError('Unknown model')
