import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(32, 32, kernel_size=1, padding='same')  
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()  

    def forward(self, x):
        out = self.conv(x) 
        out = self.bn(out)
        out = self.relu(out)
        return out + x  


class FinalModel3(nn.Module):
    def __init__(self, num_blocks, num_classes, dropout_rate):
        super(FinalModel3, self).__init__()
        self.initial_conv = nn.Conv2d(3, 32, kernel_size=3, padding='same')
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) 
        
        self.residual_blocks = nn.ModuleList([ResidualBlock() for _ in range(num_blocks)])

        
        self.final_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, num_classes) 
        self.dropout = nn.Dropout(dropout_rate)  

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_pool(x)  

        for block in self.residual_blocks:
            x = block(x)

        
        x = self.final_pool(x)

        x = self.flatten(x)
        x = self.dropout(x)  
        x = self.fc(x)
        return x
