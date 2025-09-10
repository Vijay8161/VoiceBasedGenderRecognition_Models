import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(32, 8, kernel_size=1, padding='same')
        self.branch5x5_1 = nn.Conv2d(32, 8, kernel_size=1, padding='same')
        self.branch3x3_1 = nn.Conv2d(32, 8, kernel_size=1, padding='same')

    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch3x3 = self.branch3x3_1(x)
        outputs = torch.cat([branch1, branch5x5, branch3x3], dim=1)
        return outputs

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.inception = InceptionBlock()
        self.conv = nn.Conv2d(24, 32, kernel_size=1, padding='same')  # Changed to 128

    def forward(self, x):
        out = self.inception(x)
        out = self.conv(out)
        return out + x

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention, _ = self.attention(x, x, x)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        return self.norm2(forward + x)

class FinalModel(nn.Module):
    def __init__(self, num_blocks, num_classes, embed_size, heads, forward_expansion, dropout):
        super(FinalModel, self).__init__()
        self.initial_conv = nn.Conv2d(3, 32, kernel_size=3, padding='same')
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.residual_blocks = nn.ModuleList([ResidualBlock() for _ in range(num_blocks)])
        self.final_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.transformer_block = TransformerBlock(embed_size, heads, forward_expansion, dropout)
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_pool(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.final_pool(x)
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N, H * W, C)  # Ensure C is 128
        x = self.transformer_block(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
