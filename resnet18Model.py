import torch
import torch.nn as nn
from torchvision import models  

class CSIModelWithResNet(nn.Module):
    def __init__(self, text_embedding_dim, user_feature_dim, lstm_hidden_dim, fc_hidden_dim, dropout_rate=0.5):
        super(CSIModelWithResNet, self).__init__()

        # Capture Module: LSTM for text embeddings
        self.lstm = nn.LSTM(text_embedding_dim, lstm_hidden_dim, batch_first=True)
        self.fc_capture = nn.Sequential(
            nn.Linear(lstm_hidden_dim, fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )

        # ResNet18 Module for User Features
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adapt for 1-channel input
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Final Output Layer
        self.fc_output = nn.Linear(fc_hidden_dim, 1)
        self.sigmoid_output = nn.Sigmoid()

    def forward(self, text_embeddings, user_features):
        # Capture Module: Process text embeddings with LSTM
        lstm_out, _ = self.lstm(text_embeddings)
        capture_output = self.fc_capture(lstm_out[:, -1, :])

        # ResNet18 Module: Process user features
        batch_size = user_features.size(0)
        # Reshape user_features to [batch_size, 1, 1, feature_dim]
        user_features = user_features.view(batch_size, 1, 1, -1)
        # Expand user_features to [batch_size, 1, 224, 224]
        user_features = user_features.repeat(1, 1, 224, 224 // user_features.size(-1))
        # Repeat channels to make 3-channel input for ResNet
        user_features = user_features.repeat(1, 3, 1, 1)
        # Pass through ResNet
        resnet_output = self.resnet(user_features)

        # Combine Outputs
        combined_output = capture_output + resnet_output

        # Final Classification Layer
        output = self.sigmoid_output(self.fc_output(combined_output))

        return output







