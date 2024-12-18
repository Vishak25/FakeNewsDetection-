import torch.nn as nn

class CSIModel(nn.Module):
    def __init__(self, text_embedding_dim, user_feature_dim, lstm_hidden_dim, fc_hidden_dim, dropout_rate=0.4): #changed dropout_rate, 0.3->0.5->0.4
        super(CSIModel, self).__init__()

        # Capture Module: LSTM for text embeddings
        self.lstm = nn.LSTM(text_embedding_dim, lstm_hidden_dim, batch_first=True)
        self.fc_capture = nn.Sequential(
            nn.Linear(lstm_hidden_dim, fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )

        # Score Module: Fully connected layer for user features
        self.fc_user = nn.Linear(user_feature_dim, fc_hidden_dim)
        self.sigmoid_user = nn.Sigmoid()

        # Final Output Layer
        self.fc_output = nn.Linear(fc_hidden_dim, 1)
        self.sigmoid_output = nn.Sigmoid()

    def forward(self, text_embeddings, user_features):
        # Capture Module: Process text embeddings with LSTM
        lstm_out, _ = self.lstm(text_embeddings)
        capture_output = self.fc_capture(lstm_out[:, -1, :])

        # Score Module: Process user features
        user_scores = self.sigmoid_user(self.fc_user(user_features))
        user_scores_mean = user_scores.mean(dim=1).unsqueeze(1)

        # Combine Outputs: Use summation
        combined_output = capture_output + user_scores_mean

        # Final Classification Layer
        output = self.sigmoid_output(self.fc_output(combined_output))

        return output
