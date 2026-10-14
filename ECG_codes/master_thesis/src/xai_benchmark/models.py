"""ECG model architectures """

import torch
import torch.nn as nn


class ECGCNN1D_Baseline(nn.Module):
    #2-block CNN baseline

    def __init__(self, in_ch: int = 12, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.fc(self.net(x).squeeze(-1))


class ECGCNN1D_Deeper(nn.Module):
    #3-block CNN with BatchNorm that we used in our thesis

    def __init__(self, in_ch: int = 12, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))


class ECGCNNLSTMAttn(nn.Module):
    #CNN feature extractor followed by a BiLSTM with attention pooling

    def __init__(
        self,
        in_ch: int = 12,
        num_classes: int = 2,
        dropout: float = 0.3,
        lstm_hidden: int = 64,
        attn_hidden: int = 64,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        lstm_out_dim = lstm_hidden * 2
        self.attn_mlp = nn.Sequential(
            nn.Linear(lstm_out_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn_mlp(out), dim=1)
        ctx = torch.sum(attn_w * out, dim=1)
        return self.fc(self.dropout(ctx))
