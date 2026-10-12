import torch

import config


class WineQualityClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(len(config.FEATURES), 88),
            torch.nn.BatchNorm1d(88),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(88, 10),
            torch.nn.BatchNorm1d(10),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(p=0.10),
            torch.nn.Linear(10, config.NUM_CLASSES),
            torch.nn.Softmax(dim=1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[:8](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < 1 or x.shape[1] != len(config.FEATURES):
            raise ValueError(
                f"Input tensor must have shape (N, {len(config.FEATURES)}) where N >= 1."
            )

        return self.layers(x)
