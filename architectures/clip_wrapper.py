import torch.nn as nn

class CLIPConvNeXtClassifier(nn.Module):
    def __init__(self, clip_model, num_classes, dropout=0.0):
        super().__init__()

        # Access the timm convnext backbone from OpenCLIP wrapper
        self.backbone = clip_model.visual  # this is a TimmModel
        self.embedding_dim = self.backbone.head.proj.out_features  # This will give you 640


        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)  # [B, D]
        return self.head(features)