import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class PatchEmbedding(nn.Module):
    def __init__(self, img_size:int=224, patch_size: int=16, in_channels=3):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=patch_size**2 * in_channels, kernel_size=patch_size, stride=patch_size, padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.conv2d(x)).permute(0, 2, 1)

class ViT(nn.Module):
    def __init__(self,
                 img_size: int=224,
                 patch_size: int=16,
                 in_channels=3,
                 embedding_dim: int=768,
                 num_heads:int =12,
                 mlp_size: int=3072,
                 num_transformers=12,
                 dropout=0.1,
                 num_classes: int=3,
                 device: str=device):
        super().__init__()
        self.num_patches = img_size**2 // patch_size**2
        assert(img_size**2 % patch_size**2 == 0)

        self.pathify = PatchEmbedding(img_size, patch_size, in_channels)

        self.class_token = nn.Parameter(torch.randn(size=(1, 1, embedding_dim)), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches+1, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                    nhead=num_heads,
                                                                    dim_feedforward=mlp_size,
                                                                    dropout=dropout,
                                                                    activation="gelu",
                                                                    batch_first=True,
                                                                    norm_first=True),
                                                         num_layers=num_transformers
                                                         )
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                    out_features=num_classes)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # Equation 1
        x = self.pathify(x)

        # expand class token's batch_size
        class_token = self.class_token.expand(size=(batch_size, -1, -1))
        # add class token embedding
        x = torch.cat((class_token, x), dim=1)
        # add position embedding
        x = x + self.position_embedding
        # embedding dropout
        x = self.embedding_dropout(x)
        # Equation 2-3
        x = self.transformer_encoder(x)

        # get class token logit and classify
        return self.classifier(x[:, 0, :])

