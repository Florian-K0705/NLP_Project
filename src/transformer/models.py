import torch

class GoEmotionsBasicTransformerModel(torch.nn.Module):

    def __init__(self, num_layers = 6, num_heads = 8, feature_dim = 512, num_classes = 28):
        super().__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(feature_dim, nhead=num_heads, batch_first=True)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.classifier = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, token_embeddings):
        
        # token_embedding should be of size (N, S, D). N = Batch size, S = Sequence length, D = dimension of embedding.

        out1 = self.encoder(token_embeddings)

        out2 = torch.sum(out1, dim=1)

        out3 = self.classifier(out2)

        out4 = torch.nn.functional.softmax(out3, dim=1)

        return out4