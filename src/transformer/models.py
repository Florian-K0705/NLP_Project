import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertForSequenceClassification
import fasttext

class BasicTransformerModel(torch.nn.Module):

    def __init__(self, num_layers = 6, num_heads = 8, feature_dim = 512, num_classes = 28):
        super().__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(feature_dim, nhead=num_heads, batch_first=True)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.classifier = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, text):

        
        # token_embedding should be of size (N, S, D). N = Batch size, S = Sequence length, D = dimension of embedding.
        token_embeddings = None
        out1 = self.encoder(token_embeddings)

        #out2 = torch.sum(out1, dim=1) / s  # Average pooling over the sequence length
        out2 = torch.mean(out1, dim=1)  # Alternative: Mean pooling over the sequence length

        out3 = self.classifier(out2)

        out4 = torch.nn.functional.softmax(out3, dim=1)

        return out4
    
class BertLiteModel(torch.nn.Module):

    def __init__(self, num_classes=28):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("boltuix/bert-lite")

        self.bert_model = AutoModelForMaskedLM.from_pretrained("boltuix/bert-lite")

        self.bert_model.cls.predictions.decoder = nn.Sequential(nn.Linear(256, 30522), nn.Linear(30522, num_classes), nn.Softmax())

    def forward(self, text):
        
        tokens = self.tokenizer(text, return_tensors="pt")

        out = self.bert_model(tokens["input_ids"])[0]

        return out
    

class BertModel(torch.nn.Module):

    def __init__(self, num_classes=28):
        super().__init__()


        self.tokenizer = AutoTokenizer.from_pretrained(
            "google-bert/bert-base-uncased",
        )

        self.bert = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")
        self.bert.classifier = nn.Sequential(nn.Linear(768, num_classes), nn.Softmax(dim=1))

        

    def forward(self, text, device="cpu"):
        
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens.to(device)

        outputs = self.bert(**tokens)
        predictions = outputs.logits

        return predictions





if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    token_ids = tokenizer.encode("Hello world")

    print(token_ids)