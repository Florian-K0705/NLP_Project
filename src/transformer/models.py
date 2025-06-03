import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification
from preprocessing.embedding import load_embedding_model

class BasicTransformerModel(torch.nn.Module):

    def __init__(self, num_layers = 6, num_heads = 8, feature_dim = 512, num_classes = 28):
        super().__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(feature_dim, nhead=num_heads, batch_first=True)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.classifier = torch.nn.Linear(feature_dim, num_classes)

        self.embedding_model = load_embedding_model("fasttext", path="/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/fastText")



    def forward(self, text, device):

        # token_embedding should be of size (N, S, D). N = Batch size, S = Sequence length, D = dimension of embedding. (1, S, D)
        token_embeddings = self.embedding_model(text).unsqueeze(dim=0).to(device)

        out1 = self.encoder(token_embeddings)

        out2 = torch.mean(out1, dim=1)

        out3 = self.classifier(out2)

        out4 = torch.nn.functional.softmax(out3, dim=1)

        return out4
    

class NeuroBertSmallModel(torch.nn.Module):

    def __init__(self, num_classes=28):
        super().__init__()


        self.tokenizer = AutoTokenizer.from_pretrained("boltuix/NeuroBERT-Small")
        self.neuroBert_model = AutoModelForSequenceClassification.from_pretrained("boltuix/NeuroBERT-Small", num_labels=num_classes)


    def forward(self, text, device):
        
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens.to(device)

        out0 = self.neuroBert_model(**tokens)
        out = torch.nn.functional.softmax(out0.logits, dim=1)

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