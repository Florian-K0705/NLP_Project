import torch
from nltk.tokenize import word_tokenize
import gensim.downloader
import data.data_utils as data_utils



def embedding(sentence):

    tokens = word_tokenize(sentence.lower())
    print(tokens)

    embed_model = gensim.downloader.load('glove-wiki-gigaword-50')
    emb = embed_model[tokens]

    return torch.Tensor(emb)



class GoEmotionsTransformerModel(torch.nn.Module):

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

        out4 = torch.nn.functional.softmax(out3)

        return out4





def main():

    model = GoEmotionsTransformerModel(num_heads=5, feature_dim=50)
    goEmotionsDataset = data_utils.GoEmotionsDataset(path="/home/florian/Dokumente/Programmierung/Python/NLP/NLP_Project/data/goEmotions", split="train")
    
    example = embedding(goEmotionsDataset[0][0])

    out = model(example.unsqueeze(dim=0))
    print(out)




if __name__== "__main__":
    main()