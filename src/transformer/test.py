import torch
from nltk.tokenize import word_tokenize
import gensim.downloader
import data.data_utils as data_utils



def embedding(sentence):

    tokens = word_tokenize(sentence.lower())

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


def train(model, dataloader, optimizer, criterion, device, num_classes, num_epochs=10):

    oneHot = torch.eye(num_classes).to(device)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)

            optimizer.zero_grad()

            # TODO Embedding of the input


            outputs = model(inputs)
            loss = criterion(outputs, oneHot[labels])

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')




def main():

    model = GoEmotionsTransformerModel(num_layers=2, num_heads=1, feature_dim=50)
    goEmotionsDataset = data_utils.GoEmotionsDataset(path="/home/florian/Dokumente/Programme/H-BRS/1. Semester/Natural Language Processing/NLP_Project/data/goEmotions/data", split="train")
    
    print(goEmotionsDataset[0])


    i = 0
    example = embedding(goEmotionsDataset[i][0])


    out = model(example.unsqueeze(dim=0))
    print(torch.argmax(out).item())


