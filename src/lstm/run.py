import torch
import lstm.models
import preprocessing.embedding as Embedder
import lstm

import lstm.train as train
import lstm.test as test
import data.data_utils as d

path = r"C:\Users\bberg\Documents\Natural Language Processing Data"


def run(mode, dataset):

    glov_path = path + r"\glove.6B.50d.txt"

    if dataset == "Emotions":
        train_ds = d.EmotionsDataset(path + r"\Emotions", split="train")
        test_ds = d.EmotionsDataset(path + r"\Emotions", split="train")
        pth_path = path + r"\Emotions_model.pth"
        output_dim = 6
    elif dataset == "goEmotions":
        train_ds =  d.GoEmotionsDataset(path + r"\goEmotions", split="train")
        test_ds = d.GoEmotionsDataset(path + r"\goEmotions", split="train")
        pth_path = path + r"\goEmotions_model.pth"
        output_dim = 28
    elif dataset == "AppReview":
        train_ds = d.AppReviewsDataset(path + r"\AppReviews", split="train")       
        test_ds = d.AppReviewsDataset(path + r"\AppReviews", split="train")
        pth_path = path + r"\AppReview_model.pth"
        output_dim = 5
    else:
        print("Kein Dataset angegeben")
        return

    embedding_dim = 50
    batch_size = 100
    sequenz_size = 64
    num_epochs = 5

    embedder = Embedder.Embedding(glov_path, embedding_dim, sequenz_size)
    model = lstm.models.LSTMClassifier(embedding_dim=embedding_dim, output_dim=output_dim)

    try:
        model.load_state_dict(torch.load(pth_path, weights_only=True))
    except:
        pass


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()


    if mode=="train":
        train.train_lstm(model=model, dataset=train_ds, embedder=embedder, optimizer=optimizer, criterion=criterion, batch_size=batch_size, num_epochs=num_epochs, pth_path=pth_path)
        

    elif mode == "test":
        test.test(model, embedder, test_ds)

    else:
        with torch.no_grad():
            while True:
                text = input("WHAT:")
                emb = torch.stack(embedder.glove_embedding(text))
            
                predict = torch.argmax(model(emb))
                print(predict.item())

    return

