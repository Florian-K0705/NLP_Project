import torch
import lstm.models as models
import preprocessing.embedding as Embedder
import lstm

import lstm.train as train
import lstm.test as test
import lstm.mapping as s
import data.data_utils as d

path = r"C:\Users\bberg\Documents\Natural Language Processing Data"


def run(mode, dataset):

    glov_path = path + r"\glove.6B.50d.txt"

    if dataset == "Emotions":
        train_ds = d.EmotionsDataset(path + r"\Emotions", split="train")
        test_ds = d.EmotionsDataset(path + r"\Emotions", split="val")
        pth_path = path + r"\Emotions_model_2.pth"
        output_dim = 6
    elif dataset == "goEmotions":
        train_ds =  d.GoEmotionsDataset(path + r"\goEmotions", split="train")
        test_ds = d.GoEmotionsDataset(path + r"\goEmotions", split="val")
        pth_path = path + r"\goEmotions_model_2.pth"
        output_dim = 28
    elif dataset == "AppReview":
        train_ds = d.AppReviewsDataset(path + r"\AppReviews", split="train")       
        test_ds = d.AppReviewsDataset(path + r"\AppReviews", split="val")
        pth_path = path + r"\AppReview_model_2.pth"
        output_dim = 5
    else:
        print("Kein Dataset angegeben")
        return

    embedding_dim = 50
    batch_size = 100
    sequenz_size = 128
    num_epochs = 100

    embedder = Embedder.Embedding(glov_path, embedding_dim, sequenz_size)
    model = models.LSTMClassifier(embedding_dim=embedding_dim, output_dim=output_dim)

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
    elif mode =="mapping":
        ds = d.GoEmotionsDataset(path + r"\goEmotions", split="train")
        s.simple_label_mapping(model, embedder, ds, 28)
        ds = d.EmotionsDataset(path + r"\Emotions", split="train") 
        s.simple_label_mapping(model, embedder, ds,6)
    else:
        with torch.no_grad():
            while True:
                text = input("WHAT:")
                emb = embedder.glove_embedding(text).unsqueeze(0)

                predict = torch.argmax(model(emb))
                print(predict.item())

    return

