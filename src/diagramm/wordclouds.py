from data.data_utils import GoEmotionsDataset, AppReviewsDataset, EmotionsDataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_wordCloud():
    dataset = prepare_Data()

    texts_per_label = {0: [], 1: [], 2: []}

    for record in dataset:
        label = record['label']
        text_no_stopwords = record['text_no_stopwords']
        texts_per_label[label].append(text_no_stopwords)

    text_strings = {label: " ".join(texts) for label, texts in texts_per_label.items()}

    colormaps = {
        0: 'Greens',     # positive
        1: 'Oranges',    # ambiguous
        2: 'Reds'        # negative
    }

    for label, text in text_strings.items():
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap=colormaps[label]
        ).generate(text)
    
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Wordcloud für Label {label}')
        plt.show()

def prepare_Data():
    datapath_goemotions="C:/Users/sarah/Downloads/data/goEmotions"
    datapath_emotions="C:/Users/sarah/Downloads/data/Emotions"

    train_go=GoEmotionsDataset(datapath_goemotions,split="train")
    val_go=GoEmotionsDataset(datapath_goemotions,split="val")
    test_go=GoEmotionsDataset(datapath_goemotions,split="test")
    train_em=EmotionsDataset(datapath_emotions,split="train")
    val_em=EmotionsDataset(datapath_emotions,split="val")
    test_em=EmotionsDataset(datapath_emotions,split="test")

    stopwords = set(ENGLISH_STOP_WORDS)

    dataset = []
#     train_go=[
#     {'text': 'Best I have day ever ever best', 'labels': {0}},
#     {'text': 'Ah Could be best worst better', 'labels': {2}},
#     {'text': 'I am Worst experience', 'labels': {6}}
# ]

    for ds, mapping in [
        (train_go, label_mapping_go),
        (val_go, label_mapping_go),
        (test_go, label_mapping_go),
        (train_em, label_mapping_em),
        (val_em, label_mapping_em),
        (test_em, label_mapping_em),
    ]:
        for record in ds:
            text = record[0]

            # for label, is_present in record['labels'].items():
            #     if is_present == 1:
            #         new_label = mapping[label]
            #         if new_label in [0, 1, 2]:
            #             dataset.append({ 
            #                 "text": text, 
            #                 "label": new_label,
            #                 "text_no_stopwords": remove_stopwords(text, stopwords)  # für Wordclouds
            #             })
            #             break
            new_label = mapping[record[1]]
            if new_label in [0, 1, 2]:
                dataset.append({
                 "text": text,
                 "label": new_label,
                 "text_no_stopwords": remove_stopwords(text, stopwords)
                })
    
    return dataset


def remove_stopwords(text,stopwords):
    return " ".join([word for word in text.split() if word.lower() not in stopwords])


label_mapping_go={
    0:0,        #0=positive
    1:0,
    2:2,        #3=negative
    3:2,
    4:0,
    5:0,
    6:1,         #1=ambiguous
    7:1,
    8:1,
    9:2,
    10:2,
    11:2,
    12:2,
    13:0,
    14:2,
    15:0,
    16:0,
    17:0,
    18:0,
    19:2,
    20:0,
    21:0,
    22:1,
    23:0,
    24:2,
    25:2,
    26:1,
    27:1        
}

label_mapping_em={
    0:2,        #0=positive
    1:0,        #1=ambiguous
    2:0,        #2=negative
    3:2,        
    4:2,
    5:1
}
