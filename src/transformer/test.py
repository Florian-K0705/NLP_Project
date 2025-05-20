import torch


def main():

    

    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # Download vocabulary from S3 and cache.
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', './test/bert_saved_model/')  # E.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`

    print(tokenizer)


if __name__== "__main__":

    main()