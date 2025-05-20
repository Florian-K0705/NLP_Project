from transformers import AutoModel, AutoTokenizer, AutoConfig


def main():

    model_name = 'Qwen/WorldPM-72B'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model_name,trust_remote_code=True )

    model = AutoModel.from_pretrained(
        model_name, 
        config = config, 
        device_map = "auto", 
        trust_remote_code=True,
    ).eval()


    text = "Hallo, wie geht es dir?"
    token = tokenizer.tokenize(text)

    print(tokenizer)


if __name__== "__main__":

    main()