import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification  

class NeuroBertSmallModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.neuroBert_small = AutoModelForSequenceClassification.from_pretrained("boltuix/NeuroBERT-Small", num_labels=28)


    def forward(self, inputs):
        
        o1 = self.neuroBert_small(inputs["input_ids"])
        out = torch.nn.functional.softmax(o1.logits, dim=1)

        return out