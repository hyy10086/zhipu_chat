import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from Configs import rerank_model_path


class RerankerModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.model.to(self.device)

    def score_pairs(self, sentence_pairs):
        """
        Processes a list of sentence pairs to produce their scores.
        """
        inputs = self.tokenizer(sentence_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs_on_device, return_dict=True).logits.view(-1, ).float()
        scores = torch.sigmoid(logits)
        return scores


if __name__ == '__main__':
    # Usage example:
    sentence_pairs = [('今天天气很好', '天气晴朗'), ('今天天气很好', '小明在墨西哥')]
    reranker_model = RerankerModel(rerank_model_path)
    scores = reranker_model.score_pairs(sentence_pairs)
    print(scores)
