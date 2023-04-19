import torch.nn as nn
from transformers import DistilBertModel, DistilBertForSequenceClassification


class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs


class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output


class ArticleNetwork(nn.Module):
    def __init__(self, num_classes):
        super(ArticleNetwork, self).__init__()
        self.encoder = DistilBertFeaturizer.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.encoder.d_out, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def forward_features(self, x):
        return self.encoder(x)
