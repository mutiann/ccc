import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, RobertaModel, RobertaConfig
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_


class EdgeClassification(BertPreTrainedModel):
    def __init__(self, config, hparams):
        config.hidden_dropout_prob = hparams.dropout_rate
        config.attention_probs_dropout_prob = hparams.dropout_rate
        super(EdgeClassification, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hparams.use_type_classifier:
            self.num_labels = hparams.n_edge_types
        else:
            self.num_labels = 2
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()
        self._hparams = hparams

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [B, C]
        probs = torch.softmax(logits, -1)  # [B, C]
        preds = torch.argmax(probs, -1)  # [B]
        results = {'logits': logits, 'probs': probs, 'preds': preds}

        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits, labels)
            results['loss'] = loss.mean()
            results['sample_losses'] = loss

        return results

class RobertaEdgeClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, hparams):
        config.hidden_dropout_prob = hparams.dropout_rate
        config.attention_probs_dropout_prob = hparams.dropout_rate
        super(RobertaEdgeClassification, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hparams.use_type_classifier:
            self.num_labels = hparams.n_edge_types
        else:
            self.num_labels = 2
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()
        self._hparams = hparams

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, **kwargs):
        token_type_ids = None
        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [B, C]
        probs = torch.softmax(logits, -1)  # [B, C]
        preds = torch.argmax(probs, -1)  # [B]
        results = {'logits': logits, 'probs': probs, 'preds': preds}

        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits, labels)
            results['loss'] = loss.mean()
            results['sample_losses'] = loss

        return results