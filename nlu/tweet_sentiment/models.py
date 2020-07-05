import torch
from torch import nn
from transformers.modeling_bert import BertModel


class BertClassification(nn.Module):

    def __init__(
            self,
            pretrained_bert_model_dir: str = 'bert-base-uncased',
            output_hidden_states: bool = False,
            output_attentions: bool = False
    ):
        super().__init__()
        self.num_target = 3
        self.target = 'target'
        self.class_map = {'negative': 0, 'neutral': 1, 'positive': 2}

        self.bert = BertModel.from_pretrained(
            pretrained_bert_model_dir,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions)

        self.hidden_size = self.bert.config.hidden_size
        self.batch_norm = nn.BatchNorm1d(num_features=self.hidden_size, momentum=0.1)
        self.linear = nn.Linear(self.hidden_size, self.num_target)

    def forward(
            self,
            input_ids: torch.long = None,
            attention_mask: torch.float = None,
            token_type_ids: torch.long = None,
            target: torch.long = None,
            weight: torch.float = None
    ):
        """
        Take the mean value for every channel (hidden_dimension) among all
        tokens that contribute to the question and to the answer, respectively.
        Concatenate both outputs and feed them to a linear model.
        """
        inputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

        # -> (batch_size, n_padded_tokens, config.hidden_size=784)
        last_hidden_states = inputs[0]

        # -> (batch_size, n_padded_tokens, 1)
        text_mask = attention_mask.unsqueeze(-1)

        # -> (batch_size, config.hidden_size=784)
        text_input, _ = torch.max(last_hidden_states * text_mask, dim=-2)
        # text_input = (last_hidden_states * text_mask).sum(-2) / text_mask.sum(-2)

        # -> (batch_size, config.hidden_size=784)
        text_input = self.batch_norm(text_input)

        # -> (batch_size, self.num_target)
        outputs = self.linear(text_input)

        if target is not None:
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(outputs, target)
            outputs = loss, outputs

        return outputs


class BertClassificationV2(nn.Module):

    def __init__(self, pretrained_bert_model_dir: str = 'bert-base-uncased'):
        super().__init__()
        self.num_target = 3
        self.target = 'target'
        self.class_map = {'negative': 0, 'neutral': 1, 'positive': 2}

        self.bert = BertModel.from_pretrained(
            pretrained_bert_model_dir,
            output_hidden_states=True,
            output_attentions=False)

        self.hidden_size = self.bert.config.hidden_size
        self.batch_norm = nn.BatchNorm1d(num_features=3 * self.hidden_size, momentum=0.1)
        self.linear = nn.Linear(self.hidden_size, self.num_target)

    def forward(
            self,
            input_ids: torch.long = None,
            attention_mask: torch.float = None,
            token_type_ids: torch.long = None,
            target: torch.long = None,
            weight: torch.float = None
    ):
        """
        Take the maximum value for every channel (hidden_dimension) among all
        tokens that contribute to the question and to the answer, respectively.
        Concatenate both outputs and feed them to a linear model.
        """
        # inputs -> (batch_size, n_padded_tokens, config.hidden_size=784)
        # hidden_states -> [
        #       (batch_size, n_padded_tokens, config.hidden_size=784),
        #       (batch_size, n_padded_tokens, config.hidden_size=784),
        #       ...
        #       ]
        last_hidden_state, _, hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

        # -> (batch_size, n_padded_tokens, 3 * self.hidden_size = 3* 784)
        relevant_hidden_states = torch.cat((last_hidden_state, hidden_states[0], hidden_states[1]), dim=-1)

        # -> (batch_size, n_padded_tokens, 1)
        text_mask = attention_mask.unsqueeze(-1)

        # -> (batch_size, 3 * self.hidden_size = 3 * 784)
        text_input, _ = torch.max(relevant_hidden_states * text_mask, dim=-2)

        # -> (batch_size, 3 * self.hidden_size = 3 * 784)
        text_input = self.batch_norm(text_input)

        # -> (batch_size, self.num_target)
        outputs = self.linear(text_input)

        if target is not None:
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(outputs, target)
            outputs = loss, outputs

        return outputs
