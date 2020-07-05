import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from typing import List
from .variables import target_cols

from transformers.modeling_bert import BertPreTrainedModel, BertModel


class BertMultipleProba(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_target = len(target_cols)
        self.targets = target_cols

        self.bert = BertModel(config)

        self.my_nn = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(2 * config.hidden_size, self.num_target),
        )

        self.init_weights()

    def forward(
            self,
            input_ids: torch.long = None,
            attention_mask: torch.float = None,
            token_type_ids: torch.long = None,
            target: torch.long = None
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

        last_hidden_states = inputs[0]

        q_mask = (attention_mask * (1 - token_type_ids)).unsqueeze(-1)
        a_mask = (attention_mask * token_type_ids).unsqueeze(-1)

        # -> (batch_size, config.hidden_size=784)
        q_input = (last_hidden_states * q_mask).sum(1) / q_mask.sum(1)
        a_input = (last_hidden_states * a_mask).sum(1) / a_mask.sum(1)
        # q_input, _ = torch.max(last_hidden_states * q_mask, dim=-2)
        # a_input, _ = torch.max(last_hidden_states * a_mask, dim=-2)

        # -> (batch_size, 2 * config.hidden_size)
        qa_input = torch.cat((q_input, a_input), dim=1)

        # -> (batch_size, self.num_target)
        outputs = self.my_nn(qa_input)

        if target is not None:
            loss_fct = BCEWithLogitsLoss1D(targets=self.targets)
            loss, loss_dict = loss_fct(outputs, target)
            outputs = loss, loss_dict, outputs

        return outputs


class BCEWithLogitsLoss1D(torch.nn.Module):
    """ multiple BCEWithLogitsLoss
    Shape:
    - Input: :math:`(batch_size, n,)`
    - Target: :math: `(batch_size, n,)`
    - Output: :math: `(n,)`
    """

    def __init__(self, targets: List[str]):
        super(BCEWithLogitsLoss1D, self).__init__()
        self.targets = targets
        self.elementwise_loss = BCEWithLogitsLoss(reduction='none')

    def forward(self, input_, target):

        loss = self.elementwise_loss(input_, target).mean(0)

        # used later in the summary writer
        loss_dict = dict((k, loss[i].item()) for i, k in enumerate(self.targets))

        return loss.mean(), loss_dict
