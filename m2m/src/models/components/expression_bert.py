import math

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class ScaledSigmoid(nn.Module):
    def __init__(self, scale):
        super(FeatureClip, self).__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * torch.sigmoid(x)


class FeatureClip(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x):
        return torch.clamp(x, self.low, self.high)


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class ExpressionBert(nn.Module):
    def __init__(
        self,
        input_features: list = None,
        output_features: list = None,
        feature_boundaries: dict = None,
        max_position_embeddings: int = 256,
        vocab_names: list = None,
        position_embedding_type: str = "relative_key_query",
        hidden_size: int = 512,
        num_hidden_layers: int = 4,
        precision: int = 16,
        num_attention_heads: int = 4,
        intermediate_size: int = 128,
        output_attentions: bool = True,
        num_of_styles: int = 6,
        style_embed_integration_type: str = "add",
        style_emb_size: int = 128,
    ):
        super().__init__()

        configuration = BertConfig(
            max_position_embeddings=max_position_embeddings,
            position_embedding_type=position_embedding_type,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            precision=precision,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            output_attentions=output_attentions,
        )

        self.bert = BertModel(configuration)
        self.d_model = hidden_size
        self.hidden_size = hidden_size
        self.bertConfig = configuration
        self.max_len = max_position_embeddings

        if style_embed_integration_type == "add":
            self.style_emb_size = self.d_model
        else:
            self.style_emb_size = style_emb_size
        self.style_emb = Embeddings(num_of_styles, self.style_emb_size)
        self.style_embed_integration_type = style_embed_integration_type

        self.input_features = input_features
        self.output_features = output_features
        self.vocab_names = vocab_names
        self.feature_boundaries = feature_boundaries
        """For training a tranditional Bert model # token types: [Pitch, Velocity, Duration,
        Position, Bar] self.n_tokens = [89, 66, 4609, 1537, 518]      # Vocabulary size for
        different features self.classes = ['Pitch', 'Velocity', 'Duration', 'Position', 'Bar']
        self.emb_sizes = [32, 64, 512, 256, 128] # Embedding sizes.

        # word_emb: embeddings to change token ids into embeddings self.word_emb = [] for i in
        range(len(self.classes)):     self.word_emb.append(Embeddings(self.n_tokens[i],
        self.emb_sizes[i])) self.word_emb = nn.ModuleList(self.word_emb)

        #linear layer to merge embeddings from different token types self.in_linear =
        nn.Linear(np.sum(self.emb_sizes), self.d_model)
        """
        self.in_linear = nn.Linear(len(input_features), self.d_model)

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        # convert input_ids into embeddings and merge them through linear layer
        emb_linear = self.in_linear(input_ids)
        # feed to bert
        y = self.bert(
            inputs_embeds=emb_linear,
            attention_mask=attn_mask,
            output_hidden_states=output_hidden_states,
        )
        return y

    """ For training a tranditional Bert model
    def get_rand_tok(self):
        vel_rand = random.choice(range(self.n_tokens[1]))
        return vel_rand
    """


class ExpressionBertLM(nn.Module):
    def __init__(self, bert: ExpressionBert):
        super().__init__()

        if bert.style_embed_integration_type == "add":
            self.style_proj = torch.nn.Linear(bert.hidden_size, bert.hidden_size)
        else:
            self.style_proj = torch.nn.Linear(
                bert.hidden_size + bert.style_emb_size, bert.hidden_size
            )

        # proj: project embeddings to logits for prediction
        self.proj = []
        self.feature_activation = []
        self.bert = bert
        hidden_size = bert.hidden_size

        for i in range(len(bert.output_features)):
            self.proj.append(nn.Linear(hidden_size, 1))
            feature = self.bert.output_features[i]
            # self.feature_activation.append(ScaledSigmoid(self.bert.feature_boundaries[feature][1]))
            self.feature_activation.append(
                FeatureClip(
                    self.bert.feature_boundaries[feature][0],
                    self.bert.feature_boundaries[feature][1],
                )
            )

        self.proj = nn.ModuleList(self.proj)
        self.out_linear = nn.Linear(hidden_size, hidden_size)

        self.style_emb = bert.style_emb
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask, style):
        # feed to bert
        y = self.bert(x, mask)
        y = y.hidden_states[-1]

        style_emb = self.style_emb(style.view(-1))
        y = y + style_emb.unsqueeze(1)
        y = self.out_linear(y)
        # if self.bert.style_embed_integration_type == "concat":
        #     style_emb = style_emb.unsqueeze(0)
        #     style_emb = F.normalize(style_emb).unsqueeze(1).expand(-1, y.size(1), -1)
        #     y = self.style_proj(torch.cat([y, style_emb], dim=-1))
        # else:
        #     style_emb = style_emb.unsqueeze(0)
        #     style_emb = self.style_proj(F.normalize(style_emb))
        #     y = y + style_emb.unsqueeze(1)

        ys = []
        for i in range(len(self.bert.output_features)):
            yf = self.proj[i](y)
            # ys.append(self.feature_activation[i](yf))
            ys.append(yf)

            ####### Original Activation Function #######
            # ys.append(self.feature_activation(yf,
            #                                   self.bert.feature_boundaries[feature][0],
            #                                   self.bert.feature_boundaries[feature][1]))

        return ys


if __name__ == "__main__":
    bert = ExpressionBert()
    _ = ExpressionBertLM(bert)
