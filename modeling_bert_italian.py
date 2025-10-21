"""
Modello BERT Italiano Personalizzato per BertViz Neuron View

Questo modulo estende le classi BERT standard di Hugging Face Transformers
per supportare la visualizzazione della Neuron View con modelli italiani.

Le modifiche principali riguardano l'aggiunta di query_layer e key_layer
agli output del modello per permettere la visualizzazione delle attenzioni.
"""

import math
import torch
from torch import nn
from typing import Optional, Tuple, Dict, Any
from transformers.models.bert import modeling_bert as bert_modeling
from transformers import BertModel as HFBertModel, BertConfig


class BertSelfAttentionIT(bert_modeling.BertSelfAttention):
    """
    Versione personalizzata di BertSelfAttention che restituisce query e key layers
    per la visualizzazione con BertViz.
    """

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Riorganizza il tensore per il calcolo multi-head attention.

        Args:
            x: Tensore di input [batch_size, seq_length, hidden_size]

        Returns:
            Tensore riorganizzato [batch_size, num_heads, seq_length, head_size]
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # MODIFICA PRINCIPALE per BertViz: restituiamo un dizionario con attn, queries e keys
        if output_attentions:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer,
            }
            outputs = (context_layer, attn_data)
        else:
            outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)

        return outputs


class BertAttentionIT(bert_modeling.BertAttention):
    """
    Versione personalizzata di BertAttention che utilizza BertSelfAttentionIT.
    """

    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        # Sostituiamo la self-attention standard con la nostra versione personalizzata
        self.self = BertSelfAttentionIT(
            config, position_embedding_type=position_embedding_type
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        past_key_values=None,  # Compatibilità con versioni recenti
        **kwargs,  # Cattura eventuali altri parametri
    ) -> Tuple[torch.Tensor]:
        # Se past_key_values è passato invece di past_key_value, usa quello
        if past_key_values is not None and past_key_value is None:
            past_key_value = past_key_values

        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)

        # Propaga tutti gli output (inclusi query e key se presenti)
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class BertLayerIT(bert_modeling.BertLayer):
    """
    Versione personalizzata di BertLayer che utilizza BertAttentionIT.
    """

    def __init__(self, config):
        super().__init__(config)
        # Sostituiamo l'attention standard con la nostra versione personalizzata
        self.attention = BertAttentionIT(config)
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = BertAttentionIT(
                config, position_embedding_type="absolute"
            )


class BertEncoderIT(bert_modeling.BertEncoder):
    """
    Versione personalizzata di BertEncoder che utilizza BertLayerIT.
    """

    def __init__(self, config):
        super().__init__(config)
        # Sostituiamo i layer standard con la nostra versione personalizzata
        self.layer = nn.ModuleList(
            [BertLayerIT(config) for _ in range(config.num_hidden_layers)]
        )


class BertModelIT(HFBertModel):
    """
    Modello BERT italiano personalizzato per BertViz.

    Questo modello estende BertModel di Hugging Face per restituire
    query_layer e key_layer necessari per la visualizzazione della Neuron View.

    Esempio d'uso:
        from transformers import AutoTokenizer
        from bertviz.neuron_view import show

        model_name = "dbmdz/bert-base-italian-xxl-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertModelIT.from_pretrained(model_name, output_attentions=True)

        sentence_a = "Il gatto si è seduto sul tappeto."
        sentence_b = "Il cane dormiva sulla poltrona."

        show(model, "bert", tokenizer, sentence_a, sentence_b, layer=2, head=8)
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # Sostituiamo l'encoder standard con la nostra versione personalizzata
        self.encoder = BertEncoderIT(config)

        # Reinizializziamo i pesi
        self.post_init()
