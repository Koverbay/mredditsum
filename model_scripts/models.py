import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import os
import sys
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import argparse

from transformers import ViTImageProcessor, ViTModel

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
    BartModel,
    BartDecoder,
    BartEncoder,
    BartAttention,
    shift_tokens_right,
    _expand_mask,
    LearnedPositionalEmbedding,
    invert_mask,
    fill_with_neg_inf
)


from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
    BaseModelOutput,
)

from transformers.models.lxmert.modeling_lxmert import LxmertLayer, LxmertXLayer

@dataclass
class ViTBARTModelOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    visual_last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    visual_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class ViTBARTSeq2SeqModelOutput(Seq2SeqModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    visual_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    visual_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class ViTBARTSeq2SeqLMOutput(Seq2SeqLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    visual_logits: Optional[torch.FloatTensor] = None
    visual_loss: Optional[torch.FloatTensor] = None

class ViTBartEncoder(BartEncoder):
    def __init__(self, config):
        super().__init__(config)
        
        self.visn_positions = nn.Embedding(config.max_images, config.d_model)
        self.visn_projection = nn.Linear(config.visual_feat_dim, config.d_model)

        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        visn_features=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            visn_features = self.embed_visn(visn_features)
            inputs_embeds = torch.cat([inputs_embeds, visn_features], dim=1)
            input_ids = None
        return super().forward(input_ids, attention_mask,  head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    
    def embed_visn(self,visn_features):
        m, n, dim = visn_features.size()
        visn_features = visn_features.view(-1, dim)
        visn_features = self.visn_projection(visn_features)
        visn_features = visn_features.view(m, n, -1)
        
        visn_pos = self.visn_positions(torch.arange(n, dtype=torch.long, device=visn_features.device))
        return visn_features + visn_pos


class ViTBartModel(BartModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ViTBartEncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        visn_features=None,
    ):

        # Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                visn_features=visn_features,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return ViTBARTModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_attentions=encoder_outputs.attentions,
            visual_encoder_last_hidden_state=None,
            visual_encoder_attentions=None,
        )

class VitBARTforConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = VLBartModel(config)

        self.visual_dense = nn.Linear(config.d_model, config.d_model)
        self.visual_dropout = nn.Dropout(config.classifier_dropout)
        self.visual_head = nn.Linear(config.d_model, 2)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        visn_features=None,
        image_label=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            visn_features=visn_features,
        )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        visual_loss = None
        visual_logits=None

        visual_logits = outputs.encoder_last_hidden_state[:,-self.config.max_images:,:]
        visual_logits = self.visual_dropout(visual_logits)
        visual_logits = self.visual_dense(visual_logits)
        visual_logits = torch.tanh(visual_logits)
        visual_logits = self.visual_dropout(visual_logits)
        visual_logits = self.visual_head(visual_logits).reshape(-1,self.config.max_images,2)
        
        if image_label is not None:
   
            loss_fct = CrossEntropyLoss()
            visual_loss = loss_fct(visual_logits.reshape(-1, 2), image_label.view(-1))
        return XMSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            visual_loss=visual_loss,
            visual_logits=visual_logits,
        )

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BartZero(nn.Module):
    def __init__(self, bart_config, torch_device):
        # Creating a model
        # Suppose to be the same model in Attention is All You Need
        # positional_encoder = PositionalEncoding(d_model=512)
        # model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
        #             dim_feedforward=2048, dropout=0.1,
        #             custom_encoder=None, custom_decoder=None)
        self.bart_config = BartConfig(
            activation_dropout=0.0, activation_function='gelu', vocab_size=50265,
            d_model=512, encoder_ffn_dim=2048, encoder_layers=6, encoder_attention_heads=8,
            decoder_ffn_dim=2048, decoder_layers=6, decoder_attention_heads=8,
            encoder_layerdrop=0.0, decoder_layerdrop=0.0, attention_dropout=0.0,
            dropout=0.1, max_position_embeddings=1024*2, init_std=0.02, classifier_dropout=0.0,
            num_labels=3, is_encoder_decoder=True, pad_token_id=1, bos_token_id=0, eos_token_id=2,
            normalize_before=False, add_final_layer_norm=False, scale_embedding=False, normalize_embedding=True,
            static_position_embeddings=False, add_bias_logits=False
        )

        self.bart = BartForConditionalGeneration(bart_config)

    def forward(self,input_ids,attention_mask,decoder_input_ids,decoder_attention_mask):
        x = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=target_ids,
            decoder_attention_mask=target_attention_mask,
        )
        return x

class BartAlpha(BartForConditionalGeneration):
    """
    BartForConditionalGeneration
    with a new LM head - note that bart-large-cnn shares the embedding three times:
        1) encoder
        2) decoder
        3) LM header
    but since fine-tuning this embedding would require backpropagation to first layer
    it's not memomry efficient, so we decided to add a new layer
    """
    # def __init__(self, model_name):
    #     super().__init__()
    #     self.bart = BartForConditionalGeneration.from_pretrained(model_name)
    #     self.decoder_lm_head = nn.Linear(self.bart.config.d_model, self.bart.config.vocab_size, bias=True)
    #     self.decoder_lm_head.weight.data = self.bart.model.shared.weight.data
    #     nn.init.zeros_(self.decoder_lm_head.bias)

    def __init__(self, config: BartConfig):
        super().__init__(config)

    def add_new_lm_head(self):
        self.decoder_lm_head = nn.Linear(self.model.config.d_model, self.model.config.vocab_size, bias=True)
        self.decoder_lm_head.weight.data = self.model.shared.weight.data
        nn.init.zeros_(self.decoder_lm_head.bias)

    # override
    def get_output_embeddings(self):
        if hasattr(self, 'decoder_lm_head'):
            return self.decoder_lm_head
        else:
            # self.add_new_lm_lead()
            # return self.decoder_lm_head
            print("decoder_lm_head has not been added")
            return None

    # override
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
        decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
        labels=None, use_cache=False, **unused):

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = self.decoder_lm_head(outputs[0])
        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        return outputs

    def expand_learned_embed_positions(self, multiple=4, cut=0):
        if multiple != 2 and multiple != 4:
            raise ValueError("only multiple = 2,4 supported")

        new_embed_positions_size = 1026 * multiple - cut # original is 1024+2
        new_enc_embed_positions = LearnedPositionalEmbedding(new_embed_positions_size, self.model.config.hidden_size, self.model.config.pad_token_id)
        new_enc_embed_positions.weight.data[:1026] = self.model.encoder.embed_positions.weight.data
        new_enc_embed_positions.weight.data[1026:1026*2] = torch.flip(self.model.encoder.embed_positions.weight.data, dims=[0])
        if multiple == 4:
            new_enc_embed_positions.weight.data[1026*2:1026*3] = self.model.encoder.embed_positions.weight.data
            new_enc_embed_positions.weight.data[1026*3:1026*4-cut] = torch.flip(self.model.encoder.embed_positions.weight.data, dims=[0])[:-cut]
        self.model.encoder.embed_positions = new_enc_embed_positions

        new_dec_embed_positions = LearnedPositionalEmbedding(new_embed_positions_size, self.model.config.hidden_size, self.model.config.pad_token_id)
        new_dec_embed_positions.weight.data[:1026] = self.model.decoder.embed_positions.weight.data
        new_dec_embed_positions.weight.data[1026:1026*2] = torch.flip(self.model.decoder.embed_positions.weight.data, dims=[0])
        if multiple == 4:
            new_dec_embed_positions.weight.data[1026*2:1026*3] = self.model.decoder.embed_positions.weight.data
            new_dec_embed_positions.weight.data[1026*3:1026*4-cut] = torch.flip(self.model.decoder.embed_positions.weight.data, dims=[0])[:-cut]
        self.model.decoder.embed_positions = new_dec_embed_positions
        self.config.max_position_embeddings = new_embed_positions_size

        print("expanded learned_embed_positions to {} tokens".format(self.model.config.max_position_embeddings))

    def freeze_exclude_k_layers(self, k=1):
        for param in self.model.parameters(): param.requires_grad = False

        # for param in self.model.shared.parameters(): param.requires_grad = True
        # for param in self.model.encoder.embed_positions.parameters(): param.requires_grad = True
        # for param in self.model.encoder.layernorm_embedding.parameters(): param.requires_grad = True
        for _k in range(k):
            for param in self.model.encoder.layers[-(_k+1)].parameters(): param.requires_grad = True

        # for param in self.model.decoder.embed_positions.parameters(): param.requires_grad = True
        # for param in self.model.decoder.layernorm_embedding.parameters(): param.requires_grad = True
        for _k in range(k):
            for param in self.model.decoder.layers[-(_k+1)].parameters(): param.requires_grad = True

        print("freeze excluding top {} layer(s)".format(k))

class BartBeta(BartAlpha):
    def __init__(self, config: BartConfig):
        super().__init__(config)

    def pooling_layers(self):
        pass

    # override
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
        decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
        labels=None, use_cache=False, **unused):

        output_attentions    = False
        output_hidden_states = False

        # make masks if user doesn't supply
        decoder_input_ids, decoder_padding_mask, causal_mask = prepare_bart_decoder_inputs(
            self.config,
            input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_attention_mask,
            causal_mask_dtype=self.model.shared.weight.dtype,
        )

        if encoder_outputs is None:
            # encoder_outputs = self.model.encoder(
            encoder_outputs, reduced_attention_mask = self.forward_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        assert isinstance(encoder_outputs, tuple)

        if reduced_attention_mask is not None:
            attention_mask = (~reduced_attention_mask).long()

        decoder_outputs = self.model.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_cached_states=decoder_cached_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs: Tuple = _filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0], torch.Tensor)
        encoder_outputs: Tuple = _filter_out_falsey_values(encoder_outputs)
        # return decoder_outputs + encoder_outputs
        outputs = decoder_outputs + encoder_outputs

        # lm_logits = self.decoder_lm_head(outputs[0])
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        return outputs

    def forward_encoder(self, input_ids, attention_mask=None,
                    output_attentions=False, output_hidden_states=False):

        # Why do we need to invert?? ANS: the EncoderLayer defined in huggingface is designed this way...
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.model.encoder.embed_tokens(input_ids) * self.model.encoder.embed_scale
        embed_pos     = self.model.encoder.embed_positions(input_ids)

        x = inputs_embeds + embed_pos
        x = self.model.encoder.layernorm_embedding(x)
        x = F.dropout(x, p=self.model.encoder.dropout, training=self.model.encoder.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for layer_i, encoder_layer in enumerate(self.model.encoder.layers):
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.model.encoder.training and (dropout_probability < self.model.encoder.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask)

                if (layer_i+1) % 2 == 9999: # there are 12 layers
                    # ------------------------------- Pyramidal Style ------------------------------- #
                    stride = 2
                    len_0 = x.size(0)
                    len_1 = int(len_0/stride)

                    x_1 = torch.zeros((len_1, x.size(1), x.size(2)), dtype=x.dtype)
                    attention_mask_1 = torch.zeros((attention_mask.size(0),len_1), dtype=attention_mask.dtype)

                    for i in range(len_1):
                        x_1[i,:,:] = (x[i*2,:,:] + x[(i*2)+1,:,:]) / 2
                        attention_mask_1[:,i] = ~(~attention_mask[:,i*2] + ~attention_mask[:,(i*2)+1])

                    x = x_1.to(x.device)
                    attention_mask = attention_mask_1.to(attention_mask.device)
                    # -------------------------------------------------------------------------------- #
            if output_attentions:
                all_attentions.append(attn)

        if self.model.encoder.layer_norm:
            x = self.model.encoder.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)

        return (x, encoder_states, all_attentions), attention_mask

def prepare_bart_decoder_inputs(
    config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask

def _filter_out_falsey_values(tup):
    """Remove entries that are None or [] from an iterable."""
    return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)