
import copy
import math
import os
import warnings
from typing import Optional, Tuple, Union
from torch.nn import functional as F

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from models.model_output import PromptT5Output
# from models.model_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import *

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_CHECKPOINT_FOR_DOC = "t5-small"


class T5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.d_model = config.d_model
        # self.label_embeds = None
        # self.Mask_modality = None
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        # self.decoder_query = nn.Linear(config.d_model,config.d_kv,bias = False)
        # self.decoder_key = nn.Linear(config.d_model,config.d_kv,bias = False)
        # self.decoder_value = nn.Linear(config.d_model,config.d_model,bias = False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

  
    def forward(
        self,
        Label_att = None,
        Mask_modality = None,
        label_embeds = None,
        css_input = None,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
        # if Mask_modality is not None:
        #     self.Mask_modality = Mask_modality
        # if  self.Mask_modality:
        # if label_embeds is not None:
        #     # k = self.decoder_key(F.normalize(hidden_states[:,:-1,:]))
        #     # q = self.decoder_query(F.normalize(label_embeds))
        #     k = self.decoder_key(self.layer_norm(hidden_states[:,:-1,:]))
        #     q = self.decoder_query(self.layer_norm(label_embeds[:,8:,:]))

        #     cross_att_score = torch.bmm(k, q.transpose(-2, -1)) # [b, t, l]
        #     cross_att_score = F.max_pool2d(cross_att_score,kernel_size = (1,cross_att_score.shape[-1])).squeeze(1)  # [b, t, 1]
        #     cross_att_score = torch.softmax(cross_att_score, dim=1)  # [b, t, 1]
        #     cross_att_score = torch.cat((cross_att_score,torch.zeros(cross_att_score.shape[0],1,1).to(cross_att_score.device)),1)
        #     # v = self.decoder_value( self.layer_norm(hidden_states))
        #     hidden_states = self.dropout(hidden_states *cross_att_score) + hidden_states 
        #     # print("decoder score: ",cross_att_score.squeeze())
        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

T5_START_DOCSTRING = r"""
   
"""
@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class EHRGeneration(T5ForConditionalGeneration):
    def __init__(self,  config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder_query = nn.Linear(config.d_model,config.d_kv,bias = False)
        self.encoder_key = nn.Linear(config.d_model,config.d_kv,bias = False)
        self.encoder_value = nn.Linear(config.d_model,config.d_model,bias = False)
        self.parent_clss = nn.Linear(config.d_model,5)
        # self.fusion_layer = nn.Linear(config.d_model,5)

        self.sigmoid = nn.Sigmoid()

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.Label_att = False
        self.label_embeds = None
        self.css_input = None
    def forward(
        self,
        Label_att:Optional = None,
        Mask_modality:Optional = None,
        label_embeds:Optional[torch.FloatTensor] = None,
        css_input:Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration
        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        # inputs_embeds,label_input,css_input,Mask_modality,Label_att = inputs_embeds[0],inputs_embeds[1],inputs_embeds[2],inputs_embeds[3],inputs_embeds[4]
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                Mask_modality= Mask_modality,
                label_embeds = label_embeds,
                Label_att = Label_att,
                css_input = css_input,
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        # encoder_att_hidden_embedding = None

        tmp_label_embeds = None

        if label_embeds is not None:
            self.label_embeds = label_embeds
            tmp_label_embeds = label_embeds
        else:
            tmp_label_embeds = self.label_embeds

        tmp_css_embeds = None
        css_matrix = None
        cross_att_score = torch.zeros(hidden_states.shape[0],hidden_states.shape[1],1).to(hidden_states.device)
        if css_input is not None:
            self.css_input = css_input
            tmp_css_embeds = css_input
        else:
            tmp_css_embeds = self.css_input

        tmp_css_embeds = tmp_css_embeds[:1,:,:].repeat(hidden_states.shape[0],1,1)
        if Label_att is not None:
            self.Label_att = Label_att
        if self.Label_att:
            k = self.encoder_key(self.layer_norm(hidden_states[:,:-1,:]))
            q = self.encoder_query(self.layer_norm(tmp_label_embeds[:1,8:,:])).repeat(k.shape[0],1,1)
            # k = self.encoder_key(F.normalize(hidden_states[:,:-1,:]))
            # q = self.encoder_query(F.normalize(tmp_label_embeds))
            cross_att_score = torch.bmm(k, q.transpose(-2, -1)) # [b, t, l]
    
            cross_att_score = torch.cat((cross_att_score,torch.zeros(cross_att_score.shape[0],1,q.shape[1]).to(cross_att_score.device)),1)
            # print(cross_att_score.shape,hidden_states.shape)
            # print( torch.bmm(cross_att_score,hidden_states).shape)

            cross_att_score = F.avg_pool2d(cross_att_score,kernel_size = (1,cross_att_score.shape[-1])).squeeze(1)  # [b, t, 1]

            cross_att_score = torch.softmax(cross_att_score, dim=1)  # [b, t, 1]

            # print(cross_att_score.shape)
            v = self.encoder_value(self.layer_norm(hidden_states))

            # v = hidden_states
            att_hidden_states = self.dropout(v *cross_att_score) + v 
            css_matrix = self.sigmoid(self.parent_clss(att_hidden_states.mean(1)))
            att_css_embeds = self.layer_norm(torch.cat([tmp_css_embeds[:,:8,:],css_matrix.unsqueeze(-1)*tmp_css_embeds[:,8:,:] + tmp_css_embeds[:,8:,:]],1))
            att_css_embeds = att_css_embeds[:1,:,:].repeat(k.shape[0],1,1)
            # print(att_hidden_states.shape)
            # hidden_states = torch.cat([hidden_states,att_css_embeds],axis = 1)
            # print(att_hidden_states.shape,hidden_states.shape,att_css_embeds.shape)
            # hidden_states = self.layer_norm(torch.cat([att_hidden_states,att_css_embeds],axis = 1))
            hidden_states = self.layer_norm(torch.cat([att_hidden_states,hidden_states,att_css_embeds],axis = 1))

            # print(hidden_states.shape)
            attention_mask = torch.ones(hidden_states.shape[0],hidden_states.shape[1]).to(att_css_embeds.device)

        else:
            label_embeds = None
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # print("input output shape",inputs_embeds.shape,decoder_input_ids.shape)

        # Decode
        # print("attention_mask",hidden_states.shape)
        decoder_outputs = self.decoder(
            Mask_modality= False,
            label_embeds = None,
            Label_att = False,
            css_input = None,
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        return PromptT5Output(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=cross_att_score,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            css_matrix=css_matrix,
        )
    def prepare_inputs_for_generation(
        self,
        input_ids,
        Label_att = None,
        Mask_modality = None,
        label_embeds = None,
        css_input = None,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "Label_att": Label_att,
            "Mask_modality": Mask_modality,
            "label_embeds": label_embeds,
            "css_input": css_input,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

   