
import types
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModelOutputWithPast,
    BaseModelOutputWithPast
)
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache, DynamicCache

import matplotlib.pyplot as plt
import os

ATT_LAYER = 22
# ATT_LAYER = 40
def apply_qwen3vl_attention_hooks(model):
    """应用Qwen3VL模型的attention输出hooks"""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLModel,
        Qwen3VLTextModel,
        Qwen3VLTextDecoderLayer,
        Qwen3VLTextAttention
    )
    
    # 保存原始的 forward 方法
    if not hasattr(model, '_original_forwards'):
        model._original_forwards = {
            'conditional_generation_forward': model.forward,
            'model_forward': model.model.forward,
            'text_model_forward': model.model.language_model.forward,
            'layer_forwards': [],

            'prepare_inputs_for_generation': model.prepare_inputs_for_generation,
            '_update_model_kwargs_for_generation': model._update_model_kwargs_for_generation
        }
        
        # 保存每个 decoder layer 和 attention 的原始 forward 方法
        for i, layer in enumerate(model.model.language_model.layers):
            model._original_forwards['layer_forwards'].append(layer.forward)

    
    # 替换forward方法
    model.forward = types.MethodType(forward_qwen3vl_conditional_generation, model)
    model.model.forward = types.MethodType(forward_qwen3vl_model, model.model)
    model.model.language_model.forward = types.MethodType(forward_qwen3vl_text_model, model.model.language_model)

    model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation_new, model)
    model._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation_new, model)
    
    # 替换所有decoder层的forward方法
    # for layer in model.model.language_model.layers:
    for i in range(len(model.model.language_model.layers)):
        model.model.language_model.layers[i].forward = types.MethodType(forward_qwen3vl_text_decoder_layer, model.model.language_model.layers[i])
        # model.model.language_model.layers[i].self_attn.config._attn_implementation = "eager"


def restore_original_forwards_qwen3vl(model):
    """恢复Qwen3VL模型的原始forward方法"""
    if hasattr(model, '_original_forwards'):
        # 恢复模型级别的 forward
        model.forward = model._original_forwards['conditional_generation_forward']
        model.model.forward = model._original_forwards['model_forward']
        model.model.language_model.forward = model._original_forwards['text_model_forward']
        
        # 恢复每个 decoder layer 和 attention 的 forward
        # for i, layer in enumerate(model.model.language_model.layers):
        for i in range(len(model.model.language_model.layers)):
            model.model.language_model.layers[i].forward = model._original_forwards['layer_forwards'][i]
        model.model.language_model.layers[i].self_attn.config._attn_implementation = "sdpa"

        model.prepare_inputs_for_generation = model._original_forwards['prepare_inputs_for_generation']
        model._update_model_kwargs_for_generation = model._original_forwards['_update_model_kwargs_for_generation']
        

def forward_qwen3vl_text_decoder_layer(self, *args, **kwargs):
    """
    Qwen3VLTextDecoderLayer的forward钩子函数，支持输出attention权重
    """
    # 提取参数
    hidden_states = args[0] if args else kwargs.get('hidden_states')
    position_embeddings = args[1] if len(args) > 1 else kwargs.get('position_embeddings')
    attention_mask = args[2] if len(args) > 2 else kwargs.get('attention_mask')
    position_ids = args[3] if len(args) > 3 else kwargs.get('position_ids')
    past_key_values = args[4] if len(args) > 4 else kwargs.get('past_key_values')
    use_cache = kwargs.get('use_cache', False)
    cache_position = kwargs.get('cache_position')
    output_attentions = kwargs.get('output_attentions', False)
    
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        
    # Self Attention
    hidden_states_out, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        # output_attentions=False,
    )
    
    self_attn_weights = None
    if output_attentions:
        with torch.no_grad():
            # att_map = _[0].mean(dim=0)[-1,4:2918].reshape(47,62).cpu().to(torch.int).numpy().tolist()
            # att_map = _[0].mean(dim=0)[2925,4:2918]
            # att_map = att_map/(att_map.max()-att_map.min())
            # att_map = att_map.reshape(47,62).cpu().to(torch.float).numpy()
            # plt.figure(figsize=(8, 8))
            # plt.imshow(att_map, cmap='hot', interpolation='nearest')  #'magma'
            # plt.colorbar(label='Attention Value')
            # plt.axis('off')
            # plt.savefig("attention_heatmap.jpg", bbox_inches='tight', pad_inches=0.1, dpi=150)
            # plt.close()  # 关闭图形以释放内存
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
            
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.self_attn.head_dim)
            
            q = self.self_attn.q_norm(self.self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            k = self.self_attn.k_norm(self.self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            
            cos, sin = position_embeddings
            # apply rope
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
            k = repeat_kv(k, self.self_attn.num_key_value_groups)
            self_attn_weights = (q, k)
            
    hidden_states = residual + hidden_states_out

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs


def forward_qwen3vl_text_model(self, *args, **kwargs):
    """
    Qwen3VLTextModel的forward钩子函数，支持输出attention权重
    """
    # 提取参数
    input_ids = kwargs.get('input_ids')
    attention_mask = kwargs.get('attention_mask')
    position_ids = kwargs.get('position_ids')
    past_key_values = kwargs.get('past_key_values')
    inputs_embeds = kwargs.get('inputs_embeds')
    use_cache = kwargs.get('use_cache')
    cache_position = kwargs.get('cache_position')
    visual_pos_masks = kwargs.get('visual_pos_masks')
    deepstack_visual_embeds = kwargs.get('deepstack_visual_embeds')
    output_attentions = kwargs.get('output_attentions')

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    from transformers.masking_utils import create_causal_mask
    attention_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_self_attns = () if output_attentions else None

    # print("layer:", len(self.layers))
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_attentions and layer_idx == ATT_LAYER:
        # if output_attentions:  #jsm
            output_attentions_cur = True
            # output_attentions_cur = layer_idx
        else:
            output_attentions_cur = False
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions_cur,
            # **kwargs,
        )
        hidden_states = layer_outputs[0]

        if output_attentions_cur:
            all_self_attns += (layer_outputs[1],)
            
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                attentions=all_self_attns,
            )  #jsm
        # add visual features to the hidden states of first several layers
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer

    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        attentions=all_self_attns,
    )


def forward_qwen3vl_model(self, *args, **kwargs):
    """
    Qwen3VLModel的forward钩子函数，支持输出attention权重
    """
    # 提取参数
    input_ids = kwargs.get('input_ids')
    attention_mask = kwargs.get('attention_mask')
    position_ids = kwargs.get('position_ids')
    past_key_values = kwargs.get('past_key_values')
    inputs_embeds = kwargs.get('inputs_embeds')
    pixel_values = kwargs.get('pixel_values')
    pixel_values_videos = kwargs.get('pixel_values_videos')
    image_grid_thw = kwargs.get('image_grid_thw')
    video_grid_thw = kwargs.get('video_grid_thw')
    cache_position = kwargs.get('cache_position')
    output_attentions = kwargs.get('output_attentions')
    attention_mask_llm = kwargs.get('attention_mask_llm')

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    from transformers.utils import is_torchdynamo_compiling
    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        # Calculate RoPE index once per generation in the pre-fill stage only.
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    if attention_mask_llm is not None:  #jsm
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask_llm,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            output_attentions=output_attentions,
            # **kwargs,
        )
    else:  
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            output_attentions=output_attentions,
            # **kwargs,
        )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


def forward_qwen3vl_conditional_generation(self, *args, **kwargs):
    """
    Qwen3VLForConditionalGeneration的forward钩子函数，支持输出attention权重
    """
    # 提取参数
    input_ids = kwargs.get('input_ids')
    attention_mask = kwargs.get('attention_mask')
    position_ids = kwargs.get('position_ids')
    past_key_values = kwargs.get('past_key_values')
    inputs_embeds = kwargs.get('inputs_embeds')
    labels = kwargs.get('labels')
    pixel_values = kwargs.get('pixel_values')
    pixel_values_videos = kwargs.get('pixel_values_videos')
    image_grid_thw = kwargs.get('image_grid_thw')
    video_grid_thw = kwargs.get('video_grid_thw')
    cache_position = kwargs.get('cache_position')
    logits_to_keep = kwargs.get('logits_to_keep', 0)
    output_attentions = kwargs.get('output_attentions')
    attention_mask_llm = kwargs.get('attention_mask_llm')

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        output_attentions=output_attentions,
        attention_mask_llm=attention_mask_llm
        # **kwargs,
    )

    hidden_states = outputs[0]

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

    return Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )


from transformers import Qwen3VLForConditionalGeneration

# 保存原始方法，以便在 wrapper 中调用
_original_prepare_inputs_for_generation = Qwen3VLForConditionalGeneration.prepare_inputs_for_generation
_original_update_model_kwargs_for_generation = Qwen3VLForConditionalGeneration._update_model_kwargs_for_generation

def prepare_inputs_for_generation_new(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask_llm=None,  # 新增参数
    **kwargs,
):
    # 如果 attention_mask_llm 在 kwargs 中，提取出来
    if attention_mask_llm is None:
        attention_mask_llm = kwargs.get("attention_mask_llm", None)

    # 调用原始方法
    model_inputs = _original_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        position_ids=position_ids,
        use_cache=use_cache,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        **kwargs,
    )

    # 确保 attention_mask_llm 存在于 model_inputs 中
    if attention_mask_llm is not None:
        model_inputs["attention_mask_llm"] = attention_mask_llm

    return model_inputs

def _update_model_kwargs_for_generation_new(
    self,
    outputs,
    model_kwargs,
    is_encoder_decoder=False,
    # standardize_cache_format=False,
    **kwargs # 接收额外参数以防签名变化
):
    # 调用原始方法更新常规参数
    model_kwargs = _original_update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder=is_encoder_decoder,
        # standardize_cache_format=standardize_cache_format,
        **kwargs
    )

    # 更新 attention_mask_llm
    if "attention_mask_llm" in model_kwargs:
        attention_mask_llm = model_kwargs["attention_mask_llm"]
        # 假设新生成的 token 对所有内容可见（或者根据你的逻辑填充 0 或 1）
        # 这里我们追加 1，形状为 (batch_size, 1)
        new_mask = torch.ones(
            (attention_mask_llm.shape[0], 1),
            dtype=attention_mask_llm.dtype,
            device=attention_mask_llm.device
        )
        model_kwargs["attention_mask_llm"] = torch.cat([attention_mask_llm, new_mask], dim=-1)

    return model_kwargs
