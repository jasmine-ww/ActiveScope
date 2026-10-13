
import types
import os
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModelOutputWithPast,
    BaseModelOutputWithPast
)
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging as hf_logging
import numpy as np
from utils import *

import matplotlib.pyplot as plt
import math
from pathlib import Path

ATT_LAYER = 29   #22
DEBUG_HEATMAP_DIR = Path(__file__).resolve().parents[1] / "outputs" / "debug" / "attention_heatmaps"
hf_logging.set_verbosity_error()

import importlib



def prepare_internvl_input(pixel_values=None, question="", model=None, tokenizer=None, IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',verbose=False, history=None, num_patches_list=None, output_eos=False):

    base_module = model.__class__.__module__.rsplit(".", 1)[0]
    conversation_module = importlib.import_module(f"{base_module}.conversation")

    get_conv_template = conversation_module.get_conv_template
    # conv = get_conv_template("default")

    if history is None and pixel_values is not None and '<image>' not in question:
        question = '<image>\n' + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    history = [] if history is None else history
    for (old_question, old_answer) in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f'dynamic ViT batch size: {image_bs}')

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors='pt')

    if output_eos:
        return model_inputs, eos_token_id
    return model_inputs

import re
def extract_targets(sentence: str):
    # 尝试匹配 "objects:" 及其后的内容
    match = re.search(r"objects:\s*(.*)", sentence, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 如果没有匹配到 "objects:"，尝试用冒号分割，取最后一部分
    if ':' in sentence:
        return sentence.split(':')[-1].strip()
        
    # 如果以上都不符合，假设整个句子就是对象列表
    return sentence.strip()
def split_targets_sentence(targets_sentence:str, split_tag = r' and |, '):
    if targets_sentence.endswith('.'):
        targets_sentence = targets_sentence[:-1]
    targets = re.split(split_tag, targets_sentence)
    return [t.strip() for t in targets if t.strip()]

def generate_visual_cues_using_ic(question: str, model, processor, split_tag = r' and |, '):
    ic_examples = {
        "question_template": "Question: {}\nIf you want to answer the question, which objects' information do you need?",
        "question_list": [
            "What is the color of the boy's bag?",
            "Is the yellow car on the left or right side of the white car?",
            "Tell me the number on the black board.",
            "Is the girl with pink hair on the left or right side of the man with backpack?",
            "What kind of animal is on the red sign?",
            "From the information on that advertising board, what is the type of this shop?"
        ],
        "response_list": [
            "To answer the question, I need know the location of the boy with a bag so that I can determine the color of the bag. So I need the information about the following objects: boy with a bag.",
            "To answer the question, I need know the location of the yellow car and the white car so that I can determine the positional relationship between the two of them. So I need the information about the following objects: white car and yellow car.",
            "To answer the question, I need know the location of the black board so that I can determine the number on it. So I need the information about the following objects: black board",
            "To answer the question, I need know the location of the girl with pink hair and the man with backpack so that I can determine the positional relationship between the two of them. So I need the information about the following objects: girl with pink hair and man with backpack.",
            "To answer the question, I need know the location of the red sign so that I can determine the kind of animal on it. So I need the information about the following objects: red sign.",
            "To answer the question, I need know the location of the advertising board so that I can determine the type of the shop. So I need the information about the following objects: advertising board."
        ]
    }
    ic_question_template = ic_examples["question_template"]
    ic_question_list = ic_examples["question_list"]
    ic_response_list = ic_examples["response_list"]
    
    messages = []
    for q, a in zip(ic_question_list, ic_response_list):
        messages.append({"role": "user", "content": [{"type": "text", "text": ic_question_template.format(q)}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
    messages.append({"role": "user", "content": ic_question_template.format(question)})
    history = []
    for q, a in zip(ic_question_list, ic_response_list):
        history.append((ic_question_template.format(q), a))
    
    model_inputs, eos_token_id = prepare_internvl_input(question=ic_question_template.format(question), model=model, tokenizer=processor, output_eos=True, history=history)

    base_module = model.__class__.__module__.rsplit(".", 1)[0]
    conversation_module = importlib.import_module(f"{base_module}.conversation")
    get_conv_template = conversation_module.get_conv_template
    template = get_conv_template(model.template)

    generation_config = dict(max_new_tokens=1024, do_sample=True)
    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)
    generation_config['eos_token_id'] = eos_token_id
    generation_config['pad_token_id'] = processor.pad_token_id if processor.pad_token_id is not None else eos_token_id
    generation_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config
    )
    response = processor.batch_decode(generation_output, skip_special_tokens=True)[0]
    response = response.split(template.sep.strip())[0].strip()
    
    
    # targets_sentence = extract_targets(outputs)
    outputs = extract_targets(response)
    if outputs is not None:
        targets = split_targets_sentence(outputs, split_tag)
    else:
        targets = []
    
    return targets

def find_target_indices(input_ids, target_text, tokenizer):
    # target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    target_ids = tokenizer(target_text)
    if not target_ids:
        return []
    
    indices = []
    # Search for the sequence
    for i in range(len(input_ids) - len(target_ids) + 1):
        if input_ids[i:i+len(target_ids)] == target_ids:
            indices.extend(range(i, i+len(target_ids)))
            
    # Try with leading space
    target_ids_space = tokenizer.encode(" " + target_text, add_special_tokens=False)
    if target_ids_space and target_ids_space != target_ids:
        for i in range(len(input_ids) - len(target_ids_space) + 1):
            if input_ids[i:i+len(target_ids_space)] == target_ids_space:
                indices.extend(range(i, i+len(target_ids_space)))
                
    return sorted(list(set(indices)))


def verify_target_existence(image, bbox, target, model, tokenizer):
    x1, y1, x2, y2 = map(int, bbox)
    if x2 <= x1 or y2 <= y1:
        return False
    
    cropped_image = image.crop((x1, y1, x2, y2))
    
    prompt = f"Does the {target} exist in this image? Answer Yes or No."
    
    # pixel_values = load_image(image, max_num=12, use_anyres=True).to(torch.bfloat16).cuda()
    pixel_values_crop = load_image(cropped_image, max_num=12, use_anyres=True).to(torch.bfloat16).cuda()

    # pixel_values_multi = torch.cat((pixel_values, pixel_values_crop), dim=0)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    # output_text = model.chat(tokenizer, pixel_values_multi, prompt, generation_config)
    output_text = model.chat(tokenizer, pixel_values_crop, prompt, generation_config)
        
    return "yes" in output_text.lower()
    
def compute_internvl_patch_grid(image_size, pixel_values, input_size=448, max_num=12):
    """
    Infer InternVL anyres patch tiling (tiles_w, tiles_h) and whether a thumbnail exists.
    Follows the same logic in utils.dynamic_preprocess.
    """
    orig_w, orig_h = image_size
    aspect_ratio = orig_w / orig_h

    # Build target ratio candidates exactly as in dynamic_preprocess
    target_ratios = set(
        (i, j)
        for n in range(1, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= 1
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Use the same helper from utils
    tiles_w, tiles_h = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_w, orig_h, input_size
    )

    patches_count = pixel_values.shape[0]
    has_thumbnail = patches_count != 1  # dynamic_preprocess appends thumbnail when blocks != 1

    blocks = tiles_w * tiles_h
    # Sanity fallback if mismatch (rare); try factorization closest to aspect
    expected = blocks + (1 if has_thumbnail else 0)
    if expected != patches_count:
        main_patches = patches_count - (1 if has_thumbnail else 0)
        # find factor pair (a, b) for main_patches closest to aspect_ratio
        best = None
        best_diff = float('inf')
        for a in range(1, int(math.sqrt(main_patches)) + 1):
            if main_patches % a == 0:
                b = main_patches // a
                ratio_diff = abs((a / b) - aspect_ratio)
                if ratio_diff < best_diff:
                    best_diff = ratio_diff
                    best = (a, b)
        if best:
            tiles_w, tiles_h = best

    return tiles_w, tiles_h, has_thumbnail

def att_vector_to_grid(att_vec: np.ndarray, tiles_w: int, tiles_h: int, num_image_token: int):
    """
    Convert a flat attention vector over InternVL visual tokens into a 2D grid
    by tiling per-patch squares (sqrt(num_image_token) x sqrt(num_image_token)).
    The last patch (thumbnail) is ignored.
    """
    s = int(math.sqrt(num_image_token))
    grid_h = tiles_h * s
    grid_w = tiles_w * s

    main_tokens = tiles_w * tiles_h * num_image_token
    att_vec = att_vec[:main_tokens]  # drop thumbnail tokens if present

    grid = np.zeros((grid_h, grid_w), dtype=np.float32)
    for p in range(tiles_h * tiles_w):
        row = p // tiles_w
        col = p % tiles_w
        seg = att_vec[p * num_image_token:(p + 1) * num_image_token].reshape(s, s)
        y0, x0 = row * s, col * s
        grid[y0:y0 + s, x0:x0 + s] = seg
    return grid

def _grid_mask_to_token_mask(grid_mask: np.ndarray, tiles_w: int, tiles_h: int, num_image_token: int, has_thumb: bool, total_patches: int):
    """
    Convert grid mask (tiles_h*s, tiles_w*s) → token mask (length total_patches*num_image_token),
    respecting InternVL’s per-patch ordering and appending 1s for thumbnail tokens.
    """
    s = int(math.isqrt(num_image_token))
    parts = []
    for p in range(tiles_h * tiles_w):
        row, col = divmod(p, tiles_w)
        sub = grid_mask[row * s:(row + 1) * s, col * s:(col + 1) * s]
        parts.append(sub.reshape(-1))
    if has_thumb and total_patches == tiles_w * tiles_h + 1:
        parts.append(np.ones((num_image_token,), dtype=np.float32))
    return np.concatenate(parts, axis=0)

def target_iterate_attention_internvl(image, pixel_values, prompt, general_prompt, model, tokenizer, image_size=None, max_iters=5):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    if "\nA" in prompt:
        prompt=prompt.split('\nA')[0]
    targets = generate_visual_cues_using_ic(prompt, model, tokenizer)
    results = {}
    results_his = {}
    for target in targets:
        results_his[target] = []

    model_inputs = prepare_internvl_input(pixel_values, prompt, model, tokenizer)

    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)
    input_ids_list = input_ids[0].tolist()
    # generation_config['eos_token_id'] = eos_token_id
    
    model_inputs_multi = prepare_internvl_input(pixel_values, general_prompt, model, tokenizer)

    input_ids_multi = model_inputs_multi['input_ids'].to(model.device)
    attention_mask_multi = model_inputs_multi['attention_mask'].to(model.device)

    vision_start_token_id = tokenizer.convert_tokens_to_ids('<img>')
    vision_end_token_id = tokenizer.convert_tokens_to_ids('</img>')

    pos = input_ids.tolist()[0].index(vision_start_token_id) + 1
    pos_end = input_ids.tolist()[0].index(vision_end_token_id)

    # num_vision_tokens = pos_end - pos
    # h = w = int(np.sqrt(num_vision_tokens))
    # att_shape = [h, w]
    # exclusion_mask = np.ones((h, w), dtype=np.float32)

    # Build proper grid based on anyres tiling
    tiles_w, tiles_h, has_thumb = compute_internvl_patch_grid(image_size, pixel_values)
    s = int(math.sqrt(model.num_image_token))
    grid_h, grid_w = tiles_h * s, tiles_w * s
    exclusion_mask = np.ones((grid_h, grid_w), dtype=np.float32)
    
    apply_qwen2_attention_hooks(model)


    for i in range(max_iters):
        if all(t in results for t in targets):
            break
            
        # flat_mask = exclusion_mask.flatten()
        # att_map_mask = torch.from_numpy(flat_mask).to(model.device)

        att_map_mask_vec = _grid_mask_to_token_mask(
            exclusion_mask, tiles_w, tiles_h, model.num_image_token, has_thumb, total_patches=int(pixel_values.shape[0])
        )
        att_map_mask = torch.from_numpy(att_map_mask_vec).to(model.device)
        
        current_inputs = prepare_internvl_input(pixel_values, prompt, model, tokenizer)
        current_inputs_multi = prepare_internvl_input(pixel_values, general_prompt, model, tokenizer)

        input_ids_curr = current_inputs['input_ids'].to(model.device)
        attention_mask_llm = current_inputs['attention_mask'].to(model.device)
        
        input_ids_multi = current_inputs_multi['input_ids'].to(model.device)
        general_mask_llm = current_inputs_multi['attention_mask'].to(model.device)
        
        attention_mask_llm[0, pos:pos_end] = attention_mask_llm[0, pos:pos_end] * att_map_mask.long()
        general_mask_llm[0, pos:pos_end] = general_mask_llm[0, pos:pos_end] * att_map_mask.long()
        
        with torch.no_grad():
            att_matrix = model(
                pixel_values=pixel_values,
                input_ids=input_ids_curr,
                attention_mask=attention_mask_llm,
                output_attentions=True
            ).attentions[ATT_LAYER][0].mean(dim=0)
            #  att_matrix = outputs

            att_matrix_g = model(
                pixel_values=pixel_values,
                input_ids=input_ids_multi,
                attention_mask=general_mask_llm,
                output_attentions=True
            ).attentions[ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
            # ).attentions[ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
            #  att_matrix_g = general_outputs

        found_new = False
        for target in targets:
            if target in results:
                continue
                
            indices = find_target_indices(input_ids_list, target, tokenizer)
            if indices:
                target_att = att_matrix[indices, pos:pos_end].mean(dim=0)
                target_att = target_att.to(torch.float32).detach().cpu().numpy()
                
                # Map to 2D grid (ignore thumbnail tokens)
                target_grid = att_vector_to_grid(target_att, tiles_w, tiles_h, model.num_image_token)
                # # target_att = np.divide(target_att, att_matrix_g, out=np.zeros_like(target_att), where=att_matrix_g != 0)
                # target_att = target_att.reshape(att_shape)
                
                bbox_size = int(image_size[0]*0.2)
                # bbox, mask_region = bbox_from_att_image_adaptive(target_att, image_size, bbox_size=bbox_size, return_mask=True)
                bbox, mask_region = bbox_from_att_image_adaptive(target_grid, image_size, bbox_size=bbox_size, return_mask=True)
                results_his[target].append(bbox)
                
                if verify_target_existence(image, bbox, target, model, tokenizer):
                    results[target] = bbox
                    found_new = True
                else:
                    mask_cur = (mask_region == 0).astype(int)
                    exclusion_mask = exclusion_mask * mask_cur
                    
        if not found_new and i < max_iters - 1:
             pass
        
    # print("results: ", results)
    # print("results_his: ", results_his)
    # 把results_his的信息放到results里
    for target in results_his:
        if target in results:
            results[target] = (results[target], results_his[target])
             
    return results

def rel_attention_internvl(pixel_values, prompt, general_prompt, model, tokenizer, image_path=""):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    model_inputs = prepare_internvl_input(pixel_values, prompt, model, tokenizer)

    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)
    # generation_config['eos_token_id'] = eos_token_id
    
    model_inputs_multi = prepare_internvl_input(pixel_values, general_prompt, model, tokenizer)

    input_ids_multi = model_inputs_multi['input_ids'].to(model.device)
    attention_mask_multi = model_inputs_multi['attention_mask'].to(model.device)


    # att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()

    # vision_start_token_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    # vision_end_token_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    vision_start_token_id = tokenizer.convert_tokens_to_ids('<img>')
    vision_end_token_id = tokenizer.convert_tokens_to_ids('</img>')

    pos = input_ids.tolist()[0].index(vision_start_token_id) + 1
    pos_end = input_ids.tolist()[0].index(vision_end_token_id)
    
    apply_qwen2_attention_hooks(model)
    att = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,

    )['attentions'][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()

    general_att = model(
        pixel_values=pixel_values,
        input_ids=input_ids_multi,
        attention_mask=attention_mask_multi,
        output_attentions=True,
    )['attentions'][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
    
    # att_map = att / general_att
    att_map = np.divide(att, general_att, out=np.zeros_like(att), where=general_att != 0)

    num_vision_tokens = pos_end - pos
    h = w = int(np.sqrt(num_vision_tokens))
    att_shape = [h, w]
    att_map = att_map.reshape(att_shape)

    return att_map

    ###-------------存每一层注意力图----------------

    att = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,

    )['attentions']

    general_att = model(
        pixel_values=pixel_values,
        input_ids=input_ids_multi,
        attention_mask=attention_mask_multi,
        output_attentions=True,
    )['attentions']

    num_vision_tokens = pos_end - pos
    h = w = int(np.sqrt(num_vision_tokens))
    att_shape = [h, w]

    for i in range(len(att)):
        att_cur = att[i][0,:,-1,pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
        general_att_cur = general_att[i][0,:,-1,pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()

        att_map = np.divide(att_cur, general_att_cur, out=np.zeros_like(att_cur), where=general_att_cur != 0)

        att_map = att_map.reshape(att_shape)

        plt.figure(figsize=(8, 8))
        plt.imshow(att_map, cmap='hot', interpolation='nearest')  #'magma'
        plt.colorbar(label='Attention Value')
        plt.axis('off')
        DEBUG_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
        heatmap_save_path = DEBUG_HEATMAP_DIR / (os.path.basename(image_path)[:-4] + "__" + str(i) + ".jpg")
        plt.savefig(heatmap_save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close()  # 关闭图形以释放内存

    return att
    ###-------------end存每一层注意力图----------------
    # return att_map_pic


###---------------Qwen2ForCausalLM hook-----------

import types
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache


def apply_qwen2_attention_hooks(model):
    """应用Qwen2模型的attention输出hooks
    
    Args:
        model: InternVLChatModel 实例，其 language_model 是 Qwen2ForCausalLM
    """
    # 获取 language_model (Qwen2ForCausalLM)
    language_model = model.language_model
    
    # 保存原始的 forward 方法
    if not hasattr(language_model, '_original_forwards'):
        language_model._original_forwards = {
            'causal_lm_forward': language_model.forward,
            'model_forward': language_model.model.forward,
            'layer_forwards': [],
            'attention_forwards': []
        }
        
        # 保存每个 decoder layer 和 attention 的原始 forward 方法
        for layer in language_model.model.layers:
            language_model._original_forwards['layer_forwards'].append(layer.forward)
            language_model._original_forwards['attention_forwards'].append(layer.self_attn.forward)

    
    # 替换 forward 方法
    language_model.forward = types.MethodType(forward_qwen2_causal_lm, language_model)
    language_model.model.forward = types.MethodType(forward_qwen2_model, language_model.model)
    
    # 替换所有 decoder 层和 attention 的 forward 方法
    for i, layer in enumerate(language_model.model.layers):
        layer.forward = types.MethodType(forward_qwen2_decoder_layer, layer)
        # layer.self_attn.forward = types.MethodType(forward_qwen2_attention, layer.self_attn)
        layer.self_attn.config._attn_implementation = "eager"



def restore_original_forwards_qwen2(model):
    """恢复Qwen2模型的原始forward方法
    
    Args:
        model: InternVLChatModel 实例，其 language_model 是 Qwen2ForCausalLM
    """
    language_model = model.language_model
    
    if hasattr(language_model, '_original_forwards'):
        # 恢复模型级别的 forward
        language_model.forward = language_model._original_forwards['causal_lm_forward']
        language_model.model.forward = language_model._original_forwards['model_forward']
        
        # 恢复每个 decoder layer 和 attention 的 forward
        for i, layer in enumerate(language_model.model.layers):
            layer.forward = language_model._original_forwards['layer_forwards'][i]
            layer.self_attn.forward = language_model._original_forwards['attention_forwards'][i]
        
        # 清除保存的原始方法
        delattr(language_model, '_original_forwards')


def forward_qwen2_attention(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor = None,
    past_key_values: Cache = None,
    cache_position: torch.LongTensor = None,
    output_attentions: bool = False,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Qwen2Attention的forward钩子函数，支持输出attention权重
    基于原始代码，只修改为始终返回 attention weights
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # 导入 apply_rotary_pos_emb
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # 使用 eager attention 来获取 attention weights
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    key_states_repeated = repeat_kv(key_states, self.num_key_value_groups)
    value_states_repeated = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) * self.scaling
    
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states_repeated.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states_repeated)
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    
    return attn_output, attn_weights


def forward_qwen2_decoder_layer(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor = None,
    position_ids: torch.LongTensor = None,
    past_key_values: Cache = None,
    use_cache: bool = False,
    cache_position: torch.LongTensor = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    output_attentions: bool = False,
    **kwargs,
) -> tuple:
    """
    Qwen2DecoderLayer的forward钩子函数，支持输出attention权重
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    
    # Self Attention
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        output_attentions=output_attentions,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs


def forward_qwen2_model(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor = None,
    position_ids: torch.LongTensor = None,
    past_key_values: Cache = None,
    inputs_embeds: torch.FloatTensor = None,
    use_cache: bool = None,
    cache_position: torch.LongTensor = None,
    output_attentions: bool = None,
    output_hidden_states: bool = None,
    return_dict: bool = None,
    **kwargs,
) -> BaseModelOutputWithPast:
    """
    Qwen2Model的forward钩子函数，支持输出attention权重
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # 准备 causal mask
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    
    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, past_key_values if use_cache else None, all_hidden_states, all_self_attns] if v is not None)
    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def forward_qwen2_causal_lm(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor = None,
    position_ids: torch.LongTensor = None,
    past_key_values: Cache = None,
    inputs_embeds: torch.FloatTensor = None,
    labels: torch.LongTensor = None,
    use_cache: bool = None,
    cache_position: torch.LongTensor = None,
    logits_to_keep: int = 0,
    output_attentions: bool = None,
    output_hidden_states: bool = None,
    return_dict: bool = None,
    **kwargs,
) -> CausalLMOutputWithPast:
    """
    Qwen2ForCausalLM的forward钩子函数，支持输出attention权重
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
    
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
