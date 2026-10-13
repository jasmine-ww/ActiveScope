import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from skimage.measure import block_reduce
from utils import *
import re
import matplotlib.pyplot as plt
import os
import pickle
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# currently select 22 but feel free to try other layers
ATT_LAYER = 22  #22


def get_pad_token_id(processor):
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    return pad_token_id

# def extract_targets(sentence: str, pattern = r"So I need the information about the following objects: (.+)"):
#     match = re.search(pattern, sentence)
#     if match:
#         return match.group(1)
#     return None

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
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(model.device)

    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=256,
        pad_token_id=get_pad_token_id(processor),
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # targets_sentence = extract_targets(outputs)
    outputs = extract_targets(outputs)
    if outputs is not None:
        targets = split_targets_sentence(outputs, split_tag)
    else:
        targets = []
    
    return targets

def find_target_indices(input_ids, target_text, tokenizer):
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
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

def get_sliced_attention(attentions, q_indices, k_start, k_end, mask_llm=None):
    # attentions[0] is (q, k) since we overrode qwen3_new.py
    q, k = attentions[0]
    q = q[0]  # (heads, q_len, dim)
    k = k[0]  # (heads, k_len, dim)
    
    q_len = q.shape[1]
    
    if isinstance(q_indices, int):
        actual_q_indices = q_indices if q_indices >= 0 else q_len + q_indices
        q_sliced = q[:, actual_q_indices:actual_q_indices+1, :]
        idx_tensor = [actual_q_indices]
    else:
        q_sliced = q[:, q_indices, :]
        idx_tensor = q_indices
        
    import math
    attn_weights = torch.matmul(q_sliced, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
    
    # # Causal mask
    # causal_mask = torch.full((q_len, k.shape[1]), float('-inf'), device=q.device, dtype=q.dtype)
    # causal_mask = torch.triu(causal_mask, diagonal=1)
    # c_mask = causal_mask[idx_tensor, :]
    
    # attn_weights = attn_weights + c_mask.unsqueeze(0)
    
    if mask_llm is not None:
        # mask is 1 for allowed, 0 for blocked. Shape (1, k_len)
        m = (1.0 - mask_llm[0].to(q.dtype)) * (-100000)   #float('-inf')
        attn_weights = attn_weights + m.unsqueeze(0).unsqueeze(0)
        
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    
    # Return shape (heads, len(q_indices), k_slice_len)
    return attn_weights[:, :, k_start:k_end]

def target_attention_qwen2_5(image, prompt, general_prompt, model, processor):
    targets = generate_visual_cues_using_ic(prompt, model, processor)
    
    image_str = encode_base64(image)
    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
    general_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": general_prompt}]}]

    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    general_inputs = prepare_qwen2_5_input(general_messages, processor).to(model.device, torch.bfloat16)
    
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    
    input_ids_list = inputs['input_ids'][0].tolist()
    general_input_ids_list = general_inputs['input_ids'][0].tolist()
    try:
        pos = input_ids_list.index(vision_start_token_id) + 1
        pos_end = input_ids_list.index(vision_end_token_id)
    except ValueError:
        pos = 0
        pos_end = 0

    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()
    
    with torch.no_grad():
        # Do not use mean(dim=0) here because we need attentions as (q, k) tuple!
        att_matrix = model(**inputs, output_attentions=True).attentions
        
        general_outputs = model(**general_inputs, output_attentions=True)
        # Use the last token of the general prompt to represent general attention
        att_matrix_g = get_sliced_attention(general_outputs.attentions, -1, pos, pos_end).mean(dim=0)[0].to(torch.float32).detach().cpu().numpy()
        
    results = {}
    for target in targets:
        indices = find_target_indices(input_ids_list, target, processor.tokenizer)
        if indices:
            target_att = get_sliced_attention(att_matrix, indices, pos, pos_end).mean(dim=0).mean(dim=0)
            target_att = target_att.to(torch.float32).detach().cpu().numpy()
            try:
                target_att = np.divide(target_att, att_matrix_g, out=np.zeros_like(target_att), where=att_matrix_g != 0)
                target_att = target_att.reshape(att_shape)
                results[target] = target_att
            except:
                print(f"Shape mismatch for target {target}")
                
    return results

#---------------------------------------------

def rel_attention_qwen2_5(image, prompt, general_prompt, model, processor, att_map_mask=None, image_path=None):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    image_str = encode_base64(image)

    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
    general_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": general_prompt}]}]

    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    general_inputs = prepare_qwen2_5_input(general_messages, processor).to(model.device, torch.bfloat16)

    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()

    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)

    # ### 生成四维的attention_map
    # dtype=torch.bfloat16
    # sequence_length = inputs['input_ids'].size(1)
    # min_dtype = torch.finfo(dtype).min
    # causal_mask = torch.full(
    #     (sequence_length, sequence_length), fill_value=min_dtype, dtype=dtype, device=inputs['input_ids'].device
    # )
    # if sequence_length != 1:
    #     causal_mask = torch.triu(causal_mask, diagonal=1)
    # causal_mask *= torch.arange(sequence_length, device=inputs['input_ids'].device) > torch.arange(sequence_length, device=inputs['input_ids'].device).reshape(-1, 1)
    # causal_mask[pos:pos_end, pos:pos_end] = 0
    # causal_mask = causal_mask[None, None, :, :].expand(1, 1, -1, -1)
    # # inputs['attention_mask'] = causal_mask
    # Compute attention map for the 14th layer
    # att_map = model(**inputs, output_attentions=True)['attentions'][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(-1,pos_end-pos, pos_end-pos)
    # inputs['attention_mask'] = causal_mask


    attention_mask_llm = inputs['attention_mask'].clone()
    general_mask_llm = general_inputs['attention_mask'].clone()
    if att_map_mask is not None:
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        if isinstance(att_map_mask, np.ndarray):
            att_map_mask = torch.from_numpy(att_map_mask)
        elif not isinstance(att_map_mask, torch.Tensor):
            att_map_mask = torch.tensor(att_map_mask)
        
        att_map_mask = att_map_mask.to(inputs['input_ids'].device).flatten()
        
        if att_map_mask.shape[0] == (pos_end - pos):
            # att_map_mask: 1 表示 mask (移除), 0 表示保留
            # attention_mask: 1 表示保留, 0 表示 mask
            # inputs['attention_mask'][0, pos:pos_end] = inputs['attention_mask'][0, pos:pos_end] * att_map_mask.long()
            attention_mask_llm[0, pos:pos_end] = inputs['attention_mask'][0, pos:pos_end] * att_map_mask.long()
            general_mask_llm[0, pos:pos_end] = general_inputs['attention_mask'][0, pos:pos_end] * att_map_mask.long()
        else:
            print(f"Warning: att_map_mask shape {att_map_mask.shape} does not match image token length {pos_end - pos}")
            
    att_outputs = model(**inputs, output_attentions=True, attention_mask_llm=attention_mask_llm)['attentions']
    att = get_sliced_attention(att_outputs, -1, pos, pos_end, mask_llm=attention_mask_llm).mean(dim=0)[0].to(torch.float32).detach().cpu().numpy()

    general_att_outputs = model(**general_inputs, output_attentions=True, attention_mask_llm=general_mask_llm)['attentions']
    general_att = get_sliced_attention(general_att_outputs, -1, pos, pos_end, mask_llm=general_mask_llm).mean(dim=0)[0].to(torch.float32).detach().cpu().numpy()

    # att = outputs['attentions'][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
    # general_att = general_outputs['attentions'][ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()

    # att_map = att / general_att
    att_map = np.divide(att, general_att, out=np.zeros_like(att), where=general_att != 0)

    att_map = att_map.reshape(att_shape)

    ###-------------存每一层注意力图----------------
    # att = model(**inputs, output_attentions=True, attention_mask_llm=attention_mask_llm)['attentions']   #[ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()

    # general_att = model(**general_inputs, output_attentions=True, attention_mask_llm=general_mask_llm)['attentions']   #[ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()

    # for i in range(len(att)):
    #     att_cur = att[i][0,:,-1,pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
    #     general_att_cur = general_att[i][0,:,-1,pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()

    #     att_map = np.divide(att_cur, general_att_cur, out=np.zeros_like(att_cur), where=general_att_cur != 0)

    #     att_map = att_map.reshape(att_shape)

    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(att_map, cmap='hot', interpolation='nearest')  #'magma'
    #     plt.colorbar(label='Attention Value')
    #     plt.axis('off')
    #     plt.savefig("attention_heatmap.jpg", bbox_inches='tight', pad_inches=0.1, dpi=150)
    #     plt.close()  # 关闭图形以释放内存
    ###-------------end存每一层注意力图----------------

    return att_map
    # return att_map_pic



def rel_attention_qwen2_5_2d(image, prompt, general_prompt, model, processor):

    """
    Compute relative attention scores for Qwen2.5VL.
    
    This function computes the relative attention scores between the input and general inputs
    for Qwen2.5VL. It first computes the attention scores for the input and general inputs, 
    and finally computes the relative attention scores.
    """

    image_str = encode_base64(image)

    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
    general_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": general_prompt}]}]

    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    general_inputs = prepare_qwen2_5_input(general_messages, processor).to(model.device, torch.bfloat16)

    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()

    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)

    att_outputs = model(**inputs, output_attentions=True)['attentions']
    att_map = get_sliced_attention(att_outputs, -1, pos, pos_end).mean(dim=0)[0].to(torch.float32).detach().cpu().numpy().reshape(-1, att_shape[0], att_shape[1])
    ### 生成四维的attention_map
    dtype=torch.bfloat16
    sequence_length = inputs['input_ids'].size(1)
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full(
        (sequence_length, sequence_length), fill_value=min_dtype, dtype=dtype, device=inputs['input_ids'].device
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(sequence_length, device=inputs['input_ids'].device) > torch.arange(sequence_length, device=inputs['input_ids'].device).reshape(-1, 1)
    causal_mask[pos:pos_end, pos:pos_end] = 0
    causal_mask = causal_mask[None, None, :, :].expand(1, 1, -1, -1)
    # inputs['attention_mask'] = causal_mask
    # Compute attention map for the 14th layer
    inputs['attention_mask'] = causal_mask
    
    # We DO NOT support att_map_pic anymore because it requires pos:pos_end x pos:pos_end matrix!
    att_map_pic = None 
    # att_map = att_map.reshape(att_shape)

    return (att_map, att_map_pic), att_shape
    # return att_map_pic, att_shape



def verify_target_existence(image, bbox, target, model, processor):
    x1, y1, x2, y2 = map(int, bbox)
    if x2 <= x1 or y2 <= y1:
        return False
    
    cropped_image = image.crop((x1, y1, x2, y2))
    
    prompt = f"Does the {target} exist in this image? Answer Yes or No."
    
    image_str = encode_base64(cropped_image)
    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
    
    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=20,
            pad_token_id=get_pad_token_id(processor),

            # num_beams=1,
            # # top_k=1,
            # top_p=1.0,
            # # temperature=0.0,
            # max_new_tokens=2,
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return "yes" in output_text.lower()
import time
def target_iterate_attention_qwen2_5(image, prompt, general_prompt, model, processor, max_iters=5, bbox_size=None, image_path = ""):
    # time_a = time.time()

    targets = generate_visual_cues_using_ic(prompt, model, processor)
    # return targets
    # time_b = time.time()
    # print(f"Time for generating visual cues: {time_b - time_a:.2f} seconds")
    results = {}
    results_his = {}
    for target in targets:
        results_his[target] = []
    
    image_str = encode_base64(image)
    messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": prompt}]}]
    general_messages = [{"role": "user", "content": [{"type": "image", "image": f'data:image;base64,{image_str}'}, {"type": "text", "text": general_prompt}]}]

    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    general_inputs = prepare_qwen2_5_input(general_messages, processor).to(model.device, torch.bfloat16)
    
    att_shape = (inputs['image_grid_thw'][0, 1:] / 2).cpu().numpy().astype(int).tolist()
    h, w = att_shape
    
    exclusion_mask = np.ones((h, w), dtype=np.float32)
    
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    
    input_ids_list = inputs['input_ids'][0].tolist()
    try:
        pos = input_ids_list.index(vision_start_token_id) + 1
        pos_end = input_ids_list.index(vision_end_token_id)
    except ValueError:
        pos = 0
        pos_end = 0

    
    for i in range(max_iters):
        if all(t in results for t in targets):
            break
            
        flat_mask = exclusion_mask.flatten()
        att_map_mask = torch.from_numpy(flat_mask).to(model.device)
        
        current_inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
        current_general_inputs = prepare_qwen2_5_input(general_messages, processor).to(model.device, torch.bfloat16)
        
        
        attention_mask_llm = current_inputs['attention_mask'].clone()
        general_mask_llm = current_general_inputs['attention_mask'].clone()
        
        attention_mask_llm[0, pos:pos_end] = attention_mask_llm[0, pos:pos_end] * att_map_mask.long()
        general_mask_llm[0, pos:pos_end] = general_mask_llm[0, pos:pos_end] * att_map_mask.long()
        
        # time_c = time.time()
        with torch.no_grad():
        # with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):

            att_matrix = model(**current_inputs, output_attentions=True, attention_mask_llm=attention_mask_llm).attentions
            
            #  att_matrix = outputs
            del current_inputs
            torch.cuda.empty_cache()
            
            gen_att_matrix = model(**current_general_inputs, output_attentions=True, attention_mask_llm=general_mask_llm).attentions
            att_matrix_g = get_sliced_attention(gen_att_matrix, -1, pos, pos_end, mask_llm=general_mask_llm).mean(dim=0)[0].to(torch.float32).detach().cpu().numpy()
            del current_general_inputs
            torch.cuda.empty_cache()           #  att_matrix_g = general_outputs
        
        # time_d = time.time()
        # print(f"Time for attention computation in iteration {i+1}: {time_d - time_c:.2f} seconds")

        # found_new = False
        # for i in range(3):
        #     target_att = att_matrix[-1, pos:pos_end]
        #     target_att = target_att.to(torch.float32).detach().cpu().numpy()
            
        #     target_att = np.divide(target_att, att_matrix_g, out=np.zeros_like(target_att), where=att_matrix_g != 0)
        #     target_att = target_att.reshape(att_shape)
            
        #     bbox, mask_region = bbox_from_att_image_adaptive(target_att, image.size, return_mask=True)
        #     results_his[target].append(bbox)
            
        #     mask_cur = (mask_region == 0).astype(int)
        #     exclusion_mask = exclusion_mask * mask_cur
        found_new = False
        for target in targets:
            if target in results:
                continue
                
            indices = find_target_indices(input_ids_list, target, processor.tokenizer)
            if indices:
                target_att = get_sliced_attention(att_matrix, indices, pos, pos_end, mask_llm=attention_mask_llm).mean(dim=0).mean(dim=0)
                target_att = target_att.to(torch.float32).detach().cpu().numpy()
                
                target_att = np.divide(target_att, att_matrix_g, out=np.zeros_like(target_att), where=att_matrix_g != 0)
                target_att = target_att.reshape(att_shape)
                
                bbox_size = int(image.size[0]*0.2)
                bbox, mask_region = bbox_from_att_image_adaptive(target_att, image.size, return_mask=True, bbox_size=bbox_size)
                results_his[target].append(bbox)

                if verify_target_existence(image, bbox, target, model, processor):
                    results[target] = bbox
                    found_new = True
                else:
                    mask_cur = (mask_region == 0).astype(int)
                    exclusion_mask = exclusion_mask * mask_cur
        # time_e = time.time()
        # print(f"Time for target verification in iteration {i+1}: {time_e - time_d:.2f} seconds")
                    
        if not found_new and i < max_iters - 1:
             pass
        
    # print("results: ", results)
    # print("results_his: ", results_his)
    # 把results_his的信息放到results里
    for target in results_his:
        if target in results:
            results[target] = (results[target], results_his[target])
             
    return results
