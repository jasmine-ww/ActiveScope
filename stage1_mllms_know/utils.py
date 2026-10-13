import torchvision.transforms.functional as TF
import numpy as np
import os
from scipy.ndimage import median_filter
# from skimage.measure import block_reduce
from qwen_vl_utils import process_vision_info
from io import BytesIO
import base64

def encode_base64(image):
    """
    Encodes a PIL image to a base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def prepare_qwen2_5_input(messages, processor):

    """
    Prepare the input for Qwen2.5VL.
    """

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    return inputs


def prepare_qwen2_5_input_encodedimg(messages, processor, image_inputs=None, video_inputs=None):

    """
    Prepare the input for Qwen2.5VL.
    """

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    return inputs

# def high_pass_filter(image, resolusion, km=7, kh=3, reduce=True):
#     """
#     Applies a high-pass filter to an image to highlight edges and fine details.
    
#     This function resizes the image, applies a Gaussian blur to create a low-frequency version,
#     subtracts it from the original to get high-frequency components, and then applies median filtering.
    
#     Args:
#         image: Input PIL image
#         resolusion: Target resolution to resize the image to
#         km: Kernel size for median filtering (default: 7)
#         kh: Kernel size for Gaussian blur (default: 3)
#         reduce: Whether to reduce the output size using block reduction (default: True)
        
#     Returns:
#         h_brightness: A 2D numpy array representing the high-frequency components of the image
#     """

#     image = TF.resize(image, (resolusion, resolusion))
#     image = TF.to_tensor(image).unsqueeze(0)
#     l = TF.gaussian_blur(image, kernel_size=(kh, kh)).squeeze().detach().cpu().numpy()
#     h = image.squeeze().detach().cpu().numpy() - l
#     h_brightness = np.sqrt(np.square(h).sum(axis=0))
#     h_brightness = median_filter(h_brightness, size=km)
#     if reduce:
#         h_brightness = block_reduce(h_brightness, block_size=(14, 14), func=np.sum)

#     return h_brightness

def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336, return_mask=False, bbox=None):
    """
    Generates an adaptive bounding box for original image from an attention map.
    
    This function finds the region with the highest attention in the attention map
    and creates a bounding box around it. It tries different crop ratios and selects
    the one that produces the sharpest attention difference.
    
    Args:
        att_map: A 2D numpy array representing the attention map (e.g., 24x24 for LLaVA or 16x16 for BLIP)
        image_size: Tuple of (width, height) of the original image
        bbox_size: Base size for the bounding box (default: 336)
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box in the original image
    """
    if bbox is not None:
        att_map = att_map.copy()
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]
        x1, y1, x2, y2 = bbox
        att_x1 = int(x1 / block_size[0])
        att_y1 = int(y1 / block_size[1])
        att_x2 = int(np.ceil(x2 / block_size[0]))
        att_y2 = int(np.ceil(y2 / block_size[1]))
        att_x1 = max(0, att_x1)
        att_y1 = max(0, att_y1)
        att_x2 = min(att_map.shape[1], att_x2)
        att_y2 = min(att_map.shape[0], att_y2)
        att_map[att_y1:att_y2, att_x1:att_x2] = 0

    # the ratios corresponds to the bounding box we are going to crop the image
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        # perform a bbox_size*r width and bbox_size*r height crop, where bbox_size is the size of the model's original image input resolution. (336 for LLaVA, 224 for BLIP)

        # the size of each block in the attention map, in the original image
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        # if I want a bbox_size*r width and bbox_size*r height crop from the original image, the number of blocks I need (x, y)
        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))
        
        # attention aggregation map
        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        # sliding window to find the block with the highest attention
        for x in range(att_map.shape[1]-block_num[0]+1): 
            for y in range(att_map.shape[0]-block_num[1]+1): 
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)
        
        # we have the position of max attention, we can calculate the difference between the max attention and the average of its adjacent attentions, to see if it is sharp enough, the more difference, the sharper
        # we choose the best ratio r according to their attention difference
        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]-1])
        if max_att_pos[0] < sliding_att.shape[1]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]+1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1]-1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1]+1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)
    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]
    
    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)
    
    x_center = selected_bbox_size//2 if x_center < selected_bbox_size//2 else x_center
    y_center = selected_bbox_size//2 if y_center < selected_bbox_size//2 else y_center
    x_center = image_size[0] - selected_bbox_size//2 if x_center > image_size[0] - selected_bbox_size//2 else x_center
    y_center = image_size[1] - selected_bbox_size//2 if y_center > image_size[1] - selected_bbox_size//2 else y_center

    x1 = max(0, x_center - selected_bbox_size//2)
    y1 = max(0, y_center - selected_bbox_size//2)
    x2 = min(image_size[0], x_center + selected_bbox_size//2)
    y2 = min(image_size[1], y_center + selected_bbox_size//2)

    if return_mask:
        # 生成与 att_map 相同形状的 mask
        mask = np.zeros_like(att_map)
        
        # 将原图像坐标转换回 att_map 坐标
        att_x1 = int(x1 / block_size[0])
        att_y1 = int(y1 / block_size[1])
        att_x2 = int(np.ceil(x2 / block_size[0]))
        att_y2 = int(np.ceil(y2 / block_size[1]))
        
        # 确保坐标在有效范围内
        att_x1 = max(0, att_x1)
        att_y1 = max(0, att_y1)
        att_x2 = min(att_map.shape[1], att_x2)
        att_y2 = min(att_map.shape[0], att_y2)
        
        # 设置 bbox 区域为有效值（可以是1，或者使用原始的attention值）
        mask[att_y1:att_y2, att_x1:att_x2] = att_map[att_y1:att_y2, att_x1:att_x2]
        return (x1, y1, x2, y2), mask

    return x1, y1, x2, y2

def high_res_split_threshold(image, res_threshold=1024):
    """
    Splits a high-resolution image into smaller patches.
    
    This function divides a large image into smaller patches to process them individually,
    which is useful for handling high-resolution images that might be too large for direct processing.
    
    Args:
        image: Input PIL image
        res_threshold: Maximum resolution threshold before splitting (default: 1024)
        
    Returns:
        tuple: (split_images, vertical_split, horizontal_split)
            - split_images: List of PIL image patches
            - vertical_split: Number of vertical splits
            - horizontal_split: Number of horizontal splits
    """

    vertical_split = int(np.ceil(image.size[1] / res_threshold))
    horizontal_split = int(vertical_split * image.size[0] / image.size[1])

    split_num = (horizontal_split, vertical_split)
    split_size = int(np.ceil(image.size[0] / split_num[0])), int(np.ceil(image.size[1] / split_num[1]))
    
    split_images = []
    for j in range(split_num[1]):
        for i in range(split_num[0]):
            split_image = image.crop((i*split_size[0], j*split_size[1], (i+1)*split_size[0], (j+1)*split_size[1]))
            split_images.append(split_image)
    
    return split_images, vertical_split, horizontal_split

def high_res(map_func, image, prompt, general_prompt, model, processor):
    """
    Applies an attention mapping function to high-resolution images by splitting and recombining.
    
    This function splits a high-resolution image into smaller patches, applies the specified
    attention mapping function to each patch, and then recombines the results into a single
    attention map.
    
    Args:
        map_func: The attention mapping function to apply to each patch
        image: Input PIL image
        prompt: Text prompt for the attention function
        general_prompt: General text prompt for baseline comparison
        model: Model instance (LLaVA or BLIP)
        processor: Processor for the corresponding model
        
    Returns:
        block_att: A 2D numpy array representing the combined attention map for the entire image
    """

    split_images, num_vertical_split, num_horizontal_split = high_res_split_threshold(image)
    att_maps = []
    for split_image in split_images:
        att_map = map_func(split_image, prompt, general_prompt, model, processor)
        # att_map = att_map / att_map.mean()
        att_maps.append(att_map)
    block_att = np.block([att_maps[j:j+num_horizontal_split] for j in range(0, num_horizontal_split * num_vertical_split, num_horizontal_split)])

    return block_att



def bbox_difference(att_map, image_size, bbox_size=336, return_mask=False):
    """
    Generates an adaptive bounding box for original image from an attention map.
    
    This function finds the region with the highest attention in the attention map
    and creates a bounding box around it. It tries different crop ratios and selects
    the one that produces the sharpest attention difference.
    
    Args:
        att_map: A 2D numpy array representing the attention map (e.g., 24x24 for LLaVA or 16x16 for BLIP)
        image_size: Tuple of (width, height) of the original image
        bbox_size: Base size for the bounding box (default: 336)
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box in the original image
    """

    # the ratios corresponds to the bounding box we are going to crop the image
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        # perform a bbox_size*r width and bbox_size*r height crop, where bbox_size is the size of the model's original image input resolution. (336 for LLaVA, 224 for BLIP)

        # the size of each block in the attention map, in the original image
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        # if I want a bbox_size*r width and bbox_size*r height crop from the original image, the number of blocks I need (x, y)
        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                return 0    #, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))
        
        # attention aggregation map
        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        # sliding window to find the block with the highest attention
        for x in range(att_map.shape[1]-block_num[0]+1): 
            for y in range(att_map.shape[0]-block_num[1]+1): 
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)
        
        # we have the position of max attention, we can calculate the difference between the max attention and the average of its adjacent attentions, to see if it is sharp enough, the more difference, the sharper
        # we choose the best ratio r according to their attention difference
        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]-1])
        if max_att_pos[0] < sliding_att.shape[1]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]+1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1]-1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1]+1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)
    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]
    return max(differences)
    

def nms(bboxes, scores, iou_threshold=0.5, top_k=5):
    """
    非极大值抑制（NMS），去除重叠度高的bbox，保留top_k个分数高的
    bboxes: [(x1, y1, x2, y2), ...]
    scores: [score1, score2, ...]
    返回：保留的bbox索引列表
    """
    if len(bboxes) == 0:
        return []
    bboxes_ = []
    for bbox in bboxes:
        bboxes_.append(bbox[:4])

    bboxes = np.array(bboxes_)
    scores = np.array(scores)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < top_k:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def bbox_patch_from_att_image(att_map, image_size, bbox_size=336, return_mask=False):
    """
    Generates an adaptive bounding box for original image from an attention map.
    
    This function finds the region with the highest attention in the attention map
    and creates a bounding box around it. It tries different crop ratios and selects
    the one that produces the sharpest attention difference.
    
    Args:
        att_map: A 2D numpy array representing the attention map (e.g., 24x24 for LLaVA or 16x16 for BLIP)
        image_size: Tuple of (width, height) of the original image
        bbox_size: Base size for the bounding box (default: 336)
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box in the original image
    """

    # the ratios corresponds to the bounding box we are going to crop the image
    ratios = [1]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        # perform a bbox_size*r width and bbox_size*r height crop, where bbox_size is the size of the model's original image input resolution. (336 for LLaVA, 224 for BLIP)

        # the size of each block in the attention map, in the original image
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        # if I want a bbox_size*r width and bbox_size*r height crop from the original image, the number of blocks I need (x, y)
        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                # return [(0, 0, image_size[0], image_size[1])]
                return []
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))
        
        # attention aggregation map
        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        # sliding window to find the block with the highest attention
        for x in range(att_map.shape[1]-block_num[0]+1): 
            for y in range(att_map.shape[0]-block_num[1]+1): 
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)
        
        # # 自适应确定bbox数量
        # k = determine_adaptive_k(sliding_att, att_map)

        # # 获取top-k个候选位置（取2倍用于NMS筛选）
        # candidate_k = min(k * 3, sliding_att.size)  # 候选数量，用于后续NMS
        # flat_indices = np.argpartition(sliding_att.flatten(), -candidate_k)[-candidate_k:]
        # topk_indices = flat_indices[np.argsort(sliding_att.flatten()[flat_indices])[::-1]]

        #接下来选择sliding_att中的最大的k个值，以及对应的位置
        # k = int(len(sliding_att.flatten())/4)  # 你可以根据需要调整k的数量
        k = int(len(sliding_att.flatten()))  # 你可以根据需要调整k的数量
        flat_indices = np.argpartition(sliding_att.flatten(), -k)[-k:]
        topk_indices = flat_indices[np.argsort(sliding_att.flatten()[flat_indices])[::-1]]
        bbox_addr = []
        scores = []
        for idx in topk_indices:
            y, x = np.unravel_index(idx, sliding_att.shape)
            bbox_addr.append((x,y, x+block_num[0], y+block_num[1], block_size))
            # # 计算bbox在原图中的坐标
            # x1 = int(x * block_size[0])
            # y1 = int(y * block_size[1])
            # x2 = int(min(image_size[0], (x + block_num[0]) * block_size[0]))
            # y2 = int(min(image_size[1], (y + block_num[1]) * block_size[1]))
            # bbox_addr.append((x1, y1, x2, y2))
            scores.append(sliding_att[y, x])
        k = 10
        keep = nms(bbox_addr, scores, iou_threshold=0.3, top_k=k)
        bbox_addr = [bbox_addr[i] for i in keep]

        if return_mask:
            # 生成与 att_map 相同形状的 mask
            mask = np.zeros_like(att_map)
            for idx in topk_indices:
                y, x = np.unravel_index(idx, sliding_att.shape)
                att_x1 = x
                att_y1 = y
                att_x2 = min(att_map.shape[1], x + block_num[0])
                att_y2 = min(att_map.shape[0], y + block_num[1])
                mask[att_y1:att_y2, att_x1:att_x2] = att_map[att_y1:att_y2, att_x1:att_x2]
            return bbox_addr, mask


    return bbox_addr



import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12, use_anyres:bool=True):

    transform = build_transform(input_size=input_size)
    if use_anyres:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        images = [image.resize((input_size, input_size))]
    # images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
