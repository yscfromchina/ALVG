import argparse
import torch.distributed as dist
from transformers import XLMRobertaTokenizer
import cv2
import torch
import numpy as np
import sys
import torch.nn.functional as F
from alvg.apis import evaluate_model, set_random_seed
from alvg.datasets import build_dataset, build_dataloader
from alvg.models import build_model, ExponentialMovingAverage
from alvg.utils import (get_root_logger, load_checkpoint, init_dist,
                         is_main, load_pretrained_checkpoint)
import pycocotools.mask as maskUtils

from mmcv.runner import get_dist_info
from mmcv.utils import Config, DictAction
from mmcv.parallel import MMDistributedDataParallel
import os
try:
    import apex
except:
    pass

mean = torch.tensor([123.675, 116.28, 103.53]) / 255.0 
std = torch.tensor([58.395, 57.12, 57.375]) / 255.0

def normalize_image(image_tensor):
    mean_tensor = mean.view(1, 3, 1, 1)
    std_tensor = std.view(1, 3, 1, 1)
    return (image_tensor - mean_tensor) / std_tensor


def main_worker(cfg):
    cfg.distributed = False
    if cfg.launcher == "pytorch":
        cfg.distributed = True
        init_dist()
    cfg.rank, cfg.world_size = get_dist_info()


    model = build_model(cfg.model)
    model = model.cuda()

    if cfg.use_fp16:
        model = apex.amp.initialize(
            model, opt_level="O1")
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True
    if cfg.distributed:
        model = MMDistributedDataParallel(model, device_ids=[cfg.rank])
    model_ema = ExponentialMovingAverage(model, cfg.ema_factor) if cfg.ema else None
    if cfg.load_from:
        load_checkpoint(model, model_ema, load_from=cfg.load_from)
    elif cfg.finetune_from:
        # hacky way
        load_pretrained_checkpoint(model, model_ema, cfg.finetune_from, amp=cfg.use_fp16)





def parse_args():
    parser = argparse.ArgumentParser(description="SeqTR-test")
    parser.add_argument('config', help='test configuration file path.')
    parser.add_argument(
        '--load-from', help='load from the saved .pth checkpoint, only used in validation.')
    parser.add_argument(
        '--finetune-from', help='load from the pretrained checkpoint, only used in validation.')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    tokenizer = XLMRobertaTokenizer("pretrain_weights/beit3.spm")
    expression = 'the man with a camera'
    tokens = tokenizer.tokenize(expression)
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens = [tokenizer.bos_token_id] + tokens[:] + [tokenizer.eos_token_id]
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (20 - num_tokens)
    ref_expr_inds = tokens + [tokenizer.pad_token_id] * (20 - num_tokens)
    ref_expr_inds = np.array(ref_expr_inds, dtype=int)
    ref_expr_inds = torch.tensor(ref_expr_inds, dtype=torch.int64).unsqueeze(0)
    text_attention_mask = torch.tensor(padding_mask, dtype=torch.int64).unsqueeze(0)

    image_path = 'img/000000005193.jpg'
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    image_tensor = normalize_image(image_tensor)
    img_metas = []
    img_metas.append({"batch_input_shape": (640, 640), "img_shape": (640, 640, 3), "target": 1})
    cfg = Config.fromfile("work_dir/single/ViT-base/refcocog/refcocog/20250122_090728/20250122_090728_refcocog.py")
    cfg.model['head']['img_enhance'] = True
    model = build_model(cfg.model)
    model = model.cuda()
    load_checkpoint(model, None, load_from="work_dir/single/ViT-base/refcocog/refcocog/20250122_090728/det_best.pth")


    device = torch.device("cuda" if torch.cuda.is_available() else "")
    image_tensor = image_tensor.to(device)
    ref_expr_inds = ref_expr_inds.to(device)
    text_attention_mask = text_attention_mask.to(device)
    model.eval()
    res, attn_map, score = model.forward_test(image_tensor, ref_expr_inds, img_metas, text_attention_mask)
    mask = res[0]['pred_masks'][0]['masks']


    original_image = cv2.imread(image_path)
    height, width = original_image.shape[:2]
    h_scale, w_scale = 640 / height, 640 / width

    grid_size = 32

    # for i in range(0, width, grid_size):
    #     cv2.line(original_image, (i, 0), (i, height), (255, 255, 255), 1)
    #
    # for i in range(0, height, grid_size):
    #     cv2.line(original_image, (0, i), (width, i), (255, 255, 255), 1)
    if cfg.model['head']['img_enhance']:
        attention = score[0]  # Shape: [h, w]

        attention_resized = F.interpolate(attention.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bilinear")
        attention_resized = attention_resized.squeeze().cpu().numpy()
        light = attention_resized.reshape(height, width, 1)
        light = light / light.max()
        enhanced_image = original_image * light
        attention_normalized = cv2.normalize(attention_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        heatmap = cv2.applyColorMap(attention_normalized, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)
        output_path = f'output/test_score_with_heatmap.jpg'
        cv2.imwrite(output_path, blended)
        cv2.imwrite(f'output/enhance.jpg', enhanced_image)
    num_layers, bs, num_queries, h, w = attn_map.shape

    for layer in range(num_layers):
        for query in range(num_queries):
            attention = attn_map[layer, 0, query]  # Shape: [h, w]

            attention_resized = F.interpolate(attention.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bilinear")
            attention_resized = attention_resized.squeeze().cpu().numpy()

            attention_normalized = cv2.normalize(attention_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap = cv2.applyColorMap(attention_normalized, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)

            output_path = f'output/test_layer{layer}_query{query}_with_heatmap.jpg'
            cv2.imwrite(output_path, blended)
    res = res[0]['pred_bboxes'][0]
    scores = res['scores']
    boxes = res['boxes']
    high_confidence_scores_idx = torch.where(scores > 0.7)[0]
    high_confidence_boxes = boxes[high_confidence_scores_idx].cpu().numpy()
    for box in high_confidence_boxes:
        x1, y1, x2, y2 = box.astype(int)
        x1, x2 = x1 / w_scale, x2 / w_scale
        y1, y2 = y1 / h_scale, y2 / h_scale
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        original_image = cv2.rectangle(original_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

    mask = mask.cpu().numpy()
    mask[mask > 0.5] = 1
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    colored_mask = np.zeros_like(original_image)
    colored_mask[mask == 1] = [0, 0, 255]

    alpha = 0.5
    overlay = cv2.addWeighted(original_image, 1, colored_mask, 1 - alpha, 0)

    output_path = f'output/bbox.jpg'
    cv2.imwrite(output_path, overlay)


if __name__ == '__main__':
    main()
