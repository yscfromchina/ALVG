import time
import copy
import numpy
import torch
import random
from torchviz import make_dot

from alvg.apis.test import grec_evaluate_f1_nacc

from .test import accuracy
from alvg.datasets import extract_data
from alvg.utils import get_root_logger, reduce_mean, is_main
from collections import defaultdict
import re
from collections import deque

try:
    import apex
except:
    pass


def set_random_seed(seed, deterministic=False):
    """Args:
    seed (int): Seed to be used.
    deterministic (bool): Whether to set the deterministic option for
        CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
        to True and `torch.backends.cudnn.benchmark` to False.
        Default: False.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(epoch, cfg, model, model_ema, optimizer, loader):
    model.train()

    if cfg.distributed:
        loader.sampler.set_epoch(epoch)

    device = list(model.parameters())[0].device

    batches = len(loader)
    end = time.time()

    loss_det_list, det_acc_list, n_acc_list, f1_score_list = (defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list))
    mask_iou_list, mask_ciou_list, mask_giou_list = defaultdict(list), defaultdict(list), defaultdict(list)
    for batch, inputs in enumerate(loader):

        data_time = time.time() - end
        gt_bbox, gt_mask, is_crowd = None, None, None

        if "gt_bbox" in inputs:
            if isinstance(inputs["gt_bbox"], torch.Tensor):
                inputs["gt_bbox"] = [inputs["gt_bbox"][ind] for ind in range(inputs["gt_bbox"].shape[0])]
                gt_bbox = copy.deepcopy(inputs["gt_bbox"])
            else:
                gt_bbox = copy.deepcopy(inputs["gt_bbox"].data[0])

        img_metas = inputs["img_metas"].data[0]

        if "gt_mask" in inputs:
            gt_mask = inputs["gt_mask"].data[0]
        else:
            gt_mask = [None] * len(img_metas)
        if "is_crowd" in inputs:
            is_crowd = inputs.pop("is_crowd").data[0]

        if not cfg.distributed:
            inputs = extract_data(inputs)

        losses, predictions = model(**inputs, rescale=False)

        loss_det = losses.get("loss_total", torch.tensor([0.0], device=device))
        loss_mask = losses.get("loss_mask", torch.tensor([0.0], device=device))
        score_attention_loss = losses.get("score_attention_loss", torch.tensor([0.0], device=device))
        loss = loss_det + loss_mask

        # def parse_dot_source(dot_source):
        #     graph = {}
        #     node_labels = {}
        #     edges = []
        #
        #     for line in dot_source:
        #         line = line.strip().replace("\n", " ")
        #         match_edge = re.match(r"(\d+) -> (\d+)", line)
        #         if match_edge:
        #             src, dst = match_edge.groups()
        #             edges.append((src, dst))
        #             graph.setdefault(dst, []).append(src)
        #
        #         match_label = re.match(r"(\d+) \[label=(.+)\]", line)
        #         if match_label:
        #             node_id, label = match_label.groups()
        #             node_labels[node_id] = label
        #
        #     return graph, node_labels
        #
        # def count_layers_from_dot(loss, target_param):
        #     dot = make_dot(loss, params=dict(model.named_parameters()))
        #     graph, node_labels = parse_dot_source(dot.body)
        #
        #     loss_node = None
        #     target_node = None
        #     for node_id, label in node_labels.items():
        #         if "darkolivegreen1" in label:
        #             loss_node = node_id
        #         if target_param in label:
        #             target_node = node_id
        #
        #     if loss_node is None or target_node is None:
        #         print("未找到 loss 或目标参数在计算图中的节点")
        #         return -1
        #
        #     def bfs_shortest_path(start_node, target_node, graph):
        #         queue = deque([(start_node, 0)])
        #         visited = set()
        #
        #         while queue:
        #             node, depth = queue.popleft()
        #             if node == target_node:
        #                 return depth
        #
        #             if node in visited:
        #                 continue
        #             visited.add(node)
        #
        #             if node in graph:
        #                 for parent in graph[node]:
        #                     queue.append((parent, depth + 1))
        #
        #         return -1
        #
        #     layers_count = bfs_shortest_path(loss_node, target_node, graph)
        #
        #     return layers_count if layers_count != float('inf') else -1
        #
        # target_param_name = "head.transformer.img_enhance.text_proj.weight"
        # layers_count = count_layers_from_dot(loss_det, target_param_name)
        # print(f"Loss_det 传递到 {target_param_name} 需要经过 {layers_count} 层")
        # layers_count = count_layers_from_dot(loss_mask, target_param_name)
        # print(f"Loss_mask 传递到 {target_param_name} 需要经过 {layers_count} 层")
        # layers_count = count_layers_from_dot(score_attention_loss, target_param_name)
        # print(f"score_attention_loss 传递到 {target_param_name} 需要经过 {layers_count} 层")
        # g = make_dot(score_attention_loss, params=dict(model.named_parameters()))

        optimizer.zero_grad()
        if cfg.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if cfg.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
        optimizer.step()

        if cfg.ema:
            model_ema.update_params()

        # if cfg.distributed:
        #     loss_det = reduce_mean(loss_det)
        #     loss_mask = reduce_mean(loss_mask)

        if not isinstance(predictions, list):
            predictions_list = [predictions]
        else:
            predictions_list = predictions
            
        # statistics loss
        for loss_name, loss_value in losses.items():
            if cfg.distributed:
                loss_value = reduce_mean(loss_value)
            loss_det_list[loss_name].append(loss_value.item())
        

        # statistics informations
        map_dict = {0: "decoder"}
        det_acc_dict, f1_score_acc_dict, n_acc_dict = {}, {}, {}
        mask_iou_dict, mask_ciou_dict, mask_giou_dict = {}, {}, {}
        for ind, predictions in enumerate(predictions_list):
            predict_type = map_dict[ind]
            pred_bboxes = predictions.pop("pred_bboxes")
            pred_masks = predictions.pop("pred_masks")
            if not cfg["dataset"] == "GRefCOCO":
                with torch.no_grad():
                    batch_det_acc, batch_mask_iou, mask_acc_at_thrs = accuracy(
                        pred_bboxes,
                        gt_bbox,
                        pred_masks,
                        gt_mask,
                        is_crowd=is_crowd,
                        device=device,
                    )
                    if cfg.distributed:
                        batch_det_acc = reduce_mean(batch_det_acc)
                        batch_mask_iou = reduce_mean(batch_mask_iou)
                det_acc_list[predict_type].append(batch_det_acc.item())
                mask_iou_list[predict_type].append(batch_mask_iou.item())
                # loss_det_list[predict_type].append(loss_det.item())
                det_acc = sum(det_acc_list[predict_type]) / len(det_acc_list[predict_type])
                mask_iou = sum(mask_iou_list[predict_type]) / len(mask_iou_list[predict_type])
                det_acc_dict[predict_type] = det_acc
                mask_iou_dict[predict_type] = mask_iou
            else:
                targets = [meta["target"] for meta in img_metas]
                with torch.no_grad():
                    batch_f1_score, batch_n_acc, mask_res = grec_evaluate_f1_nacc(pred_bboxes, gt_bbox, targets, gt_mask, pred_masks, device=device, img_metas=img_metas)
                    if cfg.distributed:
                        batch_f1_score = reduce_mean(batch_f1_score)
                        batch_n_acc = reduce_mean(batch_n_acc)
                        for key, tensor in mask_res.items():
                            mask_res[key] = reduce_mean(tensor)

                f1_score_list[predict_type].append(batch_f1_score.item())
                n_acc_list[predict_type].append(batch_n_acc.item())
                mask_giou_list[predict_type].append(mask_res['gIoU'].item())
                mask_ciou_list[predict_type].append(mask_res['cIoU'].item())


                # loss_det_list[predict_type].append(loss_det.item())
                f1_score_acc = sum(f1_score_list[predict_type]) / len(f1_score_list[predict_type])
                n_acc = sum(n_acc_list[predict_type]) / len(n_acc_list[predict_type])
                giou = sum(mask_giou_list[predict_type]) / len(mask_giou_list[predict_type])
                ciou = sum(mask_ciou_list[predict_type]) / len(mask_ciou_list[predict_type])

                f1_score_acc_dict[predict_type] = f1_score_acc
                n_acc_dict[predict_type] = n_acc
                mask_giou_dict[predict_type] = giou
                mask_ciou_dict[predict_type] = ciou

        # logging informations
        if is_main() and ((batch + 1) % cfg.log_interval == 0 or batch + 1 == batches):
            loss_str_list = ["{}:{:.3f}".format(loss_n.split("loss_")[-1], sum(loss_v)/len(loss_v)) for loss_n, loss_v in loss_det_list.items()]
            loss_str =  "loss:["+" ".join(loss_str_list) +"]"
            logger = get_root_logger()
            if not cfg["dataset"] == "GRefCOCO":
                ACC_str_list = ["{}Acc:{:.2f}, ".format(map_dict[i], det_acc_dict[map_dict[i]]) for i in range(len(predictions_list))]
                ACC_str = "".join(ACC_str_list)
                iou_str_list = ["{}_iou: {:.2f}, ".format(map_dict[i], mask_iou_dict[map_dict[i]]) for i in range(len(predictions_list))]
                iou_str = "".join(iou_str_list)

                logger.info(
                    f"train-epoch[{epoch+1}]-[{batch+1}/{batches}] "
                    + f"time:{(time.time()- end):.2f}, data_time: {data_time:.2f}, "
                    # + f"loss_det:{sum(loss_det_list[predict_type]) / len(loss_det_list[predict_type]) :.4f}, "
                    + f"{loss_str}, "
                    + f"lr:{optimizer.param_groups[0]['lr']:.6f}, "
                    # + f"DetACC@0.5: {det_acc:.2f}, "
                    + ACC_str
                    + iou_str
                )
            else:
                F1_Score_str_list = [
                    "{}_f1: {:.2f}, ".format(map_dict[i], f1_score_acc_dict[map_dict[i]]) for i in range(len(predictions_list))
                ]
                n_acc_str_list = ["{}_Nacc: {:.2f}, ".format(map_dict[i], n_acc_dict[map_dict[i]]) for i in range(len(predictions_list))]
                giou_str_list = ["{}_giou: {:.2f}, ".format(map_dict[i], mask_giou_dict[map_dict[i]]) for i in range(len(predictions_list))]
                ciou_str_list = ["{}_ciou: {:.2f}, ".format(map_dict[i], mask_ciou_dict[map_dict[i]]) for i in range(len(predictions_list))]

                F1_Score_str = "".join(F1_Score_str_list)
                n_acc_str = "".join(n_acc_str_list)
                giou_str = "".join(giou_str_list)
                ciou_str = "".join(ciou_str_list)

                logger.info(
                    f"train-epoch[{epoch+1}]-[{batch+1}/{batches}] "
                    + f"time:{(time.time()- end):.2f}, data_time: {data_time:.2f}, "
                    # + f"loss_det:{sum(loss_det_list[predict_type]) / len(loss_det_list[predict_type]) :.4f}, "
                    +f"{loss_str}, "
                    + f"lr:{optimizer.param_groups[0]['lr']:.6f}, "
                    + F1_Score_str
                    + n_acc_str
                    + giou_str
                    + ciou_str
                )

        end = time.time()
