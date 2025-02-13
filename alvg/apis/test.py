import time
import torch
import numpy
import json
import numpy as np
import pycocotools.mask as maskUtils
from alvg.datasets import extract_data
from alvg.utils import get_root_logger, reduce_mean, is_main
from torchvision.ops.boxes import box_area
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from collections import defaultdict


def computeIoU(pred_seg, gd_seg):

    pred_seg = pred_seg.bool()
    gd_seg = gd_seg.bool()

    I = torch.sum(torch.logical_and(pred_seg, gd_seg))
    U = torch.sum(torch.logical_or(pred_seg, gd_seg))

    return I, U


def mask_overlaps(gt_mask, pred_masks, is_crowd):
    """Args:
    gt_mask (list[RLE]):
    pred_mask (list[RLE]):
    """

    def computeIoU_RLE(gt_mask, pred_masks, is_crowd):
        mask_iou = maskUtils.iou(pred_masks, gt_mask, [False] * len(pred_masks))
        mask_iou = numpy.diag(mask_iou)
        return mask_iou

    mask_iou = computeIoU_RLE(gt_mask, pred_masks, is_crowd)
    mask_iou = torch.from_numpy(mask_iou)

    return mask_iou


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def accuracy(pred_bboxes, gt_bbox, pred_masks, gt_mask, is_crowd=None, device="cuda:0"):
    eval_det = pred_bboxes is not None
    eval_mask = pred_masks is not None and isinstance(gt_mask[0], torch.Tensor)

    det_acc = torch.tensor([0.0], device=device)
    bbox_iou = torch.tensor([0.0], device=device)
    if eval_det:
        gt_bbox = torch.stack(gt_bbox).to(device)
        bbox_iou = bbox_overlaps(gt_bbox, pred_bboxes, is_aligned=True)
        det_acc = (bbox_iou >= 0.5).float().mean()

    mask_iou = torch.tensor([0.0], device=device)
    mask_acc_at_thrs = torch.full((5,), -1.0, device=device)
    if eval_mask:
        pred_masks = [mask >= 0.5 for mask in pred_masks]
        pred_rles = [maskUtils.encode(np.asfortranarray(mask[0].cpu().numpy().astype(np.uint8))) for mask in pred_masks]
        gt_rles = [maskUtils.encode(np.asfortranarray(mask[0].cpu().numpy().astype(np.uint8))) for mask in gt_mask]

        mask_iou = mask_overlaps(gt_rles, pred_rles, is_crowd).to(device)
        for i, iou_thr in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
            mask_acc_at_thrs[i] = (mask_iou >= iou_thr).float().mean()

    return det_acc * 100.0, torch.sum(mask_iou * 100.0) / len(mask_iou), mask_acc_at_thrs * 100.0


def grec_evaluate_f1_nacc(predictions, gt_bboxes, targets, gt_masks, pred_masks, thresh_score=0.7, thresh_iou=0.5, thresh_F1=1.0, device="cuda:0", img_metas=None):
    correct_image = torch.tensor(0, device=device)
    num_image = torch.tensor(0, device=device)
    nt = {
        "TP": torch.tensor(0.0, device=device),
        "TN": torch.tensor(0.0, device=device),
        "FP": torch.tensor(0.0, device=device),
        "FN": torch.tensor(0.0, device=device),
    }
    ### seg ###
    nt_seg = {
        "TP": torch.tensor(0.0, device=device),
        "TN": torch.tensor(0.0, device=device),
        "FP": torch.tensor(0.0, device=device),
        "FN": torch.tensor(0.0, device=device),
    }

    accum_I = torch.tensor(0.0, device=device)
    accum_U = torch.tensor(0.0, device=device)
    accum_IoU = torch.tensor(0.0, device=device)
    not_empty_count = torch.tensor(0.0, device=device)
    empty_count = torch.tensor(0.0, device=device)
    pred_masks = [pred_mask['masks'] >= 0.5 for pred_mask in pred_masks]
    ### seg ###

    if predictions is None:
        return torch.tensor(0.0, device=device).float(), torch.tensor(0.0, device=device).float()
    # error_info_list = []
    for prediction, gt_bbox, target, img_meta, gt_mask, pred_mask in zip(predictions, gt_bboxes, targets, img_metas, gt_masks, pred_masks):
        TP = 0

        assert prediction is not None
        sorted_scores_boxes = sorted(zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True)
        sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
        sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
        converted_bbox_all = []
        no_target_flag = False
        for converted_bbox, one_target in zip(gt_bbox, target):
            if one_target["category_id"] == -1:
                no_target_flag = True
            # target_bbox = one_target["bbox"]
            # converted_bbox = [
            #     target_bbox[0],
            #     target_bbox[1],
            #     target_bbox[2] + target_bbox[0],
            #     target_bbox[3] + target_bbox[1],
            # ]
            converted_bbox_all.append(converted_bbox)
        gt_bbox_all = torch.stack(converted_bbox_all, dim=0)

        sorted_scores_array = numpy.array(sorted_scores)
        idx = sorted_scores_array >= thresh_score
        filtered_boxes = sorted_boxes[idx]
        # filtered_boxes = sorted_boxes[0:1]
        # error_info = {"filename": img_meta["filename"], "expression": img_meta["expression"]}

        giou = generalized_box_iou(filtered_boxes, gt_bbox_all.view(-1, 4))
        # error_info["gt_bbox"] = gt_bbox_all.view(-1, 4).cpu().numpy().tolist()
        num_prediction = filtered_boxes.shape[0]
        num_gt = gt_bbox_all.shape[0]
        # error_info["num_prediction"] = num_prediction
        # error_info["num_gt"] = num_gt
        if gt_mask is None:                 # not evaluating mask
            gt_mask = pred_mask.cpu()
        gt_mask = torch.any(gt_mask, dim=0)
        pred_mask = pred_mask.cpu()
        pred_nt = torch.max(pred_mask) == 0

        I, U = computeIoU(pred_mask, gt_mask)

        if no_target_flag:
            empty_count += 1
            if pred_nt:
                nt_seg['TP'] += 1
                accum_IoU += 1
                accum_I += 0
                accum_U += 0
            else:
                nt_seg["FN"] += 1
                accum_IoU += 0
                accum_I += 0
                accum_U += int(U)

            if num_prediction >= 1:
                # error_info["no_target"] = True
                nt["FN"] += 1
                F_1 = torch.tensor(0.0, device=device)
            else:
                nt["TP"] += 1
                F_1 = torch.tensor(1.0, device=device)
        else:
            if pred_nt:
                nt_seg["FP"] += 1
                I = 0
            # True Negative
            else:  # TN
                nt_seg["TN"] += 1
            this_iou = float(0) if U == 0 else float(I) / float(U)

            accum_IoU += this_iou
            accum_I += I
            accum_U += U

            not_empty_count += 1

            if num_prediction >= 1:
                nt["TN"] += 1
            else:
                nt["FP"] += 1
            for i in range(min(num_prediction, num_gt)):
                top_value, top_index = torch.topk(giou.flatten(0, 1), 1)
                if top_value < thresh_iou:
                    # error_info["top_value"] = top_value.cpu().numpy().tolist()
                    break
                else:
                    top_index_x = top_index // num_gt
                    top_index_y = top_index % num_gt
                    TP += 1
                    giou[top_index_x[0], :] = 0.0
                    giou[:, top_index_y[0]] = 0.0
            FP = num_prediction - TP
            FN = num_gt - TP
            F_1 = 2 * TP / (2 * TP + FP + FN)
            # error_info["True Positives"] = TP
            # error_info["False Positives"] = FP
            # error_info["False Negatives"] = FN

        if F_1 >= thresh_F1:
            correct_image += 1
            # error_info_list.append(error_info)
        num_image += 1

    F1_score = correct_image / num_image
    # T_acc = nt["TN"] / (nt["TN"] + nt["FP"])
    N_acc = nt["TP"] / (nt["TP"] + nt["FN"]) if nt["TP"] != 0 else torch.tensor(0.0, device=device)

    res = {}
    res['gIoU'] = 100. * (accum_IoU / num_image)
    res['cIoU'] = accum_I * 100. / accum_U

    if empty_count > 0:
        res['T_acc'] = nt_seg['TP'] / (nt_seg['TP'] + nt_seg['FP'] + 1)
        res['N_acc'] = nt_seg['TP'] / (nt_seg['TP'] + nt_seg['FN'] + 1)
    else:
        res['T_acc'] = res['N_acc'] = torch.tensor(0.0, device=device)

    return F1_score.float() * 100, N_acc.float() * 100, res


def evaluate_model(epoch, cfg, model, loader):
    model.eval()

    device = list(model.parameters())[0].device

    batches = len(loader)
    end = time.time()

    with_bbox, with_mask = False, False
    det_acc_list, mask_iou_list, mask_acc_list, f1_score_list, n_acc_list = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    mask_ciou_list, mask_giou_list = defaultdict(list), defaultdict(list)

    error_info_lists = []
    with torch.no_grad():
        for batch, inputs in enumerate(loader):
            gt_bbox, gt_mask, is_crowd = None, None, None

            if "gt_bbox" in inputs:
                with_bbox = True
                if isinstance(inputs["gt_bbox"], torch.Tensor):
                    inputs["gt_bbox"] = [inputs["gt_bbox"][ind] for ind in range(inputs["gt_bbox"].shape[0])]
                    gt_bbox = inputs.pop("gt_bbox")
                else:
                    gt_bbox = inputs.pop("gt_bbox").data[0]

            if "gt_mask" in inputs:
                with_mask = True
                gt_mask = inputs.pop("gt_mask").data[0]
            else:
                gt_mask = [None] * len(gt_bbox)
            if "is_crowd" in inputs:
                is_crowd = inputs.pop("is_crowd").data[0]

            img_metas = inputs["img_metas"].data[0]

            if not cfg.distributed:
                inputs = extract_data(inputs)

            predictions, _, _ = model(
                **inputs,
                return_loss=False,
                rescale=False,
                with_bbox=with_bbox,
                with_mask=with_mask,
            )

            if not isinstance(predictions, list):
                predictions_list = [predictions]
            else:
                predictions_list = predictions

            # statistics informations
            map_dict = {0: "decoder", 1: "token"}
            det_acc_dict, f1_score_acc_dict, n_acc_dict = {}, {}, {}
            mask_iou_dict, mask_ciou_dict, mask_giou_dict = {}, {}, {}

            for ind, predictions in enumerate(predictions_list):
                predict_type = map_dict[ind]
                pred_bboxes = predictions.pop("pred_bboxes")
                pred_masks = predictions.pop("pred_masks")
                if not cfg["dataset"] == "GRefCOCO":
                    with torch.no_grad():
                        batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs = accuracy(
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
                            batch_mask_acc_at_thrs = reduce_mean(batch_mask_acc_at_thrs)
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
                        batch_f1_score, batch_n_acc, mask_res = grec_evaluate_f1_nacc(pred_bboxes, gt_bbox, targets,
                                                                                      gt_mask, pred_masks,
                                                                                      device=device,
                                                                                      img_metas=img_metas)
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
                logger = get_root_logger()

                if not cfg["dataset"] == "GRefCOCO":
                    ACC_str_list = [
                        "{}Det@.5: {:.2f}, ".format(map_dict[i], det_acc_dict[map_dict[i]]) for i in range(len(predictions_list))
                    ]
                    ACC_str = "".join(ACC_str_list)
                    iou_str_list = ["{}_iou: {:.2f}, ".format(map_dict[i], mask_iou_dict[map_dict[i]]) for i in
                                    range(len(predictions_list))]
                    iou_str = "".join(iou_str_list)
                    logger.info(f"val - epoch [{epoch+1}]-[{batch+1}/{batches}] " + f"time: {(time.time()- end):.2f}, " + ACC_str + iou_str)
                    
                else:
                    F1_Score_str_list = [
                        "{}_f1_score: {:.2f}, ".format(map_dict[i], f1_score_acc_dict[map_dict[i]]) for i in range(len(predictions_list))
                    ]
                    n_acc_str_list = [
                        "{}_n_acc: {:.2f}, ".format(map_dict[i], n_acc_dict[map_dict[i]]) for i in range(len(predictions_list))
                    ]
                    giou_str_list = ["{}_giou: {:.2f}, ".format(map_dict[i], mask_giou_dict[map_dict[i]]) for i in
                                     range(len(predictions_list))]
                    ciou_str_list = ["{}_ciou: {:.2f}, ".format(map_dict[i], mask_ciou_dict[map_dict[i]]) for i in
                                     range(len(predictions_list))]
                    F1_Score_str = "".join(F1_Score_str_list)
                    n_acc_str = "".join(n_acc_str_list)
                    giou_str = "".join(giou_str_list)
                    ciou_str = "".join(ciou_str_list)
                    logger.info(
                        f"Validate - epoch [{epoch+1}]-[{batch+1}/{batches}] "
                        + f"time: {(time.time()- end):.2f}, "
                        + F1_Score_str
                        + n_acc_str
                        + giou_str
                        + ciou_str
                    )
            

            end = time.time()
    
    if not cfg["dataset"] == "GRefCOCO":
        det_acc = sum(list(det_acc_dict.values())) / len(det_acc_dict)
        mask_iou = sum(list(mask_iou_dict.values())) / len(mask_iou_dict)

    else:
        det_acc = sum(list(f1_score_acc_dict.values())) / len(f1_score_acc_dict)
        mask_iou = sum(list(mask_giou_dict.values())) / len(mask_giou_dict)


    return det_acc, mask_iou
