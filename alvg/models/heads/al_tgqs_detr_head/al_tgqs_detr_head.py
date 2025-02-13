from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from detrex.layers.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou
from detrex.utils import get_world_size, is_dist_avail_and_initialized
from detrex.layers.position_embedding import PositionEmbeddingLearned, PositionEmbeddingSine
from detrex.modeling.matcher.matcher import HungarianMatcher
from detectron2.structures import Boxes, ImageList, Instances
from alvg.models.utils import freeze_params
from .transformer import DetrTransformer, DetrTransformerEncoder, DetrTransformerDecoder
from alvg.models import HEADS
from alvg.core.criterion.distill_criterion import DistillCriterion
from alvg.models.heads.utils import PositionEmbeddingSine1D, MLP
from alvg.core.criterion.criterion import SetCriterion
from .img_enhance import ImgEnhance
import cv2
# import numpy as np
from .mask_head import Easy_Mask_Head


@HEADS.register_module()
class TextGuidedQuerySelectKDDETRHead(nn.Module):
    def __init__(
            self,
            num_queries=100,
            in_channels=768,
            text_max_token=20,
            embed_dim=256,
            num_classes=1,
            aux_loss=True,
            num_encoder_layers=6,
            num_decoder_layers=6,
            num_tgqg_layers=1,
            only_decoder=False,
            text_embed_aug=False,
            branch_loss_weight={},
            as_target_query_thr=0.0,
            distill_type="",  # "hard", "hard_weighted", "soft"
            decoder_freeze=False,
            prepare_target_mode="score_weighted",  # "score_weighted", "score_iou_weighted"
            share_predicthead=False,
            num_token_mlp_layers=3,
            mlp_aux_loss=False,
            tgqs_mid_dim=512,
            aux_distill_mode="klloss",  # "klloss" "smoothl1loss"
            text_guided_query_generation=False,
            score_attn_loss_weight=0.0,
            decoder_attn_loss_weight=0.0,
            img_enhance=False,
    ):
        super(TextGuidedQuerySelectKDDETRHead, self).__init__()
        self.transformer = DetrTransformer(
            encoder=None,
            decoder=None,
            embed_dim=embed_dim,
            only_decoder=only_decoder,
            img_enhance=img_enhance,
            num_decoder_layers=num_decoder_layers,
        )
        assert prepare_target_mode in ["score_weighted", "score_iou_weighted"]
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.input_text_proj = nn.Linear(in_channels, embed_dim)
        self.num_queries = num_queries
        self.text_embed_aug = text_embed_aug
        self.as_target_query_thr = as_target_query_thr
        self.prepare_target_mode = prepare_target_mode
        self.mlp_aux_loss = mlp_aux_loss
        self.text_guided_query_generation = text_guided_query_generation
        self.branch_loss_weight = branch_loss_weight
        # self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=embed_dim // 2,
            temperature=10000,
            normalize=True,
        )
        self.position_embedding_1d = PositionEmbeddingSine1D(
            num_pos_feats=embed_dim // 2,
            temperature=10000,
            normalize=True,
        )
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        # define classification head and box head
        if share_predicthead:
            self.class_embed_decoder = nn.Linear(embed_dim, num_classes + 1)
            self.bbox_embed_decoder = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
            self.class_embed_token = self.class_embed_decoder
            self.bbox_embed_token = self.bbox_embed_decoder
        else:
            self.class_embed_decoder = nn.Linear(embed_dim, num_classes + 1)
            self.bbox_embed_decoder = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)
            self.class_embed_token = nn.Linear(embed_dim, num_classes + 1)
            self.bbox_embed_token = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3)

        if text_guided_query_generation:
            self.text_guided_query_generation_transformer = DetrTransformerDecoder(
                embed_dim=embed_dim,
                num_heads=8,
                attn_dropout=0.1,
                feedforward_dim=tgqs_mid_dim,
                ffn_dropout=0.1,
                num_layers=num_tgqg_layers,
                return_intermediate=False,
                post_norm=True,
            )

        self.matcher = HungarianMatcher(
            cost_class=1,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="ce_cost",
        )
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=self.matcher,
            weight_dict={
                "loss_class": 1,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_mask": 15.0,
                "loss_dice": 6.0,
            },
            loss_class_type="ce_loss",
            eos_coef=0.1,
        )
        self.mask_head = Easy_Mask_Head(embed_dim)

        if self.aux_loss:
            weight_dict = self.criterion.weight_dict
            aux_weight_dict = {}
            for i in range(self.transformer.decoder.num_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
            self.criterion.weight_dict = weight_dict

        self.score_attn_loss_weight = score_attn_loss_weight
        self.decoder_attn_loss_weight = decoder_attn_loss_weight

        if decoder_freeze:
            self.transformer.eval()
            freeze_params(self.transformer)
            freeze_params(self.input_proj)
            freeze_params(self.text_guided_query_generation_proj)
            freeze_params(self.input_text_proj)
            freeze_params(self.class_embed_decoder)
            freeze_params(self.box)

    def prepare_targets(self, targets, img_metas):
        new_targets = []
        for target_bbox, img_meta in zip(targets, img_metas):
            h, w = img_meta["img_shape"][:2]
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=target_bbox.device)
            if len(target_bbox.shape) == 1:
                target_bbox = target_bbox.unsqueeze(0)
                gt_classes = torch.zeros(1, device=target_bbox.device).long()
            else:  # for grec # TODO None object can be set as label 1 ? or just set no GT
                assert int(target_bbox.shape[0]) == len(img_meta["target"])
                gt_classes = torch.tensor([1 if t["category_id"] == -1 else 0 for t in img_meta["target"]],
                                          device=target_bbox.device).long()
            gt_boxes = target_bbox.float() / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def prepare_soft_targets(self, targets, decoder_branch_output, img_metas, gt_masks, predict_threahold=0.0,
                             prepare_target_mode="iou_weighted"):
        new_targets_pred = []
        new_targets_gt = []
        # decoder_branch_output_new = deepcopy(decoder_branch_output)
        decoder_pred_logits = decoder_branch_output["pred_logits"].detach().data
        decoder_pred_boxes = decoder_branch_output["pred_boxes"].detach().data
        decoder_scores = F.softmax(decoder_pred_logits, dim=-1)[:, :, 0:1].detach().data
        # decoder_scores = F.softmax(decoder_branch_output["pred_logits"], dim=-1)
        for target_bbox, gt_mask, img_meta in zip(targets, gt_masks, img_metas):
            h, w = img_meta["img_shape"][:2]
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=target_bbox.device)
            if len(target_bbox.shape) == 1:
                target_bbox_ = target_bbox.unsqueeze(0)
                gt_classes = torch.zeros(1, device=target_bbox_.device).long()
            else:  # for grec
                assert int(target_bbox.shape[0]) == len(img_meta["target"])
                gt_classes = []
                target_bbox_ = torch.zeros((0, 4), device=target_bbox.device)
                for ind, t in enumerate(img_meta["target"]):
                    if t["category_id"] != -1:
                        gt_classes.append(0)
                        target_bbox_ = torch.concat((target_bbox_, target_bbox[ind:ind + 1]))
                gt_classes = torch.tensor(gt_classes, device=target_bbox.device).long()
                # gt_classes = torch.tensor([1 if t["category_id"] == -1 else 0 for t in img_meta["target"]], device=target_bbox.device).long()
                # target_bbox_ = target_bbox
            gt_boxes = target_bbox_.float() / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes).float()
            if gt_mask is not None:
                gt_mask = deepcopy(gt_mask)

                gt_mask = torch.any(gt_mask, dim=0).float()
                gt_mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0), size=(h // 4, w // 4), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).int()

            new_targets_gt.append({"labels": gt_classes, "boxes": gt_boxes, "masks": gt_mask})
        #
        # # use all predict
        # if prepare_target_mode == "score_weighted":
        #     for predict_bbox, predict_score, img_meta in zip(decoder_pred_boxes, decoder_scores, img_metas):
        #         mask = predict_score.squeeze(-1) > predict_threahold
        #         predict_weight = torch.zeros_like(predict_score)
        #         predict_weight[mask] = predict_score[mask]
        #         predict_bbox_ = predict_bbox[mask, :]
        #         if sum(mask) == 0:
        #             gt_classes = torch.tensor([], device=target_bbox.device).long()
        #         else:
        #             gt_classes = torch.zeros((predict_bbox_.shape[0]), device=target_bbox.device).long()
        #         new_targets_pred.append({"labels": gt_classes, "boxes": predict_bbox_, "weight": predict_weight})
        # elif prepare_target_mode == "score_iou_weighted":  # iou weighted distill
        #     new_targets_gt_out = deepcopy(new_targets_gt)
        #     decoder_pred_boxes = deepcopy(decoder_pred_boxes)
        #     decoder_scores = deepcopy(decoder_scores)
        #     indices = self.matcher(decoder_branch_output, new_targets_gt_out)
        #     for indice, predict_bbox, predict_score, target_gt, img_meta in zip(indices, decoder_pred_boxes,
        #                                                                         decoder_scores, new_targets_gt_out,
        #                                                                         img_metas):
        #         predict_bbox_ = predict_bbox[indice[0]]
        #         target_gt_ = target_gt["boxes"][indice[1]]
        #         target_gt_ = torch.cat([target_gt_], dim=0)
        #         ious = torch.diag(box_iou(box_cxcywh_to_xyxy(predict_bbox_), box_cxcywh_to_xyxy(target_gt_))[0])
        #         predict_score_ = predict_score[indice[0]].reshape(-1)
        #         predict_weight = predict_score_ * ious
        #         if predict_weight.shape[0] == 0:
        #             gt_classes = torch.tensor([], device=target_bbox.device).long()
        #         else:
        #             gt_classes = torch.zeros((predict_bbox_.shape[0]), device=target_bbox.device).long()
        #         new_targets_pred.append({"labels": gt_classes, "boxes": predict_bbox_, "weight": predict_weight})
        # else:
        #     raise TypeError(
        #         "{} type is not support yet!! you can choose [score_weighted, iou_weighted] types!!!".format(
        #             prepare_target_mode))

        return new_targets_gt, new_targets_pred

    def prepare_merge_target(self, targets, decoder_branch_output, img_metas):
        new_targets_merge = []
        new_targets_gt = []
        decoder_scores = F.softmax(decoder_branch_output["pred_logits"], dim=-1)[:, :, 0]
        # decoder_scores = F.softmax(decoder_branch_output["pred_logits"], dim=-1)
        decoder_pred_boxes = decoder_branch_output["pred_boxes"]

        decoder_scores = decoder_scores.detach()
        decoder_pred_boxes = decoder_pred_boxes.detach()

        for target_bbox, img_meta in zip(targets, img_metas):
            h, w = img_meta["img_shape"][:2]
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=target_bbox.device)
            if len(target_bbox.shape) == 1:
                target_bbox = target_bbox.unsqueeze(0)
                gt_classes = torch.zeros(1, device=target_bbox.device).long()
            else:  # for grec
                assert int(target_bbox.shape[0]) == len(img_meta["target"])
                gt_classes = torch.tensor([1 if t["category_id"] == -1 else 0 for t in img_meta["target"]],
                                          device=target_bbox.device).long()
            gt_boxes = target_bbox.float() / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets_gt.append({"labels": gt_classes, "boxes": gt_boxes})

        new_targets_gt_out = deepcopy(new_targets_gt)
        decoder_pred_boxes = deepcopy(decoder_pred_boxes)
        decoder_scores = deepcopy(decoder_scores)
        indices = self.matcher(decoder_branch_output, new_targets_gt_out)
        for indice, predict_bbox, predict_score, target_gt, img_meta in zip(indices, decoder_pred_boxes, decoder_scores,
                                                                            new_targets_gt_out, img_metas):
            predict_bbox_ = predict_bbox[indice[0]]
            target_gt_boxes = target_gt["boxes"][indice[1]]
            target_gt_labels = target_gt["labels"][indice[1]]
            target_gt_boxes = torch.cat([target_gt_boxes], dim=0)
            ious = torch.diag(box_iou(box_cxcywh_to_xyxy(predict_bbox_), box_cxcywh_to_xyxy(target_gt_boxes))[0])
            predict_score_ = predict_score[indice[0]]
            predict_weight = predict_score_ * ious
            merged_labels = torch.cat((target_gt_labels, gt_classes), dim=0)
            merged_bboxes = torch.cat((target_gt_boxes, predict_bbox_), dim=0)
            merged_weights = torch.cat(
                (torch.ones(target_gt_labels.shape[0], device=predict_weight.device), predict_weight), dim=0)
            if predict_weight.shape[0] == 0:
                gt_classes = torch.tensor([], device=target_bbox.device).long()
            else:
                gt_classes = torch.zeros((predict_bbox_.shape[0]), device=target_bbox.device).long()
            new_targets_merge.append({"labels": merged_labels, "boxes": merged_bboxes, "weight": merged_weights})

        return new_targets_gt, new_targets_merge

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def x_mask_pos_enc(self, x, img_metas):
        batch_size = x.size(0)
        try:
            input_img_h, input_img_w = img_metas[0]["batch_input_shape"]
        except:
            input_img_h, input_img_w, _ = img_metas[0]["img_shape"]
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        # CAUTION: do not support random flipping
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"]
            x_mask[img_id, :img_h, :img_w] = 0

        x_mask = F.interpolate(x_mask.unsqueeze(1), size=x.size()[-2:]).to(torch.bool).squeeze(1)

        x_pos_embeds = self.position_embedding(x_mask)

        return x_mask, x_pos_embeds

    def calc_loss(self, output_class, output_coord, pred_masks, targets):
        output = {"pred_logits": output_class[-1], "pred_boxes": output_coord[-1], "pred_masks": pred_masks}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(output_class, output_coord)

        loss_dict = self.criterion(output, targets)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict

    def calc_attn_loss(self, score, attn_map, gt_bbox, output_class, output_coord, targets_gt,
                      use_decoder_loss=True, only_last_match=True, gt_mask=None):
        """
        Calculate the loss for the attention map and relative region
        Args:
            score: Language-Image Relevance Score, shape [B, H, W]
            attn_map: Attention Map from decoder, shape [num_layers, B, num_queries, H, W]
            gt_bbox: shape [B, 4]
        """
        num_layers, bs, num_queries, h, w = attn_map.shape
        device = attn_map.device
        output = {"pred_logits": output_class[-1], "pred_boxes": output_coord[-1]}
        indices = []
        outputs_aux = self._set_aux_loss(output_class, output_coord)
        last_match = self.matcher(output, targets_gt)

        # query_no_target_mask = torch.ones(num_layers, bs, num_queries, device=device, dtype=torch.float64)
        for i, aux_outputs in enumerate(outputs_aux):
            if not only_last_match:
                match = self.matcher(aux_outputs, targets_gt)
            else:
                match = last_match
            indices.append(match)
            # for j, (pred_idx, target_idx) in enumerate(match):
            #     if pred_idx.numel() > 0:
            #         query_no_target_mask[i][j, pred_idx] = 0

        indices.append(last_match)

        score_loss = torch.tensor(0, device=device).float()
        decoder_loss = torch.tensor(0, device=device).float()

        focus_maps = []
        overlap_maps = []
        for i in range(bs):
            grec = True
            num_bboxes = gt_bbox[i].size(0)
            if gt_bbox[0].dim() == 1:
                num_bboxes = 1
                grec = False
            mask_maps = []
            overlap_map = torch.zeros(h, w, dtype=torch.float32, device=device)

            for j in range(num_bboxes):
                if gt_mask[0] is None:
                    map_ = torch.zeros(640, 640, dtype=torch.float32, device=device)
                    if grec:
                        bbox = gt_bbox[i][j]
                    else:
                        bbox = gt_bbox[i]
                    x1, y1, x2, y2 = (torch.floor(bbox[0]).to(torch.int),
                                      torch.floor(bbox[1]).to(torch.int),
                                      torch.ceil(bbox[2]).to(torch.int),
                                      torch.ceil(bbox[3]).to(torch.int))
                    map_[y1:y2 + 1, x1:x2 + 1] = 1.0
                    map_ = F.interpolate(map_.unsqueeze(0).unsqueeze(0), size=(h, w), mode='area').squeeze(0).squeeze(0)
                    area = torch.sum(map_)
                    overlap_map = torch.max(overlap_map, map_ / area)
                else:
                    mask = gt_mask[i][j].float()
                    mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(h, w), mode='area').squeeze(0).squeeze(0)
                    areas = torch.sum(mask)

                    overlap_map = torch.max(overlap_map, mask / areas)
                    mask[mask > 0] = 1
                    map_ = mask

                mask_maps.append(map_)
            if num_bboxes > 0:
                overlap_map /= num_bboxes
            overlap_area = torch.count_nonzero(overlap_map)
            if h * w > overlap_area:
                overlap_map[overlap_map == 0] = -1.0 / (h * w - overlap_area)
            if targets_gt[i]['boxes'].numel() == 0:
                overlap_map = -torch.ones(h, w, dtype=torch.float32, device=device) / (h * w)
            focus_maps.append(mask_maps)
            overlap_maps.append(overlap_map)

        overlap_maps = torch.stack(overlap_maps)
        # no_target_overlap_maps = overlap_maps.repeat(num_layers, 1, 1, 1).unsqueeze(2).repeat(1, 1, num_queries, 1, 1)

        if score is not None:
            score_loss -= torch.sum(overlap_maps * score) / bs

        if use_decoder_loss:
            if num_queries == 1:
                focus_maps = [focus_map[0] for focus_map in focus_maps]
                focus_maps = torch.stack(focus_maps)
                all_attn_map = attn_map.sum(dim=0)
                all_attn_map = all_attn_map.squeeze(1)
                in_box_weight = torch.sum(all_attn_map * focus_maps, dim=(1, 2))
                all_weight = torch.sum(all_attn_map, dim=(1, 2))
                decoder_loss = (all_weight - in_box_weight) / (in_box_weight + 0.001)
                decoder_loss = torch.sum(decoder_loss) / (bs * num_layers)
            else:
                num_all_queries = torch.tensor(0, device=device).float()
                for indices_, attn_map_ in zip(indices, attn_map):
                    for i in range(bs):
                        pred_idx, target_idx = indices_[i]
                        if pred_idx.numel() == 0:
                            continue
                        pred_idx = pred_idx.tolist()
                        target_idx = target_idx.tolist()
                        box_map = [focus_maps[i][j] for j in target_idx]
                        num_all_queries += len(box_map)
                        for j in range(len(box_map)):
                            in_box_weight = torch.sum(attn_map_[i][pred_idx[j]] * box_map[j])
                            all_weight = torch.sum(attn_map_[i][pred_idx[j]])
                            decoder_loss += (all_weight - in_box_weight) / (in_box_weight + 0.001)
                            # decoder_loss -= torch.sum(attn_map_[i][pred_idx[j]] * box_map[j])
                # no_target_loss = no_target_overlap_maps * attn_map
                # no_target_loss = no_target_loss.sum(dim=(-2, -1))
                # no_target_loss = torch.sum(no_target_loss * query_no_target_mask)
                # decoder_loss -= no_target_loss * 0.3

                num_all_queries = torch.as_tensor([num_all_queries], dtype=torch.float, device=device)
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(num_all_queries)
                num_all_queries = torch.clamp(num_all_queries / get_world_size(), min=1).item()
                if num_all_queries > 0:
                    decoder_loss = decoder_loss / num_all_queries
        return decoder_loss, score_loss

    def forward_general(self, x_mm, img_metas, cls_feat=None, text_feat=None, text_mask=None):
        # feature proj to embed channels
        x_mm = self.input_proj(x_mm)
        text_feat = self.input_text_proj(text_feat)
        img_masks, pos_embed = self.x_mask_pos_enc(x_mm, img_metas)  # TODO: fix the img mask

        text_pos_embed = (self.position_embedding_1d(text_feat).unsqueeze(0).
                          repeat(text_feat.shape[0], 1, 1).permute(1, 0, 2).cuda())
        # text guided query generation
        if self.text_guided_query_generation:
            text_feat_filter = torch.cat(list(
                map(lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0], text_feat,
                    ~text_mask))).unsqueeze(1).repeat(1, self.num_queries, 1)
            query_embed_input = self.query_embed.weight.unsqueeze(0).repeat(x_mm.shape[0], 1, 1).transpose(0, 1)
            target = torch.zeros_like(query_embed_input)
            text_feat_input = text_feat.transpose(0, 1)
            query_embed = self.text_guided_query_generation_transformer(
                query=target,
                key=text_feat_input,
                value=text_feat_input,
                key_pos=text_pos_embed,
                query_pos=query_embed_input,
                key_padding_mask=text_mask.bool())
            query_embed = query_embed[0].transpose(0, 1) + text_feat_filter + query_embed_input.transpose(0, 1)
        else:
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(x_mm.shape[0], 1, 1)

        hidden_states, memory, score, attn_map = self.transformer(x_mm, img_masks, query_embed, pos_embed,
                                                                  text_feat_filter[:, :1, :], text_feat_input, text_mask.bool(), text_pos_embed)

        outputs_class_decoder_branch = self.class_embed_decoder(hidden_states)
        outputs_coord_decoder_branch = self.bbox_embed_decoder(hidden_states).sigmoid()
        pred_masks = self.mask_head(memory, pos_embed)

        # pred_masks = self.mask_head(hidden_states[-1], pos_embed, memory)

        decoder_branch_output = {
            "pred_logits": outputs_class_decoder_branch[-1],
            "pred_boxes": outputs_coord_decoder_branch[-1],
            "pred_masks": pred_masks,
        }

        output = {
            "pred_masks": pred_masks,
            "decoder_branch_output": decoder_branch_output,
            "outputs_class_decoder_branch": outputs_class_decoder_branch,
            "outputs_coord_decoder_branch": outputs_coord_decoder_branch,
            "decoder_features": hidden_states,
            "x_mm": x_mm,
            "img_masks": img_masks,
            "score": score,
            "attn_map": attn_map,
        }

        return output

    def forward_train(self, x_mm, img_metas, cls_feat=None, text_feat=None, gt_bbox=None, gt_mask=None, text_mask=None):
        if gt_mask is None:
            gt_mask = [None] * len(img_metas)
        device = x_mm.device
        output = self.forward_general(x_mm, img_metas, cls_feat=cls_feat, text_feat=text_feat, text_mask=text_mask)

        outputs_class_decoder_branch = output["outputs_class_decoder_branch"]
        outputs_coord_decoder_branch = output["outputs_coord_decoder_branch"]
        decoder_branch_output = output["decoder_branch_output"]
        pred_masks = output["pred_masks"]
        score = output["score"]
        attn_map = output["attn_map"]

        # prepare the targets
        # targets_decoder_branch = self.prepare_targets(gt_bbox, img_metas)
        # print(decoder_branch_output)
        targets_gt, targets_predict = self.prepare_soft_targets(
            gt_bbox, decoder_branch_output, img_metas, gt_mask, predict_threahold=self.as_target_query_thr,
            prepare_target_mode=self.prepare_target_mode
        )

        decoder_attention_loss, loss_decoder_branch, score_attention_loss, loss_kd_branch, loss_merge_branch, loss_aux_distill_branch = (
            torch.tensor(0, device=device).float(),
            torch.tensor(0, device=device).float(),
            torch.tensor(0, device=device).float(),
            torch.tensor(0, device=device).float(),
            torch.tensor(0, device=device).float(),
            torch.tensor(0, device=device).float(),
        )
        loss_dict = {}
        if "decoder" in self.branch_loss_weight:
            loss_dict_decoder_branch = self.calc_loss(outputs_class_decoder_branch, outputs_coord_decoder_branch, pred_masks, targets_gt)
            loss_mask = (loss_dict_decoder_branch.pop("loss_mask", torch.tensor([0.0], device=device)) +
                         loss_dict_decoder_branch.pop("loss_dice", torch.tensor([0.0], device=device)))

            loss_decoder_branch = sum(loss_dict_decoder_branch.values())
            loss_decoder_branch = self.branch_loss_weight["decoder"] * loss_decoder_branch
            loss_dict["loss_dgt"] = loss_decoder_branch
            loss_dict["loss_mask"] = loss_mask

        if self.decoder_attn_loss_weight + self.score_attn_loss_weight > 0:
            decoder_attention_loss, score_attention_loss = self.calc_attn_loss(score, attn_map, gt_bbox,
                                                                              outputs_class_decoder_branch,
                                                                              outputs_coord_decoder_branch, targets_gt, gt_mask=gt_mask)
            decoder_attention_loss *= self.decoder_attn_loss_weight
            score_attention_loss *= self.score_attn_loss_weight
            loss_dict["decoder_attention_loss"], loss_dict["score_attention_loss"] = (
                decoder_attention_loss, score_attention_loss)

        loss_dict["loss_total"] = loss_decoder_branch + score_attention_loss + decoder_attention_loss
        return loss_dict, output

    def forward_test(self, x_mm, img_metas, text_feat=None, cls_feat=None, with_bbox=False, with_mask=False,
                     text_mask=None):
        return self.forward_general(x_mm, img_metas, text_feat=text_feat, cls_feat=cls_feat, text_mask=text_mask)

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
                zip(scores, labels, box_pred, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

