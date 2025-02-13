import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgEnhance(nn.Module):
    def __init__(self, embed_dim=256, train_mean=False):
        super(ImgEnhance, self).__init__()
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        self.img_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        if train_mean:
            self.avg_enhance_score = nn.Parameter(torch.zeros(1))
        else:
            self.avg_enhance_score = 0

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def normalize(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / std
        return x + self.avg_enhance_score

    def forward(self, text_feat_filter, img_feat, pos_embed):

        pos_embed = pos_embed.permute(1, 0, 2)
        img_feat = self.with_pos_embed(img_feat, pos_embed)
        text_feat = self.text_proj(text_feat_filter)
        img_feat = self.img_proj(img_feat)
        img_feat = self.dropout1(img_feat)
        text_feat = self.dropout2(text_feat)

        text_feat = F.normalize(text_feat, p=2, dim=-1)
        _img_feat = F.normalize(img_feat, p=2, dim=-1)
        score = torch.bmm(_img_feat, text_feat.transpose(1, 2))
        score = self.normalize(score)
        score = 2 * torch.sigmoid(score)
        img_feat = score * img_feat

        return img_feat, score


class Vltvg(nn.Module):
    def __init__(self, embed_dim=256, dropout=0.1):
        super().__init__()
        self.img2text_attn = nn.MultiheadAttention(embed_dim, 8, dropout=dropout)
        self.img_query_with_pos = True

        self.text_proj = nn.Linear(embed_dim, embed_dim)
        self.img_proj = nn.Linear(embed_dim, embed_dim)
        self.tf_pow = 2.0
        self.tf_scale = nn.Parameter(torch.Tensor([1.0]))
        self.tf_sigma = nn.Parameter(torch.Tensor([0.5]))

        self.img2textcond_attn = nn.MultiheadAttention(embed_dim, 8, dropout=dropout)

        self.img2img_attn = nn.MultiheadAttention(embed_dim, 8, dropout=dropout)

        self.norm_text_cond_img = nn.LayerNorm(embed_dim)
        self.norm_img = nn.LayerNorm(embed_dim)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat, img_pos,
                word_feat, word_key_padding_mask, word_pos=None):
        orig_img_feat = img_feat

        # visual-linguistic verification
        img_query = img_feat + img_pos if self.img_query_with_pos else img_feat
        text_info = self.img2text_attn(
            query=img_query, key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat, key_padding_mask=word_key_padding_mask)[0]

        text_embed = self.text_proj(text_info)
        img_embed = self.img_proj(img_feat)
        verify_score = (F.normalize(img_embed, p=2, dim=-1) *
                        F.normalize(text_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)

        # language-guided context encoder
        text_cond_info = self.img2textcond_attn(
            query=img_feat, key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat, key_padding_mask=word_key_padding_mask)[0]

        q = k = img_feat + text_cond_info
        text_cond_img_ctx = self.img2img_attn(
            query=q, key=k, value=img_feat)[0]

        # discriminative feature
        fuse_img_feat = (self.norm_img(img_feat) +
                         self.norm_text_cond_img(text_cond_img_ctx)) * verify_score

        return fuse_img_feat, verify_score