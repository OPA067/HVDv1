import os
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from types import SimpleNamespace
import torch
from torch import nn

from .cluster import Att_Block_Patch, PCM
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .module_pts import PatchTokenSelection
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, KL

allgather = AllGather.apply
allgather2 = AllGather2.apply

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(
            nn.Linear(d_int, d_int),
            nn.ReLU(inplace=True),
            nn.Linear(d_int, d_int), )

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x

class Model(nn.Module):
    def __init__(self, config):

        super(Model, self).__init__()

        self.config = config

        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        if torch.cuda.is_available():
            convert_weights(self.clip)

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn(config)
        self.loss_kl = KL(config)

        self.apply(self.init_weights)  # random init must before loading pretrain
        self.clip.load_state_dict(state_dict, strict=False)

        embed_dim = state_dict["text_projection"].shape[1]
        self.max_words, self.max_frames = self.config.max_words, self.config.max_frames
        self.h_max_frames = self.max_frames // 2
        self.alpha, self.beta = self.config.alpha, self.config.beta

        sr_p = [0.5, 0.5, 0.5]
        self.v_pcm_p_1 = PCM(sample_ratio=sr_p[0], embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_1 = Att_Block_Patch(dim=embed_dim, num_heads=8)
        self.v_pcm_p_2 = PCM(sample_ratio=sr_p[1], embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_2 = Att_Block_Patch(dim=embed_dim, num_heads=8)
        self.v_pcm_p_3 = PCM(sample_ratio=sr_p[2], embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_3 = Att_Block_Patch(dim=embed_dim, num_heads=8)

        # HVD
        self.s_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.w_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.f_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.p_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )

        ## ===> Initialization trick [HARD CODE]
        new_state_dict = OrderedDict()

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, text, text_mask, video, video_mask, idx=None, global_step=0):

        text_mask = text_mask.view(-1, text_mask.shape[-1])
        text = text.view(-1, text.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        s_feat, w_feat, f_feat, p_feat = self.get_text_video_feat(text, text_mask, video, video_mask, shaped=True)
        w_mask, f_mask = text_mask, video_mask
        s_feat, w_feat, w_mask = s_feat.contiguous(), w_feat.contiguous(), w_mask.contiguous()
        f_feat, p_feat, f_mask = f_feat.contiguous(), p_feat.contiguous(), f_mask.contiguous()

        logit_scale = self.clip.logit_scale.exp()

        if self.training:
            s_feat = allgather(s_feat, self.config)
            w_feat = allgather(w_feat, self.config)
            w_mask = allgather(w_mask, self.config)
            f_feat = allgather(f_feat, self.config)
            p_feat = allgather(p_feat, self.config)
            f_mask = allgather(f_mask, self.config)
            torch.distributed.barrier()

        a, s, w, d = s_feat.size(0), 1,              w_feat.size(1), w_feat.size(-1)
        b, f, p, d = f_feat.size(0), f_feat.size(1), p_feat.size(1), p_feat.size(-1)

        ###### Step-I: See all ######
        sims_sf = torch.einsum("ad,bfd->abf", [self.norm(s_feat), self.norm(f_feat)])
        sims_sf = sims_sf.diagonal(dim1=0, dim2=1).transpose(0, 1)
        sims_sf = torch.softmax(sims_sf, dim=-1)
        _, f_max_idx = torch.topk(sims_sf, k=self.h_max_frames, dim=-1, largest=True)
        f_max_idx, _ = torch.sort(f_max_idx, dim=-1)
        f_feat_h = f_feat[torch.arange(b)[:, None], f_max_idx, :]
        sims_sf_h = self.s_and_f(s_feat, f_feat_h)
        loss_sf_h = (self.loss_fct(sims_sf_h * logit_scale) + self.loss_fct(sims_sf_h.T * logit_scale)) / 2.0
        sims_wf_h = self.w_and_f(w_feat, f_feat_h)
        loss_wf_h = (self.loss_fct(sims_wf_h * logit_scale) + self.loss_fct(sims_wf_h.T * logit_scale)) / 2.0

        ###### Step-II: See one ######
        p_feat_h = p_feat.reshape(b, f, -1, d)[torch.arange(b)[:, None], f_max_idx, :, :]
        p_feat_h = p_feat_h.reshape(b, -1, d)
        p_feat_h = self.get_less_patch_feat(p_feat_h)
        sims_sp_h = self.s_and_p(s_feat, p_feat_h)
        loss_sp_h = (self.loss_fct(sims_sp_h * logit_scale) + self.loss_fct(sims_sp_h.T * logit_scale)) / 2.0
        sims_wp_h = self.w_and_p(w_feat, p_feat_h)
        loss_wp_h = (self.loss_fct(sims_wp_h * logit_scale) + self.loss_fct(sims_wp_h.T * logit_scale)) / 2.0

        ###### Step-III: See all & one ######
        p_feat = self.get_less_patch_feat(p_feat)
        sims_sf = self.s_and_f(s_feat, f_feat)
        loss_sf = (self.loss_fct(sims_sf * logit_scale) + self.loss_fct(sims_sf.T * logit_scale)) / 2.0
        sims_wf = self.w_and_f(w_feat, f_feat)
        loss_wf = (self.loss_fct(sims_wf * logit_scale) + self.loss_fct(sims_wf.T * logit_scale)) / 2.0
        sims_sp = self.s_and_p(s_feat, p_feat)
        loss_sp = (self.loss_fct(sims_sp * logit_scale) + self.loss_fct(sims_sp.T * logit_scale)) / 2.0
        sims_wp = self.w_and_p(w_feat, p_feat)
        loss_wp = (self.loss_fct(sims_wp * logit_scale) + self.loss_fct(sims_wp.T * logit_scale)) / 2.0

        ###### Step-IV: KL loss
        loss_kl_sf = (self.loss_kl(sims_sf, sims_sf_h) + self.loss_kl(sims_sf.T, sims_sf_h.T)) / 2.0
        loss_kl_wf = (self.loss_kl(sims_wf, sims_wf_h) + self.loss_kl(sims_wf.T, sims_wf_h.T)) / 2.0
        loss_kl_sp = (self.loss_kl(sims_sp, sims_sp_h) + self.loss_kl(sims_sp.T, sims_sp_h.T)) / 2.0
        loss_kl_wp = (self.loss_kl(sims_wp, sims_wp_h) + self.loss_kl(sims_wp.T, sims_wp_h.T)) / 2.0

        ##### Step-V: Total loss
        total_loss_h  = (loss_sf_h + loss_wf_h + loss_sp_h + loss_wp_h) / 4.0
        total_loss_o  = (loss_sf + loss_wf + loss_sp + loss_wp) / 4.0
        total_loss_kl = (loss_kl_sf + loss_kl_wf + loss_kl_sp + loss_kl_wp) / 4.0

        total_loss = total_loss_h + total_loss_o + total_loss_kl

        if self.training:
            return total_loss
        else:
            return None

    def agg_f_feat(self, f_feat, f_mask, agg_module):
        f_feat = f_feat.contiguous()
        if agg_module == "None":
            pass
        elif agg_module == "seqLSTM":
            f_feat_original = f_feat
            f_feat = pack_padded_sequence(f_feat, torch.sum(f_mask, dim=-1).cpu(), batch_first=True, enforce_sorted=False)
            f_feat, _ = self.lstm_visual(f_feat)
            if self.training:
                self.lstm_visual.flatten_parameters()
            f_feat, _ = pad_packed_sequence(f_feat, batch_first=True)
            f_feat = torch.cat((f_feat, f_feat_original[:, f_feat.size(1):, ...].contiguous()), dim=1)
            f_feat = f_feat + f_feat_original
        elif agg_module == "seqTransf":
            f_feat_original = f_feat
            seq_length = f_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=f_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(f_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            f_feat = f_feat + frame_position_embeddings
            extended_f_mask = (1.0 - f_mask.unsqueeze(1)) * -1000000.0
            extended_f_mask = extended_f_mask.expand(-1, f_mask.size(1), -1)
            f_feat = f_feat.permute(1, 0, 2)
            f_feat = self.transformerClip(f_feat, extended_f_mask)
            f_feat = f_feat.permute(1, 0, 2)
            f_feat = f_feat + f_feat_original

        return f_feat

    def norm(self, feat):
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def s_and_f(self, s_feat, f_feat):
        f_w = torch.softmax(self.f_feat_w(f_feat).squeeze(-1), dim=-1)
        sims_sf = torch.einsum("ad,bfd->abf", [self.norm(s_feat), self.norm(f_feat)])
        sims_sf = torch.einsum("abf,bf->ab", [sims_sf, f_w])
        return sims_sf

    def w_and_f(self, w_feat, f_feat):
        w_w = torch.softmax(self.w_feat_w(w_feat).squeeze(-1), dim=-1)
        f_w = torch.softmax(self.f_feat_w(f_feat).squeeze(-1), dim=-1)
        sims_wf = torch.einsum("awd,bfd->abwf", [self.norm(w_feat), self.norm(f_feat)])
        sims_w2f, _ = sims_wf.max(dim=-1)
        sims_w2f = torch.einsum('abw,aw->ab', [sims_w2f, w_w])
        sims_f2w, _ = sims_wf.max(dim=-2)
        sims_f2w = torch.einsum('abf,bf->ab', [sims_f2w, f_w])
        sims_wf = (sims_w2f + sims_f2w) / 2.0
        return sims_wf

    def s_and_p(self, s_feat, p_feat):
        p_w = torch.softmax(self.p_feat_w(p_feat).squeeze(-1), dim=-1)
        sims_sp = torch.einsum("ad,bpd->abp", [self.norm(s_feat), self.norm(p_feat)])
        sims_sp = torch.einsum("abp,bp->ab", [sims_sp, p_w])
        return sims_sp

    def w_and_p(self, w_feat, p_feat):
        w_w = torch.softmax(self.w_feat_w(w_feat).squeeze(-1), dim=-1)
        p_w = torch.softmax(self.p_feat_w(p_feat).squeeze(-1), dim=-1)
        sims_wp = torch.einsum("awd,bpd->abwp", [self.norm(w_feat), self.norm(p_feat)])
        sims_w2p, _ = sims_wp.max(dim=-1)
        sims_w2p = torch.einsum('abw,aw->ab', [sims_w2p, w_w])
        sims_p2w, _ = sims_wp.max(dim=-2)
        sims_p2w = torch.einsum('abf,bf->ab', [sims_p2w, p_w])
        sims_wp = (sims_w2p + sims_p2w) / 2.0

        return sims_wp

    def get_less_patch_feat(self, p_feat):
        p_idx_token = torch.arange(p_feat.size(1))[None, :].repeat(p_feat.size(0), 1)
        p_agg_weight = p_feat.new_ones(p_feat.size(0), p_feat.size(1), 1)
        p_mask = p_feat.new_ones(p_feat.size(0), p_feat.size(1))
        p_token_dict = {'x': p_feat,
                        'token_num': p_feat.size(1),
                        'idx_token': p_idx_token,
                        'agg_weight': p_agg_weight,
                        'mask': p_mask.detach()}
        p_token_dict = self.v_att_block_p_1(self.v_pcm_p_1(p_token_dict))
        p_token_dict = self.v_att_block_p_2(self.v_pcm_p_2(p_token_dict))
        p_token_dict = self.v_att_block_p_3(self.v_pcm_p_3(p_token_dict))
        p_feat = p_token_dict['x']

        return p_feat

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        s_feat, w_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        s_feat = s_feat.float().view(bs_pair, s_feat.size(-1))
        w_feat = w_feat.float().view(bs_pair, -1, w_feat.size(-1))
        return s_feat, w_feat

    def get_video_feat(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        f_feat, p_feat = self.clip.encode_image(video, return_hidden=True, mask=video_mask)
        f_feat = f_feat.float().view(bs_pair, -1, f_feat.size(-1))
        f_feat = self.agg_f_feat(f_feat, video_mask, self.agg_module)
        p_feat = p_feat.float().view(bs_pair, -1, p_feat.size(-1))

        return f_feat, p_feat

    def get_text_video_feat(self, text, text_mask, video, video_mask, shaped=False):
        if shaped is False:
            text = text.view(-1, text.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        s_feat, w_feat = self.get_text_feat(text, text_mask, shaped=True)
        f_feat, p_feat = self.get_video_feat(video, video_mask, shaped=True)

        return s_feat, w_feat, f_feat, p_feat

    def get_similarity_logits(self, s_feat, w_feat, w_mask, f_feat, p_feat, f_mask):
        p_feat = self.get_less_patch_feat(p_feat)
        sims_sf = self.s_and_f(s_feat, f_feat)
        sims_wf = self.w_and_f(w_feat, f_feat)
        sims_sp = self.s_and_p(s_feat, p_feat)
        sims_wp = self.w_and_p(w_feat, p_feat)
        sims = (sims_sf + sims_wf + sims_sp + sims_wp) / 4.0

        return sims

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples
            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
