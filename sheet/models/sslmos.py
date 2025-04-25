# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

# SSLMOS model
# modified from: https://github.com/nii-yamagishilab/mos-finetune-ssl/blob/main/mos_fairseq.py (written by Erica Cooper)

import math

import torch
import torch.nn as nn
from sheet.modules.ldnet.modules import Projection
from sheet.modules.weighted_sum import WeightedSumLayer
from sheet.modules.utils import make_non_pad_mask
class SSLMOS(torch.nn.Module):
    def __init__(
        self,
        # dummy, for signature need
        model_input: str,
        # model related
        ssl_module: str = "s3prl",
        s3prl_name: str = "openai/whisper-large-v3",
        ssl_model_output_dim: int = 1280,
        ssl_num_layers: int = 13,
        ssl_model_layer_idx: int = -1,
        ssl_weighted_sum: bool = True,
        ssl_trainable: bool = True,
        masked_mean_pooling: bool = False,
        activation: str = "PReLU",
        # mean net related
        mean_net_dnn_dim: int = 64,
        mean_net_output_type: str = "scalar",
        mean_net_output_dim: int = 5,
        mean_net_output_step: float = 0.25,
        mean_net_range_clipping: bool = True,
        # listener related
        use_listener_modeling: bool = False,
        num_listeners: int = None,
        listener_emb_dim: int = None,
        use_mean_listener: bool = True,
        # sample rate related
        use_sample_rate_modeling: bool = True,
        sample_rate_emb_dim: int = 3,
        # decoder related
        decoder_type: str = "ffn",
        decoder_dnn_dim: int = 64,
        output_type: str = "scalar",
        range_clipping: bool = True,
        # dummy, for signature need
        num_domains: int = None,
    ):
        super().__init__()  # this is needed! or else there will be an error.
        self.use_mean_listener = use_mean_listener
        self.output_type = output_type
        self.ssl_trainable = ssl_trainable
        self.masked_mean_pooling = masked_mean_pooling

        # define listener embedding
        self.use_sample_rate_modeling = use_sample_rate_modeling

        # define ssl model
        if ssl_module == "s3prl":
            from s3prl.nn import S3PRLUpstream

            if s3prl_name in S3PRLUpstream.available_names():
                self.ssl_model = S3PRLUpstream(s3prl_name, refresh=False)
                if not self.ssl_trainable:
                    self.ssl_model.eval()
            self.ssl_model_layer_idx = ssl_model_layer_idx
        else:
            raise NotImplementedError

        self.ssl_weighted_sum = ssl_weighted_sum
        if ssl_weighted_sum:
            self.weighted_sum = WeightedSumLayer(ssl_num_layers, normalize=True)
            
        self.activation = eval(f"nn.{activation}")
        
                # listener modeling related
        self.use_listener_modeling = use_listener_modeling
        if use_listener_modeling:
            self.num_listeners = num_listeners
            self.listener_embeddings = nn.Embedding(
                num_embeddings=num_listeners, embedding_dim=listener_emb_dim
            )
        if use_sample_rate_modeling:
            # 3 種取樣率對應 3 個 idx
            self.sample_rate_embeddings = nn.Embedding(
                num_embeddings=3,
                embedding_dim=sample_rate_emb_dim,
            )

        mean_net_input_dim = ssl_model_output_dim
        if use_sample_rate_modeling:
            mean_net_input_dim += sample_rate_emb_dim
            
        # default uses ffn type mean net
        self.mean_net_dnn = Projection(
            mean_net_input_dim,
            mean_net_dnn_dim,
            self.activation,
            mean_net_output_type,
            mean_net_output_dim,
            mean_net_output_step,
            mean_net_range_clipping,
        )

        # --- decoder 定義，將 embedding 維度一起納入 ---
        # 計算 decoder 輸入維度
        decoder_input_dim = ssl_model_output_dim
        if use_listener_modeling:
            decoder_input_dim += listener_emb_dim
        if use_sample_rate_modeling:
            decoder_input_dim += sample_rate_emb_dim
        
        # define decoder
        self.decoder_type = decoder_type
        if decoder_type == "ffn":
            decoder_dnn_input_dim = decoder_input_dim
        else:
            raise NotImplementedError
        # there is always dnn
        self.decoder_dnn = Projection(
            decoder_dnn_input_dim,
            decoder_dnn_dim,
            self.activation,
            output_type,
            range_clipping,
        )

    def get_num_params(self):
        return sum(p.numel() for n, p in self.named_parameters())

    def forward(self, inputs):
        """Calculate forward propagation.
        Args:
            waveform has shape (batch, time)
            waveform_lengths has shape (batch)
            listener_ids has shape (batch)
        """
        waveform = inputs["waveform"]
        waveform_lengths = inputs["waveform_lengths"]

        # ssl model forward
        if self.ssl_trainable:
            all_encoder_outputs, all_encoder_outputs_lens = self.ssl_model(
                waveform, waveform_lengths
            )
        else:
            with torch.no_grad():
                all_encoder_outputs, all_encoder_outputs_lens = self.ssl_model(
                    waveform, waveform_lengths
                )
        if not self.ssl_weighted_sum:
            encoder_outputs = all_encoder_outputs[self.ssl_model_layer_idx]
            encoder_outputs_lens = all_encoder_outputs_lens[self.ssl_model_layer_idx]
        else:
            encoder_outputs = self.weighted_sum(all_encoder_outputs)
            encoder_outputs_lens = all_encoder_outputs_lens[-1]
        
        batch, time, _ = encoder_outputs.shape
        
        # prepare embeddings list
        emb_list = [encoder_outputs]
        
        # sample rate embedding
        if self.use_sample_rate_modeling:
            sample_rate_ids = inputs["sample_rate_idxs"]
            sr_embs = self.sample_rate_embeddings(sample_rate_ids)  # (batch, emb_dim)
            sr_embs = sr_embs.unsqueeze(1).expand(-1, time, -1)
            emb_list.append(sr_embs)

        # concatenate all features
        mean_net_inputs = torch.cat(emb_list, dim=-1)

       # listener embedding
        if self.use_listener_modeling:
            listener_ids = inputs["listener_idxs"]
            listener_embs = self.listener_embeddings(listener_ids)  # (batch, emb_dim)
            listener_embs = listener_embs.unsqueeze(1).expand(-1, time, -1)
            emb_list.append(listener_embs)

        # concatenate all features
        decoder_inputs = torch.cat(emb_list, dim=-1)

        if self.masked_mean_pooling:
            masks = make_non_pad_mask(encoder_outputs_lens)
            masks = masks.unsqueeze(-1).to(decoder_inputs.device) # [B, max_time, 1]
            decoder_inputs = torch.sum(decoder_inputs * masks, dim=1) / encoder_outputs_lens.unsqueeze(-1)

        # mean net
        mean_net_outputs = self.mean_net_dnn(
            mean_net_inputs
        )  # [batch, time, 1 (scalar) / 5 (categorical)]

        # decoder
        if self.use_listener_modeling:
            if self.decoder_type == "rnn":
                decoder_outputs, (h, c) = self.decoder_rnn(decoder_inputs)
            else:
                decoder_outputs = decoder_inputs
            decoder_outputs = self.decoder_dnn(
                decoder_outputs
            )  # [batch, time, 1 (scalar) / 5 (categorical)]

        # set outputs
        # return lengths for masked loss calculation
        ret = {
            "waveform_lengths": waveform_lengths,
            "frame_lengths": encoder_outputs_lens,
        }

        # define scores
        ret["mean_scores"] = mean_net_outputs
        ret["ld_scores"] = decoder_outputs if self.use_listener_modeling else None

        return ret

    def mean_net_inference(self, inputs):
        waveform = inputs["waveform"]
        waveform_lengths = inputs["waveform_lengths"]

        # ssl model forward
        all_encoder_outputs, all_encoder_outputs_lens = self.ssl_model(
            waveform, waveform_lengths
        )
        if not self.ssl_weighted_sum:
            encoder_outputs = all_encoder_outputs[self.ssl_model_layer_idx]
        else:
            encoder_outputs = self.weighted_sum(all_encoder_outputs)
        
        batch, time, _ = encoder_outputs.shape
        
        # prepare embeddings list
        emb_list = [encoder_outputs]
        
        # sample rate embedding
        if self.use_sample_rate_modeling:
            sample_rate_ids = inputs["sample_rate_idxs"]
            sr_embs = self.sample_rate_embeddings(sample_rate_ids)  # (batch, emb_dim)
            sr_embs = sr_embs.unsqueeze(1).expand(-1, time, -1)
            emb_list.append(sr_embs)

        # concatenate all features
        mean_net_inputs = torch.cat(emb_list, dim=-1)

        mean_net_outputs = self.mean_net_dnn(
            mean_net_inputs, inference=True
        )  # [batch, time, 1 (scalar) / 5 (categorical)]
        mean_net_outputs = mean_net_outputs.squeeze(-1)
        scores = torch.mean(mean_net_outputs, dim=1) # [batch]

        return {
            "ssl_embeddings": encoder_outputs,
            "scores": scores
        }

    def mean_net_inference_p1(self, waveform, waveform_lengths):
        # ssl model forward
        all_encoder_outputs, _ = self.ssl_model(waveform, waveform_lengths)
        encoder_outputs = all_encoder_outputs[self.ssl_model_layer_idx]
        return encoder_outputs

    def mean_net_inference_p2(self, encoder_outputs):
        # mean net
        mean_net_outputs = self.mean_net_dnn(
            encoder_outputs
        )  # [batch, time, 1 (scalar) / 5 (categorical)]
        mean_net_outputs = mean_net_outputs.squeeze(-1)
        scores = torch.mean(mean_net_outputs, dim=1)

        return scores

    def get_ssl_embeddings(self, inputs):
        waveform = inputs["waveform"]
        waveform_lengths = inputs["waveform_lengths"]

        all_encoder_outputs, all_encoder_outputs_lens = self.ssl_model(
            waveform, waveform_lengths
        )
        encoder_outputs = all_encoder_outputs[self.ssl_model_layer_idx]
        return encoder_outputs
        