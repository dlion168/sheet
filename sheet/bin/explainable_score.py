#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import numpy as np
import torch
import yaml

from tqdm import tqdm
import concurrent.futures
import torchaudio
import torch.nn.functional as F

# === Captum 相關 ===
from captum.attr import (
    IntegratedGradients,
    Saliency,
    InputXGradient,
    LRP,
    GuidedGradCam,
    ShapleyValueSampling,
    Occlusion,
)
from lime.lime_tabular import LimeTabularExplainer

# === 您原本的程式中引用的模組 (示例) ===
from sheet.datasets import NonIntrusiveDataset
from sheet.models import SSLMOS, SSLMOS_wrap

# -----------------------------------------------------------------
# Global variables used in each worker (loaded once via initializer)
# -----------------------------------------------------------------
_global_model = None
_global_config = None
_global_device = None
_global_baseline = None

# ----------------------------------------
# Helper function: Min-Max Scaling
# ----------------------------------------
def minmax_scale(x: np.ndarray):
    """Scales x to [0, 1] range. If x is constant, returns x as-is to avoid zero division."""
    x_min = x.min()
    x_max = x.max()
    if abs(x_max - x_min) < 1e-9:
        return x
    return (x - x_min) / (x_max - x_min)

def match_length(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    將 waveform (1-D) 以「置中」的方式，pad 或 trim 至 target_len 長度。
    waveform shape = [time], 不包含 batch 維度。
    回傳 shape = [target_len].
    """
    cur_len = waveform.shape[0]
    if cur_len == target_len:
        return waveform
    elif cur_len < target_len:
        diff = target_len - cur_len
        left = diff // 2
        right = diff - left
        waveform = F.pad(waveform, (left, right), mode="constant", value=0)
        return waveform
    else:
        diff = cur_len - target_len
        left = diff // 2
        start = left
        end = start + target_len
        return waveform[start:end]

# ----------------------------------------
# Grad-CAM Class
# ----------------------------------------
class GradCAM:
    def __init__(self, model, target_layer_idx):
        self.model = model
        self.target_layer_idx = target_layer_idx
        self.gradients = None
        self.activation = None

        def save_gradients(grad):
            self.gradients = grad

        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                main_output = output[0]
            else:
                main_output = output
            self.activation = main_output
            main_output.register_hook(save_gradients)

        # Register hook to the target layer
        self.model.ssl_model.hooks = [
            self.model.ssl_model.upstream.model.encoder.layers[self.target_layer_idx]
            # self.model.ssl_model.upstream.model.feature_extractor.conv_layers[target_layer_idx]
            .register_forward_hook(forward_hook)
        ]

    def compute_cam(self, inputs):
        outputs = self.model.mean_net_inference(inputs)
        scores = outputs["scores"]  # 假設 key 為 "scores"
        target = torch.sum(scores)

        self.model.zero_grad()
        target.backward()

        pooled_gradients = torch.mean(self.gradients, dim=(0, 1, 2))  # [feat_dim]
        activation = self.activation.squeeze(0).squeeze(1).detach()   # [time, feat_dim]

        # Multiply activation by the mean gradient
        activation = activation * pooled_gradients
        cam = torch.mean(activation, dim=1)  # [time]

        # Interpolate up to the length of the input waveform
        cam_unsqueezed = cam.unsqueeze(0)  # [1, time]
        cam_interpolated = F.interpolate(
            cam_unsqueezed.unsqueeze(1),
            size=(inputs["waveform"].shape[1]),
            mode='linear',
            align_corners=False
        ).squeeze(1)  # [1, time]

        return cam_interpolated

# ----------------------------------------
# Attention (轉成 1D) 類
# ----------------------------------------
class AttentionVisualizer:
    def __init__(self, model, target_layer_idx):
        self.model = model
        self.target_layer_idx = target_layer_idx
        self.attentions = None

        def attn_hook(module, input, output):
            # output = (x, (attn_weights, ...))
            if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], tuple):
                attn = output[1][0]  # [batch, heads, time, time]
                self.attentions = attn.detach()

        # Register hook
        self.model.ssl_model.upstream.model.encoder.layers[self.target_layer_idx].register_forward_hook(attn_hook)

    def compute_1d_attention(self):
        """
        將 2D attentions (time x time) 簡化成 1D:
        先對 heads 取平均 -> [1, time, time],
        再對 key dim 取平均 -> [time].
        """
        avg_attn = torch.mean(self.attentions, dim=1).squeeze(0)  # [time, time]
        attn_1d = torch.mean(avg_attn, dim=1)                     # [time]
        return attn_1d

# ----------------------------------------
# Explanation Wrappers
# ----------------------------------------

def integrated_gradients_explanation(model, inputs, baseline_waveform):
    inputs["waveform"].requires_grad_(True)
    def model_forward(waveform):
        temp_inputs = {"waveform": waveform, "waveform_lengths": inputs["waveform_lengths"]}
        out = model.mean_net_inference(temp_inputs)["scores"]  # shape [1]
        return out.unsqueeze(-1)
    ig = IntegratedGradients(model_forward)
    attributions = ig.attribute(inputs["waveform"], baselines=baseline_waveform)
    return attributions

def saliency_explanation(model, inputs):
    inputs["waveform"].requires_grad_(True)
    def model_forward(waveform):
        temp_inputs = {"waveform": waveform, "waveform_lengths": inputs["waveform_lengths"]}
        scores = model.mean_net_inference(temp_inputs)["scores"]  # [1]
        return scores.unsqueeze(-1)
    sal = Saliency(model_forward)
    attributions = sal.attribute(inputs["waveform"])
    return attributions

def inputxgradient_explanation(model, inputs):
    inputs["waveform"].requires_grad_(True)
    def model_forward(waveform):
        temp_inputs = {"waveform": waveform, "waveform_lengths": inputs["waveform_lengths"]}
        scores = model.mean_net_inference(temp_inputs)["scores"]
        return scores.unsqueeze(-1)
    ixg = InputXGradient(model_forward)
    attributions = ixg.attribute(inputs["waveform"])
    return attributions

def lrp_explanation(model, inputs):
    from captum.attr import LRP
    from captum.attr._core.lrp import EpsilonRule
    import torch.nn as nn

    class ForwardWrapper(nn.Module):
        def __init__(self, actual_model):
            super().__init__()
            self.actual_model = actual_model
        
        def forward(self, x):
            wave_len = x.size(1)
            wave_len_tensor = torch.tensor([wave_len], dtype=torch.long, device=x.device)
            temp_inputs = {"waveform": x, "waveform_lengths": wave_len_tensor}
            out = self.actual_model.mean_net_inference(temp_inputs)["scores"]  # shape [1]
            return out.unsqueeze(-1)  # => [1,1]

    wrapped_model = ForwardWrapper(model)
    lrp = LRP(wrapped_model)
    # Register a default rule for Conv1d
    lrp.register_lrp_rules({nn.Conv1d: EpsilonRule()})
    attributions = lrp.attribute(inputs["waveform"])
    return attributions

def lime_explanation(model, inputs):
    waveform = inputs["waveform"].detach().cpu().numpy()[0]
    def predict_fn(samples):
        batch_scores = []
        for s in samples:
            wf_tensor = torch.from_numpy(s).unsqueeze(0).float().to(next(model.parameters()).device)
            wf_len = torch.tensor([len(s)], dtype=torch.long).to(next(model.parameters()).device)
            temp_inputs = {"waveform": wf_tensor, "waveform_lengths": wf_len}
            with torch.no_grad():
                sc = model.mean_net_inference(temp_inputs)["scores"].cpu().numpy()
            batch_scores.append(sc[0])
        return np.array(batch_scores).reshape(-1, 1)

    explainer = LimeTabularExplainer(
        training_data=waveform.reshape(1, -1),
        mode="regression"
    )
    explanation = explainer.explain_instance(
        data_row=waveform,
        predict_fn=predict_fn,
        num_features=10
    )
    return explanation.as_list()

# --- REMOVED shap_explanation ---

def shapley_value_sampling_explanation(model, inputs, baseline=0.0, n_samples=50):
    from captum.attr import ShapleyValueSampling

    def model_forward(waveform):
        temp_inputs = {"waveform": waveform, "waveform_lengths": inputs["waveform_lengths"]}
        scores = model.mean_net_inference(temp_inputs)["scores"]
        return scores.unsqueeze(-1)

    shapley = ShapleyValueSampling(model_forward)
    attributions = shapley.attribute(
        inputs["waveform"],
        baselines=baseline,
        n_samples=n_samples
    )
    return attributions

def occlusion_explanation(
    model, inputs,
    sliding_window=16,
    stride=8,
    baseline=0.0
):
    from captum.attr import Occlusion

    def model_forward(waveform):
        temp_inputs = {"waveform": waveform, "waveform_lengths": inputs["waveform_lengths"]}
        scores = model.mean_net_inference(temp_inputs)["scores"]
        return scores.unsqueeze(-1)

    occlusion = Occlusion(model_forward)
    attributions = occlusion.attribute(
        inputs=inputs["waveform"],
        sliding_window_shapes=(sliding_window,),
        strides=(stride,),
        baselines=baseline
    )
    return attributions

# ----------------------------------------
# Utilities for removing/keeping top k% attributions
# ----------------------------------------
def remove_top_k_percent(waveform: torch.Tensor, attributions: torch.Tensor, k=20.0):
    wf_clone = waveform.clone()
    attr_np = attributions.detach().cpu().numpy()
    abs_attr = np.abs(attr_np)
    threshold = np.percentile(abs_attr, 100 - k)
    mask = (abs_attr >= threshold)
    wf_clone[mask] = 0.0
    return wf_clone

def keep_top_k_percent(waveform: torch.Tensor, attributions: torch.Tensor, k=20.0):
    wf_clone = waveform.clone()
    attr_np = attributions.detach().cpu().numpy()
    abs_attr = np.abs(attr_np)
    threshold = np.percentile(abs_attr, 100 - k)
    mask = (abs_attr >= threshold)
    wf_clone[~mask] = 0.0
    return wf_clone

# ------------------------------------------------------------------
# 1) Worker initializer
# ------------------------------------------------------------------
def init_worker(model_state_dict_path, config_dict, device_str, baseline_wav_np):
    """
    Called exactly once in each worker. We load the model and baseline
    into global variables.
    """
    global _global_model, _global_config, _global_device, _global_baseline

    import torch
    from sheet.models import SSLMOS

    device = torch.device(device_str)

    # Build & load model
    model = SSLMOS(config_dict["model_input"], **config_dict["model_params"]).to(device)
    model.load_state_dict(torch.load(model_state_dict_path, map_location=device)["model"])
    model.eval()

    # Move baseline to device if provided
    if baseline_wav_np is not None:
        baseline_tensor = torch.from_numpy(baseline_wav_np).float().to(device)
    else:
        baseline_tensor = None

    _global_model = model
    _global_config = config_dict
    _global_device = device
    _global_baseline = baseline_tensor

# ------------------------------------------------------------------
# 2) Worker function for each sample
# ------------------------------------------------------------------
def process_single_sample(
    i, batch_data, method, topk_percent, shapley_n_samples, occlusion_window_size, occlusion_stride
):
    """
    Called for each dataset sample. Reuses the globally loaded model & baseline.
    Returns (remove_diff, keep_diff).
    """
    global _global_model, _global_config, _global_device, _global_baseline

    from sheet.bin.explainable_score import (
        GradCAM, integrated_gradients_explanation,
        saliency_explanation, inputxgradient_explanation,
        lrp_explanation, shapley_value_sampling_explanation,
        occlusion_explanation, AttentionVisualizer
    )

    # Prepare input
    waveform = batch_data["waveform"].unsqueeze(0).to(_global_device)
    waveform_lengths = torch.tensor([waveform.size(1)], dtype=torch.long).to(_global_device)
    inputs = {"waveform": waveform, "waveform_lengths": waveform_lengths}

    outputs = _global_model.mean_net_inference(inputs)
    pred_mos = outputs["scores"].cpu().item()
    wave_len = waveform.size(1)

    # Compute attributions based on method
    if method == "gradcam":
        cam = GradCAM(_global_model, target_layer_idx=-1)
        attributions = cam.compute_cam(inputs)
    elif method == "ig":
        if _global_baseline is not None:
            baseline_waveform = _global_baseline.unsqueeze(0)
        else:
            baseline_waveform = torch.zeros_like(waveform)
        attributions = integrated_gradients_explanation(_global_model, inputs, baseline_waveform)
    elif method == "saliency":
        attributions = saliency_explanation(_global_model, inputs)
    elif method == "attention":
        attn_viz = AttentionVisualizer(_global_model, target_layer_idx=-1)
        attn_1d = attn_viz.compute_1d_attention()
        attributions = attn_1d.unsqueeze(0)
    elif method == "lime":
        # Just random for demonstration
        attributions = torch.rand(1, wave_len, device=_global_device)
    elif method == "inputxgradient":
        attributions = inputxgradient_explanation(_global_model, inputs)
    elif method == "lrp":
        attributions = lrp_explanation(_global_model, inputs)
    elif method == "random":
        attributions = torch.rand(1, wave_len, device=_global_device)
    elif method == "shapley_sampling":
        baseline_value = _global_baseline if _global_baseline is not None else 0.0
        attributions = shapley_value_sampling_explanation(
            _global_model, inputs, baseline=baseline_value, n_samples=shapley_n_samples
        )
    elif method == "occlusion":
        baseline_value = _global_baseline if _global_baseline is not None else 0.0
        attributions = occlusion_explanation(
            _global_model,
            inputs,
            sliding_window=occlusion_window_size,
            stride=occlusion_stride,
            baseline=baseline_value
        )
    else:
        attributions = torch.zeros_like(waveform)

    # Remove / Keep top k%
    removed_waveform = remove_top_k_percent(waveform, attributions, k=topk_percent)
    removed_inputs = {"waveform": removed_waveform, "waveform_lengths": waveform_lengths}
    removed_mos = _global_model.mean_net_inference(removed_inputs)["scores"].cpu().item()
    remove_diff = removed_mos - pred_mos

    kept_waveform = keep_top_k_percent(waveform, attributions, k=topk_percent)
    kept_inputs = {"waveform": kept_waveform, "waveform_lengths": waveform_lengths}
    kept_mos = _global_model.mean_net_inference(kept_inputs)["scores"].cpu().item()
    keep_diff = kept_mos - pred_mos

    return remove_diff, keep_diff

# ------------------------------------------------------------------
# 4) Main
# ------------------------------------------------------------------
def main():
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  # Required to avoid CUDA fork issues

    parser = argparse.ArgumentParser(description="Inference with explainability.")
    parser.add_argument("--csv-path", required=True, type=str, help="csv file path.")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file.")
    parser.add_argument("--config", default=None, type=str, help="Config file path.")
    parser.add_argument(
        "--explain-method", 
        type=str, 
        choices=[
            "gradcam", 
            "ig", 
            "saliency", 
            "attention", 
            "lime", 
            "inputxgradient", 
            "random", 
            "lrp",
            "shapley_sampling",
            "occlusion",
        ],
        help="Explainability method."
    )
    parser.add_argument(
        "--baseline-wav", 
        type=str, 
        default=None, 
        help="Path to a reference audio file as baseline for IG / shapley / occlusion, etc."
    )
    parser.add_argument(
        "--topk-percent", 
        type=float,
        default=20.0,
        help="Percentage for top-k attribution threshold (default=20)."
    )
    parser.add_argument(
        "--shapley-n-samples",
        type=int,
        default=50,
        help="Number of permutations for ShapleyValueSampling"
    )
    parser.add_argument(
        "--occlusion-window-size",
        type=int,
        default=640,
        help="Sliding window size for Occlusion (default=320)."
    )
    parser.add_argument(
        "--occlusion-stride",
        type=int,
        default=320,
        help="Stride for Occlusion (default=160)."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for multi-processing. Default=4."
    )
    args = parser.parse_args()
    torch.cuda.manual_seed(1234)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # === Load dataset ===
    dataset = NonIntrusiveDataset(
        csv_path=args.csv_path,
        target_sample_rate=config_dict["sampling_rate"],
        model_input=config_dict["model_input"],
        wav_only=True,
        allow_cache=False,
    )

    # Possibly load baseline
    baseline_wav_np = None
    if args.baseline_wav is not None:
        if not os.path.isfile(args.baseline_wav):
            raise FileNotFoundError(f"Baseline audio not found: {args.baseline_wav}")
        logging.info(f"Loading baseline wav: {args.baseline_wav}")
        baseline_wav, sr_baseline = torchaudio.load(args.baseline_wav, channels_first=False)
        if sr_baseline != config_dict["sampling_rate"]:
            logging.info(f"Resampling baseline from {sr_baseline} to {config_dict['sampling_rate']}...")
            resampler = torchaudio.transforms.Resample(sr_baseline, config_dict["sampling_rate"])
            baseline_wav = resampler(baseline_wav)
        baseline_wav = baseline_wav.squeeze(-1).float()
        baseline_wav_np = baseline_wav.cpu().numpy()

    model_state_dict_path = args.checkpoint

    logging.info("Starting multi-process inference...")

    remove_diffs = []
    keep_diffs = []

    total_samples = len(dataset)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(model_state_dict_path, config_dict, device_str, baseline_wav_np)
    ) as executor:

        futures = []
        for i, batch_data in enumerate(dataset):
            fut = executor.submit(
                process_single_sample,
                i,
                batch_data,
                args.explain_method,
                args.topk_percent,
                args.shapley_n_samples,
                args.occlusion_window_size,
                args.occlusion_stride
            )
            futures.append(fut)

        # Show global progress as tasks complete
        with tqdm(total=total_samples, desc="Global Progress") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                remove_diff, keep_diff = fut.result()
                remove_diffs.append(remove_diff)
                keep_diffs.append(keep_diff)
                pbar.update(1)  # increment for each completed sample

    avg_abs_diff_remove = np.mean([abs(d) for d in remove_diffs]) if remove_diffs else 0.0
    avg_abs_diff_keep = np.mean([abs(d) for d in keep_diffs]) if keep_diffs else 0.0

    logging.info(f"Average abs difference (Remove top k%): {avg_abs_diff_remove:.6f}")
    logging.info(f"Average abs difference (Keep top k%):   {avg_abs_diff_keep:.6f}")
    logging.info("All multi-process inference complete!")

if __name__ == "__main__":
    main()