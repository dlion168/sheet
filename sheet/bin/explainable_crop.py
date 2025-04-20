#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import numpy as np
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

# === Captum 相關 ===
from captum.attr import IntegratedGradients, Saliency
from lime.lime_tabular import LimeTabularExplainer
import shap

# === 您原本的程式中引用的模組 (示例) ===
from sheet.datasets import NonIntrusiveDataset
from sheet.models import SSLMOS

# ----------------------------------------
# Helper function: Min-Max Scaling
# ----------------------------------------
def minmax_scale(x: np.ndarray):
    """Scales x to [0, 1] range. If x is constant, returns x as-is to avoid zero division."""
    x_min = x.min()
    x_max = x.max()
    if abs(x_max - x_min) < 1e-9:
        return x  # Avoid division by zero for constant arrays
    return (x - x_min) / (x_max - x_min)

# ----------------------------------------
# NEW: Crop-out function
# ----------------------------------------
def crop_waveform_by_amplitude(
    waveform_np: np.ndarray,
    sr: int,
    amp_threshold: float = 0.001,
    min_time: float = 0.01
) -> np.ndarray:
    """
    Remove (crop out) any contiguous segment where |waveform_np| > amp_threshold
    for longer than min_time seconds.
    
    Example:
      - amp_threshold = 0.001
      - min_time = 0.01 (seconds)
    Meaning we find places where the amplitude is "too large" for >0.01s, then remove them.
    """
    sample_thresh = int(sr * min_time)  # Number of samples for 0.01s

    # Boolean mask: loud samples
    loud_mask = np.abs(waveform_np) > amp_threshold

    # We'll keep everything initially, then set to False for segments to remove
    keep_mask = np.ones_like(waveform_np, dtype=bool)

    idx = 0
    n = len(waveform_np)

    while idx < n:
        if not loud_mask[idx]:
            # We've encountered a "loud" sample, find how long the loud segment is
            start = idx
            while idx < n and not loud_mask[idx]:
                idx += 1
            end = idx  # end is first index after that loud segment
            seg_length = end - start
            # If the loud segment is longer than the threshold, remove it
            if seg_length > sample_thresh:
                keep_mask[start:end] = False
        else:
            idx += 1

    # Return the cropped waveform
    cropped_wf = waveform_np[keep_mask]
    return cropped_wf

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
            .register_forward_hook(forward_hook)
        ]

    def compute_cam(self, inputs):
        outputs = self.model.mean_net_inference(inputs)
        scores = outputs["scores"]
        target = torch.sum(scores)

        self.model.zero_grad()
        target.backward()

        pooled_gradients = torch.mean(self.gradients, dim=(0, 1, 2))  # [feat_dim]
        activation = self.activation.squeeze(0).squeeze(1).detach()   # [time, feat_dim]

        activation = activation * pooled_gradients
        cam = torch.mean(activation, dim=1).clamp(min=0)              # [time]
        return cam

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
# IG, Saliency, LIME, SHAP
# ----------------------------------------
def integrated_gradients_explanation(model, inputs, baseline_waveform):
    inputs["waveform"].requires_grad_(True)
    def model_forward(waveform):
        temp_inputs = {"waveform": waveform, "waveform_lengths": inputs["waveform_lengths"]}
        out = model.mean_net_inference(temp_inputs)["scores"]  # shape [1]
        return torch.mean(out).unsqueeze(0)
    ig = IntegratedGradients(model_forward)
    attributions = ig.attribute(inputs["waveform"], baselines=baseline_waveform)
    return attributions

def saliency_explanation(model, inputs):
    inputs["waveform"].requires_grad_(True)
    def model_forward(waveform):
        temp_inputs = {"waveform": waveform, "waveform_lengths": inputs["waveform_lengths"]}
        scores = model.mean_net_inference(temp_inputs)["scores"]
        return torch.mean(scores).unsqueeze(0)
    saliency = Saliency(model_forward)
    attributions = saliency.attribute(inputs["waveform"])
    return attributions

def lime_explanation(model, inputs, sampling_rate=16000):
    waveform = inputs["waveform"].detach().cpu().numpy()[0]  # [time]
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

def shap_explanation(model, inputs):
    waveform = inputs["waveform"].detach().cpu().numpy()[0]  # [time]
    def model_forward_for_shap(waveform_batch):
        device = next(model.parameters()).device
        results = []
        for w in waveform_batch:
            wf_tensor = torch.from_numpy(w).unsqueeze(0).float().to(device)
            wf_len = torch.tensor([wf_tensor.shape[1]], dtype=torch.long).to(device)
            temp_inputs = {"waveform": wf_tensor, "waveform_lengths": wf_len}
            with torch.no_grad():
                sc = model.mean_net_inference(temp_inputs)["scores"].cpu().numpy()
            results.append(sc[0])
        return np.array(results).reshape(-1, 1)

    import shap
    explainer = shap.KernelExplainer(model_forward_for_shap, data=waveform.reshape(1, -1))
    shap_values = explainer.shap_values(waveform.reshape(1, -1), nsamples=100)
    return shap_values

# ----------------------------------------
# Main
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inference with explainability.")
    parser.add_argument("--csv-path", required=True, type=str, help="csv file path.")
    parser.add_argument("--outdir", required=True, type=str, help="Output directory.")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file.")
    parser.add_argument("--config", default=None, type=str, help="Config file path.")
    parser.add_argument(
        "--explain-method", 
        type=str, 
        choices=["gradcam", "ig", "saliency", "attention", "lime", "shap"],
        help="Explainability method."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load dataset ===
    dataset = NonIntrusiveDataset(
        csv_path=args.csv_path,
        target_sample_rate=config["sampling_rate"],
        model_input=config["model_input"],
        wav_only=True,
        allow_cache=False,
    )

    # === Load model ===
    model = SSLMOS(config["model_input"], **config["model_params"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()

    # === Possibly initialize GradCAM/Attention if needed ===
    gradcam = GradCAM(model, target_layer_idx=-1) if args.explain_method == "gradcam" else None
    attention_viz = AttentionVisualizer(model, target_layer_idx=-1) if args.explain_method == "attention" else None

    logging.info("Starting inference...")

    sr = config["sampling_rate"]

    for i, batch in enumerate(tqdm(dataset)):

        # 取得 wave 及 length
        waveform = batch["waveform"].unsqueeze(0).to(device)  # [1, time]
        # Cropping is best done in numpy
        waveform_np = waveform[0].cpu().numpy()

        # -- CROP THE WAVEFORM FIRST --
        cropped_np = crop_waveform_by_amplitude(
            waveform_np, 
            sr=sr, 
            amp_threshold=0.005, 
            min_time=0.01
        )

        # If everything is cropped out, skip
        if len(cropped_np) == 0:
            logging.info(f"Sample {i}: All data removed by threshold. Skipping.")
            continue

        # Now convert back to torch for inference
        cropped_waveform = torch.from_numpy(cropped_np).unsqueeze(0).float().to(device)
        cropped_lengths = torch.tensor([cropped_waveform.size(1)], dtype=torch.long).to(device)
        inputs = {"waveform": cropped_waveform, "waveform_lengths": cropped_lengths}

        # Ground truth 分數 (依您 CSV/資料結構；這裡假設名為 "score")
        gt_mos = batch.get("score", None)  

        # 先做一次 mean_net_inference 拿預測分數
        outputs = model.mean_net_inference(inputs)
        pred_mos = outputs["scores"].cpu().item()

        fname = os.path.basename(batch["wav_path"])  # 檔名
        wave_len = len(cropped_np)
        waveform_scaled = minmax_scale(cropped_np)
        time_axis = np.arange(wave_len) / float(sr)

        # 主標題中加上 Ground Truth 與 Predicted
        if gt_mos is not None:
            title_str = f"{fname} (Sample {i})  |  GT: {gt_mos:.3f}, Pred: {pred_mos:.3f}"
        else:
            title_str = f"{fname} (Sample {i})  |  Pred: {pred_mos:.3f}"

        if args.explain_method == "gradcam":
            cam = gradcam.compute_cam(inputs)  # shape [time_cam]
            cam_np = cam.cpu().numpy()
            cam_len = len(cam_np)

            if cam_len > 1:
                t_cam = np.linspace(0, 1, cam_len)
                t_wave = np.linspace(0, 1, wave_len)
                cam_interp = np.interp(t_wave, t_cam, cam_np)
            else:
                cam_interp = np.full(wave_len, cam_np.item() if cam_len == 1 else 0.0)
            cam_scaled = minmax_scale(cam_interp)

            plt.figure(figsize=(10, 4))
            plt.plot(time_axis, waveform_scaled, label="Waveform (scaled)", alpha=0.7, color="blue")
            plt.plot(time_axis, cam_scaled, label="Grad-CAM (scaled)", alpha=0.8, color="red")
            plt.title(f"Grad-CAM + Waveform: {title_str}")
            plt.xlabel("Time (sec)")
            plt.ylabel("Scaled amplitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"gradcam_{i}.png"))
            plt.close()

        elif args.explain_method == "ig":
            baseline_waveform = torch.zeros_like(cropped_waveform)
            attributions = integrated_gradients_explanation(model, inputs, baseline_waveform)
            ig_np = attributions[0].detach().cpu().numpy()
            ig_scaled = minmax_scale(ig_np)

            plt.figure(figsize=(10, 4))
            plt.plot(time_axis, waveform_scaled, label="Waveform (scaled)", alpha=0.7, color="blue")
            plt.plot(time_axis, ig_scaled, label="IG (scaled)", alpha=0.8, color="red")
            plt.title(f"Integrated Gradients + Waveform: {title_str}")
            plt.xlabel("Time (sec)")
            plt.ylabel("Scaled amplitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"ig_{i}.png"))
            plt.close()

        elif args.explain_method == "saliency":
            attributions = saliency_explanation(model, inputs)
            saliency_np = attributions[0].cpu().numpy()
            saliency_scaled = minmax_scale(saliency_np)

            plt.figure(figsize=(10, 4))
            plt.plot(time_axis, waveform_scaled, label="Waveform (scaled)", alpha=0.7, color="blue")
            plt.plot(time_axis, saliency_scaled, label="Saliency (scaled)", alpha=0.8, color="red")
            plt.title(f"Saliency + Waveform: {title_str}")
            plt.xlabel("Time (sec)")
            plt.ylabel("Scaled amplitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"saliency_{i}.png"))
            plt.close()

        elif args.explain_method == "attention":
            attn_1d = attention_viz.compute_1d_attention()
            attn_1d_np = attn_1d.cpu().numpy()
            if len(attn_1d_np) != wave_len:
                t_attn = np.linspace(0, 1, len(attn_1d_np))
                t_wave = np.linspace(0, 1, wave_len)
                attn_1d_np = np.interp(t_wave, t_attn, attn_1d_np)
            attn_scaled = minmax_scale(attn_1d_np)

            plt.figure(figsize=(10, 4))
            plt.plot(time_axis, waveform_scaled, label="Waveform (scaled)", alpha=0.7, color="blue")
            plt.plot(time_axis, attn_scaled, label="Attention (scaled)", alpha=0.8, color="red")
            plt.title(f"Attention + Waveform: {title_str}")
            plt.xlabel("Time (sec)")
            plt.ylabel("Scaled amplitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"attention_{i}.png"))
            plt.close()

        elif args.explain_method == "lime":
            # LIME 解析
            explanation_list = lime_explanation(model, inputs, sampling_rate=sr)
            logging.info(f"Sample {i} [{fname}] LIME Explanation: {explanation_list}")

            # 此處示範: 疊加隨機曲線
            random_curve = np.random.rand(wave_len)
            random_curve_scaled = minmax_scale(random_curve)

            plt.figure(figsize=(10, 4))
            plt.plot(time_axis, waveform_scaled, label="Waveform (scaled)", alpha=0.7, color="blue")
            plt.plot(time_axis, random_curve_scaled, label="(Demo) LIME Weighted Curve", alpha=0.8, color="red")
            plt.title(f"LIME + Waveform: {title_str}")
            plt.xlabel("Time (sec)")
            plt.ylabel("Scaled amplitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"lime_{i}.png"))
            plt.close()

        elif args.explain_method == "shap":
            shap_values = shap_explanation(model, inputs)
            logging.info(f"Sample {i} [{fname}] SHAP Values shape: {np.array(shap_values).shape}")

            shap_np = shap_values[0][0]
            if len(shap_np) != wave_len:
                t_shap = np.linspace(0, 1, len(shap_np))
                t_wave = np.linspace(0, 1, wave_len)
                shap_np = np.interp(t_wave, t_shap, shap_np)
            shap_scaled = minmax_scale(shap_np)

            plt.figure(figsize=(10, 4))
            plt.plot(time_axis, waveform_scaled, label="Waveform (scaled)", alpha=0.7, color="blue")
            plt.plot(time_axis, shap_scaled, label="SHAP (scaled)", alpha=0.8, color="red")
            plt.title(f"SHAP + Waveform: {title_str}")
            plt.xlabel("Time (sec)")
            plt.ylabel("Scaled amplitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"shap_{i}.png"))
            plt.close()

        else:
            # No special explanation method
            logging.info(f"Sample {i} [{fname}]: GT = {gt_mos}, Pred = {pred_mos:.3f}")

    logging.info("Inference complete!")

if __name__ == "__main__":
    main()
