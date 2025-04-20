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
import torchaudio
import torch.nn.functional as F

# === Captum 相關 ===
from captum.attr import IntegratedGradients, Saliency, InputXGradient
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
        # 需要 pad
        diff = target_len - cur_len
        left = diff // 2
        right = diff - left
        waveform = F.pad(waveform, (left, right), mode="constant", value=0)
        return waveform
    else:
        # 需要 trim
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
# Explanation methods
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

def inputxgradient_explanation(model, inputs):
    """
    Uses InputXGradient from captum, which multiplies the input by the gradient 
    of the output w.r.t. the input to measure feature importance.
    """
    inputs["waveform"].requires_grad_(True)

    def model_forward(waveform):
        temp_inputs = {"waveform": waveform, "waveform_lengths": inputs["waveform_lengths"]}
        scores = model.mean_net_inference(temp_inputs)["scores"]  # shape [1]
        return torch.mean(scores).unsqueeze(0)

    ixg = InputXGradient(model_forward)
    attributions = ixg.attribute(inputs["waveform"])
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
        choices=["gradcam", "ig", "saliency", "attention", "lime", "shap", "inputxgradient"],
        help="Explainability method."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

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
        # 1) 取得主要 waveform
        waveform = batch["waveform"].unsqueeze(0).to(device)  # shape [1, time]
        waveform_lengths = torch.tensor([waveform.size(1)], dtype=torch.long).to(device)
        inputs = {"waveform": waveform, "waveform_lengths": waveform_lengths}

        # 2) 取得 original_id: sample_id = {system_id}_{original_id}
        system_id = batch["system_id"]
        sample_id = batch["sample_id"]
        # 切掉前面 "system_id + _" 的部分後，剩下的就是 original_id
        # 假設 sample_id 一定符合此格式
        original_id = sample_id[len(system_id) + 1 :]  

        # 3) baseline rule: 在 wav_path 的父資料夾下找 "clean_{original_id}.wav"
        wav_dir = os.path.dirname(batch["wav_path"])
        baseline_path = os.path.join(wav_dir, f"clean_{original_id}.wav")

        baseline_waveform = None

        # 4) 若該檔案存在，載入當 baseline；否則用 zeros
        if os.path.exists(baseline_path):
            try:
                baseline_wav, sr_baseline = torchaudio.load(baseline_path, channels_first=False)
                if sr_baseline != sr:
                    resampler = torchaudio.transforms.Resample(sr_baseline, sr)
                    baseline_wav = resampler(baseline_wav)
                baseline_wav = baseline_wav.squeeze(-1).float().to(device)  # [time]
                # 與目標 waveform 做長度對齊
                wave_len = waveform.size(1)
                baseline_wav_matched = match_length(baseline_wav, wave_len)
                baseline_waveform = baseline_wav_matched.unsqueeze(0)  # [1, time]
            except Exception as e:
                logging.warning(f"Failed to load baseline from {baseline_path}, will use zeros. Error: {e}")
        if baseline_waveform is None:
            baseline_waveform = torch.zeros_like(waveform)  # fallback

        # 5) 推論 & 取得預測分數
        outputs = model.mean_net_inference(inputs)
        pred_mos = outputs["scores"].cpu().item()

        # 6) 取得繪圖所需資訊
        fname = os.path.basename(batch["wav_path"])  # 檔名
        waveform_np = waveform[0].cpu().numpy()
        wave_len = len(waveform_np)
        waveform_scaled = minmax_scale(waveform_np)
        time_axis = np.arange(wave_len) / float(sr)

        # 可能有 ground truth
        gt_mos = batch.get("avg_score", None)
        if gt_mos is not None:
            title_str = f"{fname} (Sample {i})  |  GT: {gt_mos:.3f}, Pred: {pred_mos:.3f}"
        else:
            title_str = f"{fname} (Sample {i})  |  Pred: {pred_mos:.3f}"

        # 7) 根據 explain_method，計算解釋結果
        if args.explain_method == "gradcam":
            cam = gradcam.compute_cam(inputs)
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
            # 使用上面動態找出的 baseline_waveform
            attributions = integrated_gradients_explanation(model, inputs, baseline_waveform)
            ig_np = attributions[0].detach().cpu().numpy()
            ig_scaled = minmax_scale(ig_np)

            plt.figure(figsize=(10, 4))
            plt.plot(time_axis, waveform_scaled, label="Waveform (scaled)", alpha=0.7, color="blue")
            plt.plot(time_axis, ig_scaled, label="IG (scaled)", alpha=0.8, color="red")
            plt.title(f"IG + Waveform: {title_str}")
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
            attn_1d = attention_viz.compute_1d_attention()  # [time]
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
            explanation_list = lime_explanation(model, inputs)
            logging.info(f"Sample {i} [{fname}] LIME Explanation: {explanation_list}")

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

        elif args.explain_method == "inputxgradient":
            attributions = inputxgradient_explanation(model, inputs)
            ixg_np = attributions[0].detach().cpu().numpy()
            ixg_scaled = minmax_scale(ixg_np)

            plt.figure(figsize=(10, 4))
            plt.plot(time_axis, waveform_scaled, label="Waveform (scaled)", alpha=0.7, color="blue")
            plt.plot(time_axis, ixg_scaled, label="InputXGradient (scaled)", alpha=0.8, color="red")
            plt.title(f"InputXGradient + Waveform: {title_str}")
            plt.xlabel("Time (sec)")
            plt.ylabel("Scaled amplitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"inputxgradient_{i}.png"))
            plt.close()

        else:
            logging.info(f"Sample {i} [{fname}]: GT = {gt_mos}, Pred = {pred_mos:.3f}")

    logging.info("Inference complete!")

if __name__ == "__main__":
    main()
