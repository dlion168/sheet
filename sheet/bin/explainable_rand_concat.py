#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import shap
from captum.attr import IntegratedGradients, Saliency, InputXGradient
from lime.lime_tabular import LimeTabularExplainer
import torchaudio
import torch.nn.functional as F

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

def match_length(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    用簡易方式將 wave 置中並 pad 或 trim 到 target_len 長度 (1D).
    """
    cur_len = wav.shape[0]
    if cur_len == target_len:
        return wav
    elif cur_len < target_len:
        diff = target_len - cur_len
        left = diff // 2
        right = diff - left
        return F.pad(wav, (left, right), mode="constant", value=0)
    else:
        diff = cur_len - target_len
        left = diff // 2
        start = left
        end = start + target_len
        return wav[start:end]

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
            if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], tuple):
                attn = output[1][0]  # [batch, heads, time, time]
                self.attentions = attn.detach()

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
        scores = model.mean_net_inference(temp_inputs)["scores"]
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
    parser = argparse.ArgumentParser(description="Inference with explainability on pairwise audios with the same original_id, using a clean baseline.")
    parser.add_argument("--csv-path", required=True, type=str, help="csv file path.")
    parser.add_argument("--outdir", required=True, type=str, help="Output directory.")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file.")
    parser.add_argument("--config", default=None, type=str, help="Config file path.")
    parser.add_argument(
        "--original-id",
        required=True,
        type=str,
        help="Specify the original_id (e.g., TMHINT_b2_21_06). Will concatenate all system_ids that share this original_id."
    )
    parser.add_argument(
        "--explain-method", 
        type=str, 
        choices=["gradcam", "ig", "saliency", "attention", "lime", "shap", "inputxgradient"],
        help="Explainability method."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

    # 讀 config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 載入 Dataset ===
    dataset = NonIntrusiveDataset(
        csv_path=args.csv_path,
        target_sample_rate=config["sampling_rate"],
        model_input=config["model_input"],
        wav_only=True,
        allow_cache=False,
    )
    logging.info(f"Dataset loaded with {len(dataset)} items.")

    # === 篩選出所有 sample_id 與指定 original_id 相同的項目 ===
    selected_indices = []
    for idx, item in enumerate(dataset):
        sample_id = item["sample_id"]
        # sample_id 為 {system_id}_{original_id}
        # 這裡以 endswith(args.original_id) 來判斷
        if sample_id.endswith(args.original_id):
            selected_indices.append(idx)

    logging.info(f"Found {len(selected_indices)} items matching original_id '{args.original_id}'.")

    if len(selected_indices) < 2:
        logging.error(f"Fewer than 2 items found for original_id={args.original_id}. Cannot concatenate.")
        return

    # 建立 pair 列表
    pair_list = []
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            pair_list.append((selected_indices[i], selected_indices[j]))
    logging.info(f"Total pairs to process: {len(pair_list)}")

    # === 載入模型 ===
    model = SSLMOS(config["model_input"], **config["model_params"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()

    # === (可選) 初始化 GradCAM/Attention ===
    gradcam = GradCAM(model, target_layer_idx=-1) if args.explain_method == "gradcam" else None
    attention_viz = AttentionVisualizer(model, target_layer_idx=-1) if args.explain_method == "attention" else None

    sr = config["sampling_rate"]

    # === 嘗試載入對應的 clean baseline 音檔：clean_{original_id}.wav
    #     假設 pair_list 中所有檔案都使用同一個 baseline (因為 original_id 一樣)
    # === baseline 音檔位置：以 data1["wav_path"] 的父資料夾做參考
    #     例如 /path/to/test/DDAE_snr-2_babble_TMHINT_b2_21_06.wav
    #     baseline => /path/to/test/clean_TMHINT_b2_21_06.wav
    #     只需載入一次，後面 pair 都可共用
    baseline_wav_tensor = None
    # 隨便拿 pair_list 裡的第一個，用 data1 取路徑
    first_idx1, first_idx2 = pair_list[0]
    data_example = dataset[first_idx1]
    wav_dir = os.path.dirname(data_example["wav_path"])  # 父資料夾
    baseline_path = os.path.join(wav_dir, f"clean_{args.original_id}.wav")
    logging.info(f"Trying baseline path: {baseline_path}")

    if os.path.exists(baseline_path):
        try:
            base_wav, sr_base = torchaudio.load(baseline_path, channels_first=False)
            if sr_base != sr:
                logging.info(f"Resampling baseline from {sr_base} to {sr} ...")
                resampler = torchaudio.transforms.Resample(sr_base, sr)
                base_wav = resampler(base_wav)
            baseline_wav_tensor = base_wav.squeeze(-1).float().to(device)
            logging.info(f"Baseline audio loaded: shape={baseline_wav_tensor.shape}")
        except Exception as e:
            logging.warning(f"Failed to load baseline {baseline_path}, fallback to zeros. Error: {e}")
            baseline_wav_tensor = None
    else:
        logging.warning(f"Baseline file not found: {baseline_path}, fallback to zeros.")
        baseline_wav_tensor = None

    for idx_pair, (idx1, idx2) in enumerate(pair_list):
        data1 = dataset[idx1]
        data2 = dataset[idx2]
        wave1 = data1["waveform"]  # shape [time1]
        wave2 = data2["waveform"]  # shape [time2]

        # ground truth
        gt_mos1 = data1.get("score", None)
        gt_mos2 = data2.get("score", None)

        # 檔名 or system_id
        fname1 = os.path.basename(data1["wav_path"])
        fname2 = os.path.basename(data2["wav_path"])
        system_id1 = data1.get("system_id", fname1)
        system_id2 = data2.get("system_id", fname2)

        # cat wave
        cat_waveform_np = torch.cat([wave1, wave2], dim=0).numpy()  # [time1 + time2]
        cat_waveform = torch.tensor(cat_waveform_np).unsqueeze(0).float().to(device)
        cat_length = torch.tensor([cat_waveform.size(1)], dtype=torch.long).to(device)

        inputs = {"waveform": cat_waveform, "waveform_lengths": cat_length}
        outputs = model.mean_net_inference(inputs)
        pred_mos = outputs["scores"].cpu().item()

        wave_len1 = wave1.shape[0]
        wave_len2 = wave2.shape[0]
        wave_len_total = wave_len1 + wave_len2
        time_axis = np.arange(wave_len_total) / float(sr)

        # normalize wave
        cat_waveform_scaled = minmax_scale(cat_waveform_np)
        title_str = (f"[Pair {idx_pair+1}/{len(pair_list)}] Concat:\n"
                     f"{system_id1} (GT={gt_mos1}) + {system_id2} (GT={gt_mos2}) => Pred={pred_mos:.3f}")

        # === Explanation ===
        explanation_np = None
        if args.explain_method == "gradcam":
            cam = gradcam.compute_cam(inputs)
            explanation_np = cam.detach().cpu().numpy()
        elif args.explain_method == "ig":
            # 依照需求把 baseline 同樣 concat
            if baseline_wav_tensor is not None:
                # wave1 baseline
                wave1_baseline_len = wave1.shape[0]
                wave1_bl = match_length(baseline_wav_tensor, wave1_baseline_len)
                # wave2 baseline
                wave2_baseline_len = wave2.shape[0]
                wave2_bl = match_length(baseline_wav_tensor, wave2_baseline_len)
                # cat baseline
                cat_bl = torch.cat([wave1_bl, wave2_bl], dim=0).unsqueeze(0)  # [1, wave_len_total]
                cat_bl = cat_bl.to(device)
            else:
                # fallback
                cat_bl = torch.zeros_like(cat_waveform)

            attributions = integrated_gradients_explanation(model, inputs, cat_bl)
            explanation_np = attributions[0].detach().cpu().numpy()

        elif args.explain_method == "saliency":
            attributions = saliency_explanation(model, inputs)
            explanation_np = attributions[0].detach().cpu().numpy()
        elif args.explain_method == "attention":
            attn_1d = attention_viz.compute_1d_attention()
            explanation_np = attn_1d.detach().cpu().numpy()
        elif args.explain_method == "lime":
            explanation_list = lime_explanation(model, inputs, sampling_rate=sr)
            logging.info(f"[Pair {idx_pair}] LIME Explanation: {explanation_list}")
            explanation_np = np.random.rand(wave_len_total)  # demo
        elif args.explain_method == "shap":
            shap_values = shap_explanation(model, inputs)
            explanation_np = shap_values[0][0]
        elif args.explain_method == "inputxgradient":
            attributions = inputxgradient_explanation(model, inputs)
            explanation_np = attributions[0].detach().cpu().numpy()

        # 對解釋曲線做插值 + normalize
        if explanation_np is not None:
            if len(explanation_np) != wave_len_total:
                t_exp = np.linspace(0, 1, len(explanation_np))
                t_cat = np.linspace(0, 1, wave_len_total)
                explanation_np = np.interp(t_cat, t_exp, explanation_np)
            explanation_scaled = minmax_scale(explanation_np)
        else:
            explanation_scaled = None

        # 繪圖
        plt.figure(figsize=(12, 4))
        # 前半段 => 藍色
        plt.plot(time_axis[:wave_len1], cat_waveform_scaled[:wave_len1], label=f"{system_id1}", color="blue")
        # 後半段 => 綠色
        plt.plot(time_axis[wave_len1:], cat_waveform_scaled[wave_len1:], label=f"{system_id2}", color="green")
        # 疊加解釋曲線 => 紅色
        if explanation_scaled is not None:
            plt.plot(time_axis, explanation_scaled, label=f"{args.explain_method} (scaled)", color="red", alpha=0.7)

        plt.title(title_str)
        plt.xlabel("Time (sec)")
        plt.ylabel("Scaled amplitude")
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(
            args.outdir, 
            f"pair_{idx_pair:03d}_{system_id1}_plus_{system_id2}_{args.explain_method}.png"
        )
        plt.savefig(out_png)
        plt.close()

        logging.info(f"Pair {idx_pair+1}/{len(pair_list)} done. Plot saved to {out_png}")

    logging.info("All pairwise concat done.")

if __name__ == "__main__":
    main()
