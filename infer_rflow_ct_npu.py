#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_rflow_ct_npu.py
=====================
Ascend 910B (torch_npu) 适配的 ``rflow-ct`` 图像-only 推理脚本。

参考仓库 ``inference_diff_unet_tutorial.ipynb`` 的流程，在华为昇腾 910B 上
使用 ``rflow-ct`` 模型生成一张肺部/胸部 CT 影像（NIfTI），并自动做适配性检查。

流程（与 notebook 一致）：
    1. 设置 NPU 设备（torch_npu）
    2. 读取 configs/config_network_rflow.json 构建 VAE + DiffusionUNet (rflow-ct)
    3. 加载 checkpoints（models/autoencoder_v1.pt, models/diff_unet_3d_rflow-ct.pt）
    4. Rectified Flow 采样 30 步，生成 latent
    5. SlidingWindowInferer 解码 latent，得到 CT 图像
    6. 保存 .nii.gz 并输出适配性 PASS/FAIL 结论

昇腾 NPU 适配要点（相对 CUDA 原版 scripts/diff_model_infer.py 的改动）：
    * 设备由 ``cuda`` 改为 ``npu:0``，并 ``import torch_npu``
    * 单卡推理，跳过 ``torchrun`` 多卡逻辑（``initialize_distributed``）
    * ``use_flash_attention`` 置为 ``False``：
      MONAI 1.6 的 ``DiffusionModelUNetMaisi`` 在非 CUDA 环境下
      ``use_flash_attention=True`` 会直接抛 ``ValueError``
      （见 monai/.../diffusion_model_unet_maisi.py:153）；NPU 上改用
      SABlock 的 einsum/softmax 自注意力，可正常运行。
    * autocast 使用 ``torch.npu.amp.autocast`` 代替 ``torch.amp.autocast("cuda")``

用法（在已安装 CANN + torch_npu 的昇腾 910B 上，于本仓库根目录执行）：
    python infer_rflow_ct_npu.py

    # 内存有限 / 想生成更贴近肺部临床分辨率的胸部 CT（FOV 与胸部训练分布一致）：
    python infer_rflow_ct_npu.py --dim 256 256 128 --spacing 1.2 1.2 2.5
    # 若 910B 内存充裕（≥64G），可尝试 512 分辨率：
    python infer_rflow_ct_npu.py --dim 512 512 128 --spacing 0.684 0.684 2.422

注意：dim × spacing 决定视野 (FOV)，应落在 rflow-ct 训练分布内，否则输出失真。
    肺部/胸部推荐 FOV 见 docs/inference.md 的 "Recommended Spacing for CT" 表。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime

import numpy as np

# 必须先导入 torch_npu，torch 才会注册 "npu" 设备类型
import torch
import torch_npu  # noqa: F401  (仅导入即可生效)
import nibabel as nib

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism
from tqdm import tqdm

# rflow-ct 在 NPU 上必须关闭 flash attention（MONAI 1.6 在非 CUDA 环境会直接报错）
USE_FLASH_ATTENTION = False

logger = logging.getLogger("infer_rflow_ct_npu")


# --------------------------------------------------------------------------- #
# NPU 适配 helper
# --------------------------------------------------------------------------- #
def setup_device(device_id: int = 0) -> torch.device:
    """校验 NPU 可用性并返回 npu:{device_id} 设备。"""
    if not torch.npu.is_available():
        raise RuntimeError(
            "torch.npu.is_available() 为 False。请确认在昇腾 910B 服务器上执行，"
            "且已正确安装 CANN + torch_npu（Python 侧 import torch_npu 成功）。"
        )
    torch.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")


def make_autocast():
    """构造 NPU 的 fp16 autocast 上下文，兼容不同 torch_npu 版本。

    注意：MAISI 的 VAE 内部使用 MaisiGroupNorm3D(norm_float16=True)，无论是否开启
    autocast 都会输出 fp16，因此本脚本必须在 fp16 autocast 下运行（与 CUDA 原版
    用 torch.amp.autocast("cuda") 包裹整段采样+解码的做法一致）。
    """
    try:
        return torch.npu.amp.autocast(enabled=True, dtype=torch.float16)
    except AttributeError:  # 较新版本也可用 torch.amp.autocast(device_type='npu')
        return torch.amp.autocast(device_type="npu", enabled=True, dtype=torch.float16)


def resolve_refs(obj, root: dict):
    """解析 MONAI bundle 配置里的 '@key' 引用（如 '@spatial_dims'）。"""
    if isinstance(obj, dict):
        return {k: resolve_refs(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_refs(v, root) for v in obj]
    if isinstance(obj, str) and obj.startswith("@") and obj[1:] in root:
        return root[obj[1:]]
    return obj


def build_kwargs(def_dict: dict, root: dict) -> dict:
    """把 *_def 配置字典转成模型构造 kwargs（去掉 _target_，解析 @ 引用）。"""
    d = dict(def_dict)
    d.pop("_target_", None)
    return resolve_refs(d, root)


# --------------------------------------------------------------------------- #
# 与 scripts/utils_infer.py 等价的解码器（内联，避免依赖仓库 scripts 包）
# --------------------------------------------------------------------------- #
class ReconModel(torch.nn.Module):
    """用 VAE 解码 latent：z -> image。"""

    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        return self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)


def dynamic_infer(inferer, model, images):
    """latent 小于 roi 时整卷直接解码，否则用 SlidingWindowInferer 分窗。"""
    if torch.numel(images[0:1, 0:1, ...]) <= math.prod(inferer.roi_size):
        return model(images)
    spatial_dims = images.shape[2:]
    orig_roi = inferer.roi_size
    adjusted_roi = [min(roi_dim, img_dim) for roi_dim, img_dim in zip(orig_roi, spatial_dims)]
    inferer.roi_size = adjusted_roi
    output = inferer(network=model, inputs=images)
    inferer.roi_size = orig_roi
    return output


# --------------------------------------------------------------------------- #
# 配置加载
# --------------------------------------------------------------------------- #
def load_configs(repo_root: str):
    """加载 rflow-ct 图像-only 推理所需的三份配置。"""
    with open(os.path.join(repo_root, "configs", "config_network_rflow.json"), encoding="utf-8") as f:
        model_def = json.load(f)
    with open(os.path.join(repo_root, "configs", "config_maisi_diff_model_rflow-ct.json"), encoding="utf-8") as f:
        model_cfg = json.load(f)
    with open(os.path.join(repo_root, "configs", "environment_maisi_diff_model_rflow-ct.json"), encoding="utf-8") as f:
        env_cfg = json.load(f)
    return model_def, model_cfg, env_cfg


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_sampling(
    args,
    device: torch.device,
    autoencoder,
    unet,
    scale_factor,
    scheduler,
    model_def: dict,
    infer_cfg: dict,
) -> np.ndarray:
    """Rectified Flow 采样 + VAE 解码，返回 int16 的 CT 图像。"""
    latent_channels = model_def["latent_channels"]
    output_size = tuple(infer_cfg["dim"])
    out_spacing = tuple(infer_cfg["spacing"])

    # 下采样层数 -> latent 空间压缩倍数（与 scripts/diff_model_infer.py 一致）
    num_downsample_level = len(model_def["diffusion_unet_def"]["num_channels"])
    divisor = 2 ** (num_downsample_level - 2)
    latent_shape = (
        1,
        latent_channels,
        output_size[0] // divisor,
        output_size[1] // divisor,
        output_size[2] // divisor,
    )
    logger.info(f"output_size={output_size}, latent_shape={latent_shape}")

    include_body_region = bool(unet.include_top_region_index_input)
    include_modality = unet.num_class_embeds is not None

    # 条件张量（spacing 放大 100 倍；保持 fp32，autocast 开启时会被自动转为 fp16 计算，
    # 关闭 autocast 时也不会与 UNet 的 fp32 权重产生 dtype 不匹配）
    spacing_tensor = torch.from_numpy(np.asarray(out_spacing, dtype=np.float32)[None] * 1e2).to(device)
    modality_tensor = torch.full((1,), infer_cfg["modality"], dtype=torch.long, device=device)

    noise = torch.randn(latent_shape, device=device)
    image = noise

    scheduler.set_timesteps(
        num_inference_steps=infer_cfg["num_inference_steps"],
        input_img_size_numel=torch.prod(torch.tensor(image.shape[2:])),
    )
    all_timesteps = scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
    cfg_guidance_scale = float(infer_cfg["cfg_guidance_scale"])

    autoencoder.eval()
    unet.eval()
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    with make_autocast():
        # ---- 去噪采样 30 步 ----
        for t, next_t in tqdm(
            zip(all_timesteps, all_next_timesteps),
            total=min(len(all_timesteps), len(all_next_timesteps)),
            desc="Sampling (rflow-ct)",
        ):
            unet_inputs = {
                "x": image,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
            }
            if include_body_region:  # rflow-ct 为 False，纯占位分支
                unet_inputs["top_region_index_tensor"] = None
                unet_inputs["bottom_region_index_tensor"] = None
            if include_modality:
                unet_inputs["class_labels"] = modality_tensor

            if cfg_guidance_scale > 0:
                for k in list(unet_inputs.keys()):
                    if k != "class_labels":
                        unet_inputs[k] = torch.cat([unet_inputs[k]] * 2)
                    else:
                        unet_inputs[k] = torch.cat([unet_inputs[k], torch.zeros_like(modality_tensor)])
                model_t, model_uncond = unet(**unet_inputs).chunk(2)
                model_output = model_uncond + cfg_guidance_scale * (model_t - model_uncond)
            else:
                model_output = unet(**unet_inputs)

            image, _ = scheduler.step(model_output, t, image, next_t)

        # ---- VAE 解码 latent -> 图像 ----
        inferer = SlidingWindowInferer(
            roi_size=[80, 80, 80],
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=0.4,
            sw_device=device,
            device=device,
        )
        synthetic_images = dynamic_infer(inferer, recon_model, image)
        data = synthetic_images.squeeze().cpu().detach().numpy()

    # ---- 后处理：转 CT HU (int16) ----
    modality = int(infer_cfg["modality"])
    if modality >= 8:
        # MR（本脚本聚焦 CT，仅保留分支）
        data = (data - 0.0) / (1.0 - 0.0) * (1000.0 - 0.0) + 0.0
        data = np.clip(data, 0.0, None)
    else:
        a_min, a_max, b_min, b_max = -1000.0, 1000.0, 0.0, 1.0
        data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        data = np.clip(data, a_min, a_max)
    return np.int16(data)


def save_image(data: np.ndarray, output_size: tuple, out_spacing: tuple, output_path: str) -> None:
    """保存 NIfTI，affine 由 spacing 构成（与 scripts/diff_model_infer.py 一致）。"""
    out_affine = np.eye(4)
    for i in range(3):
        out_affine[i, i] = out_spacing[i]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine=out_affine), output_path)
    logger.info(f"Saved {output_path}.")


def verify_output(data: np.ndarray) -> tuple[dict, bool]:
    """适配性自检：无 NaN/Inf、HU 范围、是否同时含空气与软组织。"""
    data_f = data.astype(np.float32)
    finite = np.isfinite(data_f)
    checks = {
        "nonfinite_count": int((~finite).sum()),
        "hu_min": int(data_f.min()),
        "hu_max": int(data_f.max()),
        "hu_mean": float(data_f.mean()),
        "hu_std": float(data_f.std()),
        "air_frac_below_-500HU": float((data_f < -500).mean()),
        "soft_tissue_frac_above_0HU": float((data_f > 0).mean()),
    }
    passed = (
        checks["nonfinite_count"] == 0
        and checks["hu_min"] >= -1024
        and checks["hu_max"] <= 1024
        and checks["hu_std"] > 0.5
        and checks["air_frac_below_-500HU"] > 0.01
        and checks["soft_tissue_frac_above_0HU"] > 0.01
    )
    return checks, passed


def save_preview(data: np.ndarray, out_png: str) -> None:
    """保存三正交切面的灰度预览 PNG，便于快速目检生成的肺部 CT。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib 未安装时跳过预览
        logger.warning(f"matplotlib 不可用，跳过 PNG 预览: {e}")
        return

    vol = np.squeeze(data)
    H, W, D = vol.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = int(vol.min()), int(vol.max())
    titles = [f"Axial  (z={D // 2})", f"Coronal (y={W // 2})", f"Sagittal (x={H // 2})"]
    slices = [vol[:, :, D // 2], vol[:, W // 2, :], vol[H // 2, :, :]]
    for ax, sl, ti in zip(axes, slices, titles):
        ax.imshow(sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(ti)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    logger.info(f"Saved preview {out_png}.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Ascend 910B (torch_npu) 上运行 rflow-ct 生成肺部 CT（图像-only），并验证模型适配性。"
    )
    p.add_argument("--repo-root", type=str, default=".", help="NV-Generate-CTMR 仓库根目录（含 configs/ 与 models/）")
    p.add_argument("--dim", nargs=3, type=int, default=[256, 256, 128], help="输出体素尺寸，需能被 16 整除")
    p.add_argument("--spacing", nargs=3, type=float, default=[1.7, 1.7, 2.0], help="输出体素间距 (mm)")
    p.add_argument("--num-steps", type=int, default=None, help="去噪步数（默认取配置 30）")
    p.add_argument("--modality", type=int, default=None, help="modality 编码（默认 1=CT）")
    p.add_argument("--cfg-guidance-scale", type=float, default=None, help="CFG 引导系数（CT 默认 0）")
    p.add_argument("--random-seed", type=int, default=0, help="随机种子，保证可复现")
    p.add_argument("--device-id", type=int, default=0, help="NPU 卡号")
    p.add_argument("--output-dir", type=str, default=None, help="输出目录（默认取环境配置 ./output）")
    p.add_argument("--output-prefix", type=str, default=None, help="输出文件名前缀（默认 unet_3d）")
    p.add_argument("--autoencoder", type=str, default=None, help="VAE 权重路径覆盖")
    p.add_argument("--unet", type=str, default=None, help="Diffusion UNet 权重路径覆盖")
    p.add_argument("--preview", action="store_true", help="额外保存三正交切面 PNG 预览")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    set_determinism(seed=args.random_seed)

    # 1) NPU 设备与环境信息
    device = setup_device(args.device_id)
    logger.info("========== NPU environment ==========")
    logger.info(f"torch            : {torch.__version__}")
    logger.info(f"torch_npu        : {getattr(torch_npu, '__version__', 'unknown')}")
    logger.info(f"NPU count        : {torch.npu.device_count()}")
    logger.info(f"NPU device       : {torch.npu.get_device_name(device_id=args.device_id)}")
    logger.info(f"infer device     : {device}")
    logger.info("=====================================")

    # 2) 配置
    model_def, model_cfg, env_cfg = load_configs(args.repo_root)
    infer_cfg = dict(model_cfg["diffusion_unet_inference"])
    if args.dim is not None:
        infer_cfg["dim"] = args.dim
    if args.spacing is not None:
        infer_cfg["spacing"] = args.spacing
    if args.num_steps is not None:
        infer_cfg["num_inference_steps"] = args.num_steps
    if args.modality is not None:
        infer_cfg["modality"] = args.modality
    if args.cfg_guidance_scale is not None:
        infer_cfg["cfg_guidance_scale"] = args.cfg_guidance_scale
    output_size = tuple(infer_cfg["dim"])
    out_spacing = tuple(infer_cfg["spacing"])
    if any(s % 16 != 0 for s in output_size):
        raise ValueError(f"dim 需能被 16 整除，当前 {output_size}")

    # 3) 构建模型（关键 NPU 适配：关闭 flash attention）
    autoencoder_kwargs = build_kwargs(model_def["autoencoder_def"], model_def)
    unet_kwargs = build_kwargs(model_def["diffusion_unet_def"], model_def)
    unet_kwargs["use_flash_attention"] = USE_FLASH_ATTENTION
    logger.info(
        "构建 DiffusionModelUNetMaisi(use_flash_attention=False)：NPU 上禁用 flash attention，"
        "改用 SABlock(einsum+softmax) 自注意力。"
    )
    autoencoder = AutoencoderKlMaisi(**autoencoder_kwargs).to(device)
    unet = DiffusionModelUNetMaisi(**unet_kwargs).to(device)
    logger.info(f"autoencoder params: {sum(p.numel() for p in autoencoder.parameters()) / 1e6:.1f}M")
    logger.info(f"diffusion unet params: {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M")

    # 4) 加载权重
    autoencoder_path = args.autoencoder or os.path.join(args.repo_root, env_cfg["trained_autoencoder_path"])
    unet_path = args.unet or os.path.join(args.repo_root, env_cfg["model_dir"], env_cfg["model_filename"])
    logger.info(f"Loading autoencoder from {autoencoder_path}")
    ckpt_ae = torch.load(autoencoder_path, map_location="cpu", weights_only=False)
    if "unet_state_dict" in ckpt_ae:
        ckpt_ae = ckpt_ae["unet_state_dict"]
    autoencoder.load_state_dict(ckpt_ae)

    logger.info(f"Loading diffusion unet from {unet_path}")
    ckpt = torch.load(unet_path, map_location="cpu", weights_only=False)
    unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
    scale_factor = ckpt["scale_factor"]
    if isinstance(scale_factor, torch.Tensor):
        scale_factor = float(scale_factor)
    logger.info(f"scale_factor -> {scale_factor}")

    # 5) 调度器
    sched_kwargs = build_kwargs(model_def["noise_scheduler"], model_def)
    scheduler = RFlowScheduler(**sched_kwargs)
    logger.info(
        f"Scheduler: RFlowScheduler, steps={infer_cfg['num_inference_steps']}, "
        f"modality={infer_cfg['modality']}, cfg_guidance_scale={infer_cfg['cfg_guidance_scale']}"
    )

    # 6) 推理
    logger.info("Running inference on NPU...")
    data = run_sampling(args, device, autoencoder, unet, scale_factor, scheduler, model_def, infer_cfg)

    # 7) 保存
    output_dir = os.path.join(args.repo_root, args.output_dir) if args.output_dir else os.path.join(args.repo_root, env_cfg["output_dir"])
    prefix = args.output_prefix or env_cfg["output_prefix"]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = (
        f"{prefix}_seed{args.random_seed}"
        f"_size{output_size[0]}x{output_size[1]}x{output_size[2]}"
        f"_spacing{out_spacing[0]:.2f}x{out_spacing[1]:.2f}x{out_spacing[2]:.2f}"
        f"_{timestamp}_rank0_modality{infer_cfg['modality']}.nii.gz"
    )
    output_path = os.path.join(output_dir, fname)
    save_image(data, output_size, out_spacing, output_path)

    if args.preview:
        save_preview(data, output_path.replace(".nii.gz", "_preview.png"))

    # 8) 适配性验证
    checks, passed = verify_output(data)
    logger.info("---------- adaptability check ----------")
    for k, v in checks.items():
        logger.info(f"  {k:32s}: {v}")
    if passed:
        logger.info("RESULT: PASS — rflow-ct 在昇腾 910B 上推理成功，输出为合理的肺部/胸部 CT 体数据。")
    else:
        logger.warning(
            "RESULT: FAIL — 输出未通过自检。若为非有限值或异常 HU 范围，"
            "请检查 FOV(dim×spacing) 是否在训练分布内、是否关闭了 flash attention、autocast 是否开启。"
        )
    logger.info(f"Output: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
