"""Export EdgeTAM's full VIDEO pipeline (with tracking memory) to ONNX for phones.

Produces 4 graphs + constants, the same split a phone app uses:
  image_encoder.onnx     frame (1,3,1024,1024) -> feat_s0, feat_s1, vision_feat, vision_pos
  decoder.onnx           features + clicks (+ optional mask prompt) -> 4 masks, 4 IoUs, 4 object pointers, object score
  memory_encoder.onnx    frame features + mask -> 512 compressed memory tokens (EdgeTAM's 2D spatial perceiver)
  memory_attention.onnx  frame features + memory bank (any length) -> memory-conditioned features
  constants.npz          small tensors used by the per-frame loop (pointer encodings, no-object embeddings, RoPE)

Then makes fp16 (phone GPU/NPU) and int8 (phone CPU) versions and checks every graph against PyTorch.

    python scripts/export_edgetam_onnx.py
"""

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "edgetam-hf"
OUT = ROOT / "models" / "edgetam-onnx-video"
OPSET = 17


def rotate_pairwise(x):
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class ImageEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        out = self.m.vision_encoder(pixel_values)
        fpn, pos = out.fpn_hidden_states, out.fpn_position_encoding
        feat_s0 = self.m.mask_decoder.conv_s0(fpn[0])
        feat_s1 = self.m.mask_decoder.conv_s1(fpn[1])
        return feat_s0, feat_s1, fpn[2], pos[2]


class Decoder(nn.Module):
    """Prompt encoder + mask decoder returning ALL 4 mask tokens; the loop picks one (multimask / stability rules)."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("image_pe", m.get_image_wide_positional_embeddings().detach().clone())

    def forward(self, pix_feat, feat_s0, feat_s1, point_coords, point_labels, mask_input, has_mask):
        pe = self.m.prompt_encoder
        md = self.m.mask_decoder
        sparse = pe._embed_points(point_coords, point_labels, pad=True)
        h = has_mask.view(1, 1, 1, 1)
        no_mask = pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, pix_feat.shape[2], pix_feat.shape[3])
        dense = h * pe.mask_embed(mask_input) + (1 - h) * no_mask

        # --- body of EdgeTamVideoMaskDecoder.forward, without the output-token slicing ---
        b, c, hh, ww = pix_feat.shape
        tokens = torch.cat([md.obj_score_token.weight, md.iou_token.weight, md.mask_tokens.weight], dim=0)
        tokens = tokens.repeat(1, 1, 1, 1)
        tokens = torch.cat((tokens, sparse), dim=2)
        image_embeddings = pix_feat + dense
        point_embeddings, image_embeddings = md.transformer(
            point_embeddings=tokens,
            image_embeddings=image_embeddings,
            image_positional_embeddings=self.image_pe,
            attention_similarity=None,
        )
        iou_token_out = point_embeddings[:, :, 1, :]
        mask_tokens_out = point_embeddings[:, :, 2 : 2 + md.num_mask_tokens, :]
        image_embeddings = image_embeddings.transpose(2, 3).reshape(b, c, hh, ww)
        up = md.upscale_conv1(image_embeddings) + feat_s1
        up = md.activation(md.upscale_layer_norm(up))
        up = md.activation(md.upscale_conv2(up) + feat_s0)
        hyper = torch.stack([md.output_hypernetworks_mlps[i](mask_tokens_out[:, :, i, :]) for i in range(md.num_mask_tokens)], dim=2)
        _, uc, uh, uw = up.shape
        masks = (hyper @ up.view(1, 1, uc, uh * uw)).view(1, 1, -1, uh, uw)
        iou = md.iou_prediction_head(iou_token_out)
        obj_score = md.pred_obj_score_head(point_embeddings[:, :, 0, :])
        pointers = self.m.object_pointer_proj(mask_tokens_out)
        return masks, iou, pointers, obj_score


class MemoryEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, vision_feat, mask_for_mem, is_obj):
        feats, pos = self.m.memory_encoder(vision_feat, mask_for_mem)
        occ = self.m.occlusion_spatial_embedding_parameter  # None when the checkpoint disables it
        if occ is not None:
            feats = feats + (1 - is_obj.view(1, 1, 1, 1)) * occ[..., None, None]
        else:
            feats = feats + 0 * is_obj.view(1, 1, 1, 1)  # keep the input in the graph so the interface is stable
        return self.m.spatial_perceiver(feats, pos)


class MemoryAttention(nn.Module):
    """EdgeTAM memory attention with key RoPE given as inputs, so ONE graph works for any memory length
    (the original passes token counts as Python ints, which ONNX would freeze). k_bias (0 / -1e4) lets a phone
    NPU use a fixed, padded memory size."""

    def __init__(self, m):
        super().__init__()
        self.ma = m.memory_attention
        cos, sin = self.ma.rotary_emb()
        self.register_buffer("q_cos", cos.detach().clone())
        self.register_buffer("q_sin", sin.detach().clone())

    @staticmethod
    def _attend(attn, q_in, k_in, v_in, q_cos, q_sin, k_cos, k_sin, k_bias):
        n_heads, d = attn.num_attention_heads, attn.head_dim
        q = attn.q_proj(q_in)
        k = attn.k_proj(k_in)
        v = attn.v_proj(v_in)
        q = q.view(1, -1, n_heads, d).transpose(1, 2)
        k = k.view(1, -1, n_heads, d).transpose(1, 2)
        v = v.view(1, -1, n_heads, d).transpose(1, 2)
        q = q * q_cos + rotate_pairwise(q) * q_sin
        k = k * k_cos + rotate_pairwise(k) * k_sin
        w = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
        if k_bias is not None:
            w = w + k_bias.view(1, 1, 1, -1)
        w = torch.softmax(w, dim=-1)
        o = torch.matmul(w, v).transpose(1, 2).reshape(1, -1, n_heads * d)
        return attn.o_proj(o)

    def forward(self, curr, curr_pos, memory, memory_pos, k_cos, k_sin, k_bias):
        out = curr + 0.1 * curr_pos
        for layer in self.ma.layers:
            q = layer.layer_norm1(out)
            out = out + self._attend(layer.self_attn, q, q, q, self.q_cos, self.q_sin, self.q_cos, self.q_sin, None)
            q = layer.layer_norm2(out)
            out = out + self._attend(layer.cross_attn_image, q, memory + memory_pos, memory,
                                     self.q_cos, self.q_sin, k_cos, k_sin, k_bias)
            out = out + layer.mlp(layer.layer_norm3(out))
        return self.ma.layer_norm(out)


def key_rope(rope_k_cos, rope_k_sin, n_frames, n_ptr_tokens, tokens_per_frame=512):
    """cos/sin per memory token: per frame 256 global tokens (no RoPE) then 256 spatial tokens (RoPE); pointers no RoPE."""
    d = rope_k_cos.shape[-1]
    spatial = rope_k_cos.shape[0]
    temporal = tokens_per_frame - spatial
    one_c = np.concatenate([np.ones((temporal, d), np.float32), rope_k_cos], 0)
    one_s = np.concatenate([np.zeros((temporal, d), np.float32), rope_k_sin], 0)
    cos = np.concatenate([np.tile(one_c, (n_frames, 1)), np.ones((n_ptr_tokens, d), np.float32)], 0)
    sin = np.concatenate([np.tile(one_s, (n_frames, 1)), np.zeros((n_ptr_tokens, d), np.float32)], 0)
    return cos, sin


def export(module, args, path, names_in, names_out, dynamic=None):
    torch.onnx.export(module, args, str(path), input_names=names_in, output_names=names_out,
                      opset_version=OPSET, dynamic_axes=dynamic or {}, do_constant_folding=True, dynamo=False)
    return path


def run_ort(path, feeds):
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    wanted = {i.name for i in s.get_inputs()}
    return s.run(None, {k: v.numpy() if torch.is_tensor(v) else v for k, v in feeds.items() if k in wanted})


def maxdiff(a, b):
    a = a.detach().numpy() if torch.is_tensor(a) else a
    b = b.detach().numpy() if torch.is_tensor(b) else b
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())


def main():
    from transformers import EdgeTamVideoModel

    torch.manual_seed(0)
    model = EdgeTamVideoModel.from_pretrained(str(SRC), dtype=torch.float32).eval()
    cfg = model.config
    fp32 = OUT / "fp32"
    fp32.mkdir(parents=True, exist_ok=True)
    report = {}

    with torch.no_grad():
        # ---------------- image encoder ----------------
        enc = ImageEncoder(model).eval()
        px = torch.randn(1, 3, 1024, 1024)
        ref = enc(px)
        print("image encoder outputs:", [tuple(t.shape) for t in ref])
        p = export(enc, (px,), fp32 / "image_encoder.onnx", ["pixel_values"], ["feat_s0", "feat_s1", "vision_feat", "vision_pos"])
        got = run_ort(p, {"pixel_values": px})
        report["image_encoder"] = max(maxdiff(a, b) for a, b in zip(ref, got))
        feat_s0, feat_s1, vision_feat, vision_pos = ref

        # ---------------- decoder ----------------
        dec = Decoder(model).eval()
        pix = vision_feat + model.no_memory_embedding.view(1, -1, 1, 1)
        pts = torch.tensor([[[[300.0, 400.0], [520.0, 610.0]]]])
        lbl = torch.tensor([[[1, 0]]], dtype=torch.int64)
        mask_in = torch.zeros(1, 1, 256, 256)
        has = torch.zeros(1)
        ref = dec(pix, feat_s0, feat_s1, pts, lbl, mask_in, has)
        # cross-check against the original HF path (multimask off -> token 0)
        sparse, dense = model.prompt_encoder(input_points=pts, input_labels=lbl, input_boxes=None, input_masks=None)
        hf_masks, hf_iou, _, hf_obj = model.mask_decoder(
            image_embeddings=pix, image_positional_embeddings=dec.image_pe, sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense, multimask_output=True, high_resolution_features=[feat_s0, feat_s1])
        report["decoder_vs_hf_masks"] = maxdiff(ref[0][:, :, 1:], hf_masks)
        report["decoder_vs_hf_objscore"] = maxdiff(ref[3], hf_obj)
        p = export(dec, (pix, feat_s0, feat_s1, pts, lbl, mask_in, has), fp32 / "decoder.onnx",
                   ["pix_feat", "feat_s0", "feat_s1", "point_coords", "point_labels", "mask_input", "has_mask"],
                   ["masks", "iou", "pointers", "obj_score"],
                   {"point_coords": {2: "n_points"}, "point_labels": {2: "n_points"}})
        got = run_ort(p, {"pix_feat": pix, "feat_s0": feat_s0, "feat_s1": feat_s1, "point_coords": pts,
                          "point_labels": lbl, "mask_input": mask_in, "has_mask": has})
        report["decoder"] = max(maxdiff(a, b) for a, b in zip(ref, got))
        # a different number of points must work with the same graph
        got3 = run_ort(p, {"pix_feat": pix, "feat_s0": feat_s0, "feat_s1": feat_s1,
                           "point_coords": torch.tensor([[[[300.0, 400.0], [10.0, 10.0], [700.0, 90.0]]]]),
                           "point_labels": torch.tensor([[[1, 0, 1]]], dtype=torch.int64),
                           "mask_input": mask_in, "has_mask": has})
        ref3 = dec(pix, feat_s0, feat_s1, torch.tensor([[[[300.0, 400.0], [10.0, 10.0], [700.0, 90.0]]]]),
                   torch.tensor([[[1, 0, 1]]], dtype=torch.int64), mask_in, has)
        report["decoder_3pts"] = max(maxdiff(a, b) for a, b in zip(ref3, got3))

        # ---------------- memory encoder ----------------
        menc = MemoryEncoder(model).eval()
        mask_for_mem = (torch.rand(1, 1, 1024, 1024) > 0.5).float() * 20 - 10
        is_obj = torch.ones(1)
        ref = menc(vision_feat, mask_for_mem, is_obj)
        print("memory encoder outputs:", [tuple(t.shape) for t in ref])
        p = export(menc, (vision_feat, mask_for_mem, is_obj), fp32 / "memory_encoder.onnx",
                   ["vision_feat", "mask_for_mem", "is_obj"], ["mem_feat", "mem_pos"])
        got = run_ort(p, {"vision_feat": vision_feat, "mask_for_mem": mask_for_mem, "is_obj": is_obj})
        report["memory_encoder"] = max(maxdiff(a, b) for a, b in zip(ref, got))
        mem_feat, mem_pos = ref

        # ---------------- memory attention ----------------
        mattn = MemoryAttention(model).eval()
        rope_k_cos, rope_k_sin = [t.numpy() for t in model.memory_attention.rotary_emb_k()]
        n_frames, n_ptr = 3, 8
        memory = torch.cat([mem_feat] * n_frames + [torch.randn(1, n_ptr, 64)], 1)
        memory_pos = torch.cat([mem_pos] * n_frames + [torch.randn(1, n_ptr, 64)], 1)
        kc, ks = key_rope(rope_k_cos, rope_k_sin, n_frames, n_ptr)
        curr = vision_feat.flatten(2).permute(0, 2, 1)
        curr_pos = vision_pos.flatten(2).permute(0, 2, 1)
        bias = torch.zeros(1, memory.shape[1])
        ref = mattn(curr, curr_pos, memory, memory_pos, torch.from_numpy(kc), torch.from_numpy(ks), bias)
        hf = model.memory_attention(
            current_vision_features=curr.permute(1, 0, 2), memory=memory.permute(1, 0, 2),
            current_vision_position_embeddings=curr_pos.permute(1, 0, 2),
            memory_posision_embeddings=memory_pos.permute(1, 0, 2),
            num_object_pointer_tokens=n_ptr, num_spatial_memory_tokens=n_frames)
        # HF returns (1, batch, 4096, 256) (seq-first transpose of a point-batched tensor)
        report["memory_attention_vs_hf"] = maxdiff(ref, hf.reshape(1, -1, hf.shape[-1]))
        p = export(mattn, (curr, curr_pos, memory, memory_pos, torch.from_numpy(kc), torch.from_numpy(ks), bias),
                   fp32 / "memory_attention.onnx",
                   ["curr", "curr_pos", "memory", "memory_pos", "k_cos", "k_sin", "k_bias"], ["conditioned"],
                   {"memory": {1: "n_mem"}, "memory_pos": {1: "n_mem"}, "k_cos": {0: "n_mem"}, "k_sin": {0: "n_mem"},
                    "k_bias": {1: "n_mem"}})
        got = run_ort(p, {"curr": curr, "curr_pos": curr_pos, "memory": memory, "memory_pos": memory_pos,
                          "k_cos": kc, "k_sin": ks, "k_bias": bias})
        report["memory_attention"] = maxdiff(ref, got[0])

        # ---------------- constants for the per-frame loop ----------------
        proj = model.temporal_positional_encoding_projection_layer
        ptr_tpe = cfg.enable_temporal_pos_encoding_for_object_pointers and isinstance(proj, nn.Linear)
        np.savez(
            OUT / "constants.npz",
            no_memory_embedding=model.no_memory_embedding.reshape(-1).numpy(),
            no_object_pointer=model.no_object_pointer.reshape(-1).numpy(),
            memory_temporal_pe=model.memory_temporal_positional_encoding.reshape(cfg.num_maskmem, -1).numpy(),
            ptr_tpos_w=proj.weight.numpy() if ptr_tpe else np.zeros((0,), np.float32),
            ptr_tpos_b=proj.bias.numpy() if ptr_tpe else np.zeros((0,), np.float32),
            mask_downsample_w=model.mask_downsample.weight.reshape(4, 4).numpy(),
            mask_downsample_b=model.mask_downsample.bias.numpy(),
            rope_k_cos=rope_k_cos,
            rope_k_sin=rope_k_sin,
        )
    meta = {
        "source": "yonigozlan/EdgeTAM-hf (Meta EdgeTAM, Apache 2.0)",
        "image_size": cfg.image_size,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "num_maskmem": cfg.num_maskmem,
        "max_object_pointers": cfg.max_object_pointers_in_encoder,
        "pointer_temporal_pe": bool(ptr_tpe),
        "sigmoid_scale_for_mem_enc": cfg.sigmoid_scale_for_mem_enc,
        "sigmoid_bias_for_mem_enc": cfg.sigmoid_bias_for_mem_enc,
        "multimask_output_in_sam": cfg.multimask_output_in_sam,
        "multimask_output_for_tracking": cfg.multimask_output_for_tracking,
        "multimask_min_pt_num": cfg.multimask_min_pt_num,
        "multimask_max_pt_num": cfg.multimask_max_pt_num,
        "stability_delta": cfg.mask_decoder_config.dynamic_multimask_stability_delta,
        "stability_thresh": cfg.mask_decoder_config.dynamic_multimask_stability_thresh,
        "mem_dim": cfg.memory_encoder_output_channels,
        "hidden_dim": cfg.vision_config.fpn_hidden_size,
        "tokens_per_memory_frame": int(mem_feat.shape[1]),
        "max_diff_vs_pytorch": report,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nmax |ONNX - PyTorch| per graph:")
    for k, v in report.items():
        print(f"  {k:<26} {v:.2e}")

    make_variants()


def make_variants():
    """fp16 (phone GPU/NPU: CoreML-style / NNAPI / XNNPACK fp16) and int8 (phone CPU, dynamic quantization)."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.transformers.float16 import convert_float_to_float16

    for name in ["image_encoder", "decoder", "memory_encoder", "memory_attention"]:
        src = OUT / "fp32" / f"{name}.onnx"
        (OUT / "fp16").mkdir(exist_ok=True)
        (OUT / "int8").mkdir(exist_ok=True)
        m16 = convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
        onnx.save(m16, str(OUT / "fp16" / f"{name}.onnx"))
        quantize_dynamic(str(src), str(OUT / "int8" / f"{name}.onnx"), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gemm"])
    for v in ["fp32", "fp16", "int8"]:
        size = sum(f.stat().st_size for f in (OUT / v).glob("*.onnx")) / 1e6
        print(f"{v}: {size:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--variants-only":
        make_variants()
    else:
        main()
