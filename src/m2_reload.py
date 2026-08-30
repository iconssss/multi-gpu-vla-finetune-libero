import json, os, torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.configs import PreTrainedConfig

CKPT = os.environ.get("CKPT", "/root/shared-nvme/project10-vla/runs/m2_fast_1k/step_1000")
BASE = "/root/shared-nvme/project10-vla/cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
DATA = os.environ.get("LIBERO_SNAP", "/root/shared-nvme/project10-vla/cache/huggingface/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
OUT = "/root/shared-nvme/project10-vla/logs/m2_reload_result.json"

res = {"ckpt": CKPT}
dataset = LeRobotDataset("lerobot/libero", root=DATA, episodes=[0], video_backend="pyav",
                         delta_timestamps={"action": [i / 10 for i in range(50)]})
try:
    cfg = PreTrainedConfig.from_pretrained(CKPT)
    res["config_load"] = "PASS (from checkpoint)"
except Exception as e:
    cfg = PreTrainedConfig.from_pretrained(BASE)
    res["config_load"] = "FALLBACK to base: " + repr(e)[:200]
cfg.pretrained_path = CKPT; cfg.device = "cuda"; cfg.input_features = {}
policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
policy.eval()
res["load"] = "PASS"
res["parameters"] = sum(p.numel() for p in policy.parameters())
res["dtype"] = str(next(policy.parameters()).dtype)

pre_overrides = {
    "device_processor": {"device": "cuda"},
    "normalizer_processor": {"stats": dataset.meta.stats, "features": {**policy.config.input_features, **policy.config.output_features}, "norm_map": policy.config.normalization_mapping},
    "rename_observations_processor": {"rename_map": {}},
}
post_overrides = {"unnormalizer_processor": {"stats": dataset.meta.stats, "features": policy.config.output_features, "norm_map": policy.config.normalization_mapping}}
preprocessor, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT,
                                           preprocessor_overrides=pre_overrides, postprocessor_overrides=post_overrides)
batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
for key in dataset.meta.camera_keys:
    if key in batch and batch[key].dtype == torch.uint8:
        batch[key] = batch[key].float() / 255.0
processed = preprocessor(batch)
res["processor"] = "PASS"
res["batch_keys"] = sorted(k for k in processed if isinstance(processed[k], torch.Tensor))
with torch.no_grad():
    out = policy(processed)

def struct(o):
    if isinstance(o, torch.Tensor): return ["tensor", list(o.shape)]
    if isinstance(o, dict): return {k: struct(v) for k, v in list(o.items())[:8]}
    if isinstance(o, (tuple, list)): return [struct(x) for x in o[:4]]
    return type(o).__name__

def find_action(o):
    if isinstance(o, dict):
        if "action" in o and isinstance(o["action"], torch.Tensor):
            return o["action"], "dict.action"
        for v in o.values():
            r = find_action(v)
            if r: return r[0], "nested." + r[1]
    if isinstance(o, (tuple, list)):
        for x in o:
            r = find_action(x)
            if r: return r[0], "seq." + r[1]
    if isinstance(o, torch.Tensor) and o.dim() >= 2:
        return o, "tensor"
    return None

res["out_structure"] = struct(out)
print("OUT_STRUCTURE:", json.dumps(res["out_structure"])[:600])
found = find_action(out)
if found:
    acts, res["output_key"] = found
    res["forward"] = "PASS"
    res["output_shape"] = list(acts.shape)
    res["output_finite"] = bool(torch.isfinite(acts.float()).all())
else:
    acc = []
    def collect(o):
        if isinstance(o, torch.Tensor): acc.append(o)
        elif isinstance(o, dict):
            for v in o.values(): collect(v)
        elif isinstance(o, (tuple, list)):
            for x in o: collect(x)
    collect(out)
    res["forward"] = "PASS"
    res["output_key"] = "all_tensors"
    res["output_shapes"] = [list(t.shape) for t in acc]
    res["output_finite"] = bool(acc) and all(bool(torch.isfinite(t.float()).all()) for t in acc)
res["status"] = "PASS" if res["output_finite"] else "FAIL"
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
