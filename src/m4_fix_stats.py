import torch
from safetensors.torch import load_file
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.configs import PreTrainedConfig

CKPT = "/root/shared-nvme/project10-vla/runs/m3_main_5k/step_5000"
DATA = "/root/shared-nvme/project10-vla/cache/huggingface/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"

dataset = LeRobotDataset("lerobot/libero", root=DATA, episodes=[0], video_backend="pyav",
                         delta_timestamps={"action": [i / 10 for i in range(50)]})
cfg = PreTrainedConfig.from_pretrained(CKPT)
cfg.pretrained_path = CKPT; cfg.device = "cuda"
policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
pre_overrides = {
    "device_processor": {"device": "cuda"},
    "normalizer_processor": {"stats": dataset.meta.stats, "features": {**policy.config.input_features, **policy.config.output_features}, "norm_map": policy.config.normalization_mapping},
    "rename_observations_processor": {"rename_map": {}},
}
post_overrides = {"unnormalizer_processor": {"stats": dataset.meta.stats, "features": policy.config.output_features, "norm_map": policy.config.normalization_mapping}}
preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT,
                                                       preprocessor_overrides=pre_overrides, postprocessor_overrides=post_overrides)
preprocessor.save_pretrained(CKPT)
postprocessor.save_pretrained(CKPT)
print("SAVED")

sd = load_file(f"{CKPT}/policy_preprocessor_step_5_normalizer_processor.safetensors")
print("state stats in ckpt:", {k: [round(float(v.min()), 4), round(float(v.max()), 4)] for k, v in sd.items() if "state" in k})
ds_state = dataset.meta.stats["observation.state"]
print("dataset state min/max:", [round(float(x), 4) for x in ds_state["min"]], [round(float(x), 4) for x in ds_state["max"]])
