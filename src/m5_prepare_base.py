from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs import PreTrainedConfig, FeatureType
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.policies import make_pre_post_processors
BASE=Path("/root/shared-nvme/project10-vla/cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205")
OUT=Path("/root/shared-nvme/project10-vla/runs/m5_base_native")
DATA="/root/shared-nvme/project10-vla/cache/huggingface/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
ds=LeRobotDataset("lerobot/libero",root=DATA,episodes=[0],video_backend="pyav",delta_timestamps={"action":[i/10 for i in range(50)]})
cfg=PreTrainedConfig.from_pretrained(BASE)
features=dataset_to_policy_features(ds.meta.features)
cfg.input_features={k:v for k,v in features.items() if v.type is not FeatureType.ACTION}
cfg.output_features={k:v for k,v in features.items() if v.type is FeatureType.ACTION}
cfg.pretrained_path=str(OUT); cfg.device="cuda"
OUT.mkdir(parents=True,exist_ok=True); cfg.save_pretrained(OUT)
preo={"device_processor":{"device":"cuda"},"normalizer_processor":{"stats":ds.meta.stats,"features":{**cfg.input_features,**cfg.output_features},"norm_map":cfg.normalization_mapping},"rename_observations_processor":{"rename_map":{}}}
posto={"unnormalizer_processor":{"stats":ds.meta.stats,"features":cfg.output_features,"norm_map":cfg.normalization_mapping}}
pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(BASE),preprocessor_overrides=preo,postprocessor_overrides=posto)
pre.save_pretrained(OUT); post.save_pretrained(OUT)
model=OUT/"model.safetensors"
if not model.exists(): model.symlink_to(BASE/"model.safetensors")
print("BASE_NATIVE_READY")
