import json, time, torch
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
policy.eval()
preprocessor, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT)

item = dataset[0]
obs = {"observation.images.image": item["observation.images.image"].unsqueeze(0),
       "observation.images.image2": item["observation.images.image2"].unsqueeze(0),
       "observation.state": item["observation.state"].unsqueeze(0).float(),
       "task": [str(item["task"])]}
for k in obs:
    if isinstance(obs[k], torch.Tensor) and obs[k].dtype == torch.uint8:
        obs[k] = obs[k].float() / 255.0
processed = preprocessor(obs)

with torch.no_grad():
    a = policy.select_action(processed)  # warmup
    torch.cuda.synchronize()
    times = []
    for i in range(300):
        t0 = time.perf_counter()
        a = policy.select_action(processed)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
ts = sorted(times)
res = {"action_shape": list(a.shape), "action_finite": bool(torch.isfinite(a.float()).all()),
       "n_calls": len(ts),
       "mean_s": sum(ts) / len(ts), "p50_s": ts[len(ts) // 2], "p95_s": ts[int(len(ts) * 0.95)],
       "max_s": ts[-1], "forward_trigger_s": ts[-1]}
with open("/root/shared-nvme/project10-vla/logs/m4_latency.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
