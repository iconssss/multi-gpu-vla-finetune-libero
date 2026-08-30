import random, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SNAP = "/root/shared-nvme/project10-vla/cache/huggingface/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
d = LeRobotDataset("lerobot/libero", root=SNAP, video_backend="pyav",
                   delta_timestamps={"action": [i / 10 for i in range(50)]})
print("frames", len(d), "episodes", d.meta.total_episodes, "tasks", d.meta.total_tasks)
print("camera_keys", list(d.meta.camera_keys))
random.seed(7)
n = len(d)
for s in [n // 7, n // 2, (5 * n) // 6]:
    item = d[s]
    imgs = {k: tuple(v.shape) for k, v in item.items() if k.startswith("observation.images")}
    finite = all(torch.isfinite(v.float()).all() for k, v in item.items() if isinstance(v, torch.Tensor))
    var_ok = all(float(v.float().var()) > 0 for k, v in item.items() if k.startswith("observation.images"))
    print("episode", int(item["episode_index"]), "idx", s, imgs, "state", tuple(item["observation.state"].shape),
          "action", tuple(item["action"].shape), "task", str(item["task"])[:40], "finite", finite, "video_nontrivial", var_ok)
print("VERIFY DONE")
