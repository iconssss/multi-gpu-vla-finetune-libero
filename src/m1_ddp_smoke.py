import json, os, time, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.configs import PreTrainedConfig

BASE = "/root/shared-nvme/project10-vla/cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
DATA = "/root/shared-nvme/project10-vla/cache/huggingface/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
OUT = "/root/shared-nvme/project10-vla/logs"


def main():
    rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"]); world = int(os.environ["WORLD_SIZE"])
    B = int(os.environ.get("PER_DEV_BS", "4")); STEPS = int(os.environ.get("STEPS", "20")); WARM = 3
    TAG = os.environ.get("TAG", f"{world}gpu_bs{B}")
    torch.manual_seed(0)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    t0 = time.perf_counter()
    dataset = LeRobotDataset("lerobot/libero", root=DATA, episodes=[0], video_backend="pyav",
                             delta_timestamps={"action": [i / 10 for i in range(50)]})
    cfg = PreTrainedConfig.from_pretrained(BASE)
    cfg.pretrained_path = BASE; cfg.device = "cuda"; cfg.input_features = {}
    policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
    pre_overrides = {
        "device_processor": {"device": "cuda"},
        "normalizer_processor": {"stats": dataset.meta.stats, "features": {**policy.config.input_features, **policy.config.output_features}, "norm_map": policy.config.normalization_mapping},
        "rename_observations_processor": {"rename_map": {}},
    }
    post_overrides = {"unnormalizer_processor": {"stats": dataset.meta.stats, "features": policy.config.output_features, "norm_map": policy.config.normalization_mapping}}
    preprocessor, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=BASE,
                                               preprocessor_overrides=pre_overrides, postprocessor_overrides=post_overrides)
    net = DDP(policy, device_ids=[local_rank], find_unused_parameters=os.environ.get("FIND_UNUSED", "0") == "1")
    net.train()
    opt = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=1e-5)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=False)
    loader = DataLoader(dataset, batch_size=B, sampler=sampler, num_workers=4, drop_last=True,
                        persistent_workers=True, prefetch_factor=2)
    it = iter(loader)
    load_s = time.perf_counter() - t0

    times = []; losses = []; finite_all = True; err = None
    t_start = time.perf_counter()
    try:
        for step in range(WARM + STEPS):
            try:
                batch = next(it)
            except StopIteration:
                sampler.set_epoch(step); it = iter(loader); batch = next(it)
            for key in dataset.meta.camera_keys:
                if key in batch and batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].float() / 255.0
            processed = preprocessor(batch)
            if step == WARM - 1:
                torch.cuda.reset_peak_memory_stats()
            ts = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss, _ = net(processed)
            finite_all = finite_all and bool(torch.isfinite(loss))
            loss.backward()
            if step == WARM or step == WARM + STEPS - 1:
                for p in net.parameters():
                    if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                        finite_all = False
            opt.step()
            torch.cuda.synchronize()
            if step >= WARM:
                times.append(time.perf_counter() - ts)
                losses.append(float(loss.detach()))
    except Exception as e:
        err = repr(e)[:500]
    wall_s = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated()
    local = torch.tensor([sum(times) / max(len(times), 1), max(times) if times else 0.0, peak,
                          1.0 if (finite_all and err is None) else 0.0], device="cuda", dtype=torch.float64)
    gathered = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    if rank == 0:
        means = [g[0].item() for g in gathered]; maxs = [g[1].item() for g in gathered]
        peaks = [g[2].item() for g in gathered]; oks = [g[3].item() for g in gathered]
        mean_dt = sum(means) / world
        res = {"tag": TAG, "world_size": world, "batch_per_gpu": B, "global_batch": world * B,
               "measured_steps": len(times), "step_time_mean_s": mean_dt, "step_time_max_s": max(maxs),
               "samples_per_s": world * B / mean_dt if mean_dt > 0 else 0.0,
               "vram_peak_bytes_per_rank": peaks, "loss_mean": (sum(losses) / len(losses)) if losses else None,
               "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
               "loss_and_grads_finite": bool(all(oks)), "load_init_s": load_s, "train_wall_s": wall_s,
               "dataset_size": len(dataset), "err": err,
               "status": "PASS" if (err is None and all(oks)) else "FAIL"}
        with open(f"{OUT}/m1_{TAG}_result.json", "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
