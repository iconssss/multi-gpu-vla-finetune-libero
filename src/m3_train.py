import json, os, time, random, shutil
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.configs import PreTrainedConfig

BASE = "/root/shared-nvme/project10-vla/cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
DATA = os.environ.get("LIBERO_SNAP", "/root/shared-nvme/project10-vla/cache/huggingface/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
RUNS = "/root/shared-nvme/project10-vla/runs"
LOGS = "/root/shared-nvme/project10-vla/logs"
RUN_NAME = os.environ.get("RUN_NAME", "m3_main_5k")
STEPS = int(os.environ.get("STEPS", "5000"))
B = 16
LR = 1e-5
SEED = 42
SAVE_AT = [1000, 2500, 5000]
LOG_EVERY = 10
PROCESSOR_FILES = ["policy_preprocessor.json", "policy_postprocessor.json",
                   "policy_preprocessor_step_5_normalizer_processor.safetensors",
                   "policy_postprocessor_step_0_unnormalizer_processor.safetensors"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def save_ckpt(policy, opt, step, d):
    os.makedirs(d, exist_ok=True)
    method = "save_pretrained"
    try:
        policy.save_pretrained(d)
    except Exception as e:
        method = "state_dict_fallback: " + repr(e)[:200]
        torch.save(policy.state_dict(), os.path.join(d, "model_state.pt"))
    torch.save(opt.state_dict(), os.path.join(d, "optimizer.pt"))
    for fn in PROCESSOR_FILES:
        src = os.path.join(BASE, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(d, fn))
    have = all(os.path.exists(os.path.join(d, fn)) for fn in PROCESSOR_FILES)
    with open(os.path.join(d, "train_meta.json"), "w") as f:
        json.dump({"step": step, "lr": LR, "seed": SEED, "batch_per_gpu": B, "save_method": method,
                   "processor_files_self_contained": have,
                   "base_checkpoint": BASE, "dataset": "lerobot/libero", "data_snapshot": DATA}, f, indent=2)
    return method


def main():
    rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"]); world = int(os.environ["WORLD_SIZE"])
    run_dir = f"{RUNS}/{RUN_NAME}"
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
    set_seed(SEED)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    t_start = time.perf_counter()
    dataset = LeRobotDataset("lerobot/libero", root=DATA, video_backend="pyav",
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
    net = DDP(policy, device_ids=[local_rank])
    net.train()
    opt = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=LR)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=SEED, drop_last=False)
    loader = DataLoader(dataset, batch_size=B, sampler=sampler, num_workers=4, drop_last=True,
                        persistent_workers=True, prefetch_factor=2)
    it = iter(loader)
    load_s = time.perf_counter() - t_start

    metrics_path = f"{LOGS}/{RUN_NAME}_metrics.jsonl"
    if rank == 0:
        open(metrics_path, "w").close()
        with open(f"{run_dir}/base_meta.json", "w") as f:
            json.dump({"base_checkpoint": BASE, "seed": SEED, "lr": LR, "steps": STEPS,
                       "batch_per_gpu": B, "global_batch": world * B, "world_size": world,
                       "features_in": {k: [str(v.type.value), list(v.shape)] for k, v in policy.config.input_features.items()},
                       "features_out": {k: [str(v.type.value), list(v.shape)] for k, v in policy.config.output_features.items()},
                       "parameters": sum(p.numel() for p in policy.parameters()),
                       "dataset_size": len(dataset)}, f, indent=2)

    step_times = []; window_losses = []; window_steps = []; stall_count = 0; err = None; nan_inf = False
    ckpt_info = {}
    t_train = time.perf_counter()
    try:
        for step in range(1, STEPS + 1):
            try:
                batch = next(it)
            except StopIteration:
                sampler.set_epoch(step); it = iter(loader); batch = next(it)
            for key in dataset.meta.camera_keys:
                if key in batch and batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].float() / 255.0
            processed = preprocessor(batch)
            if step == 3:
                torch.cuda.reset_peak_memory_stats()
            ts = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss, _ = net(processed)
            if not torch.isfinite(loss):
                nan_inf = True; raise RuntimeError(f"non-finite loss at step {step}: {float(loss)}")
            loss.backward()
            if step == 1 or step % 50 == 0:
                for p in net.parameters():
                    if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                        nan_inf = True; raise RuntimeError(f"non-finite grad at step {step}")
            opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - ts
            if step > 3:
                step_times.append(dt)
                if dt > 1.0:
                    stall_count += 1
            if step == 1 or step % LOG_EVERY == 0:
                lt = torch.tensor([float(loss.detach())], device="cuda", dtype=torch.float64)
                dist.all_reduce(lt, op=dist.ReduceOp.AVG)
                window_losses.append(lt.item()); window_steps.append(step)
                if rank == 0:
                    recent = step_times[-LOG_EVERY:]
                    rec = {"step": step, "loss_mean": lt.item(), "lr": opt.param_groups[0]["lr"],
                           "step_time_mean_recent_s": sum(recent) / max(len(recent), 1),
                           "samples_per_s": world * B * len(recent) / max(sum(recent), 1e-9)}
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
            if step in SAVE_AT:
                if rank == 0:
                    ckpt_info[step] = save_ckpt(net.module, opt, step, f"{run_dir}/step_{step}")
                dist.barrier()
    except Exception as e:
        err = repr(e)[:500]
    train_s = time.perf_counter() - t_train
    peak = torch.cuda.max_memory_allocated()

    ok = torch.tensor([1.0 if err is None else 0.0], device="cuda", dtype=torch.float64)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    peaks = [torch.zeros(1, device="cuda", dtype=torch.float64) for _ in range(world)]
    dist.all_gather(peaks, torch.tensor([peak], device="cuda", dtype=torch.float64))
    if rank == 0:
        def wmean(lo, hi):
            v = [l for l, s in zip(window_losses, window_steps) if lo <= s <= hi]
            return sum(v) / len(v) if v else None
        def roll_min(k=10):
            if len(window_losses) < k:
                return min(window_losses) if window_losses else None
            return min(sum(window_losses[i:i + k]) / k for i in range(len(window_losses) - k + 1))
        mean_dt = sum(step_times) / max(len(step_times), 1)
        summary = {"run": RUN_NAME, "status": "PASS" if (err is None and ok.item() > 0) else "FAIL",
                   "err": err, "nan_inf": nan_inf, "steps_done": len(step_times) + 3,
                   "world_size": world, "batch_per_gpu": B, "global_batch": world * B, "lr": LR, "seed": SEED,
                   "load_init_s": load_s, "train_s": train_s,
                   "step_time_mean_s": mean_dt, "step_time_max_s": max(step_times) if step_times else None,
                   "samples_per_s": world * B / mean_dt if mean_dt > 0 else 0.0,
                   "vram_peak_bytes_per_rank": [p.item() for p in peaks],
                   "stall_steps_gt1s": stall_count,
                   "first100_mean_loss": wmean(1, 100),
                   "around1000_mean_loss": wmean(900, 1100),
                   "around2500_mean_loss": wmean(2400, 2600),
                   "last100_mean_loss": wmean(STEPS - 99, STEPS),
                   "min_smoothed_loss_100w": roll_min(10),
                   "ckpt": ckpt_info}
        with open(f"{LOGS}/{RUN_NAME}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
