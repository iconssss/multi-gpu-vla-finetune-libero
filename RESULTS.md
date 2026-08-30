# RESULTS — M0 → M7-R

All training/eval runs use the frozen configuration: `lerobot/smolvla_base` + `lerobot/libero`, 4×RTX 4090 DDP, bf16, AdamW lr 1e-5, global batch 64, seed 42 (training).

## M0 — Environment & native contract gate: PASS

- torch 2.7.1+cu126, torchvision 0.22.1+cu126, LeRobot v0.6.0 (pinned commit), pip check PASS.
- Native contract accepted by official config path: state 8D, `observation.images.image` + `image2` (3×256×256), action 7D, task language → `observation.language.tokens` [1,48].
- Real LIBERO batch, official processor, forward loss 0.841 (finite), backward, optimizer step, 8-step tiny train (1.1 s, peak 1.69 GiB).
- `lerobot/smolvla_libero` checkpoint **rejected**: 6D state + 3-camera contract incompatible with native 8D/2-camera LIBERO data; no official projection mechanism; adapters forbidden.

## M1 — 4-GPU DDP throughput & stability: PASS

| Config | step time | samples/s | VRAM/rank |
|---|---|---|---|
| 1×GPU bs4 | 0.166 s | 24.1 | 2.45 GB |
| 4×GPU bs4 (global 16) | 0.174 s | 92.0 | 2.45 GB |
| 4×GPU bs8 (global 32) | 0.188 s | 170.3 | 3.33 GB |
| 4×GPU bs16 (global 64) | 0.221 s | 289.3 | 5.06 GB |

- **Speedup 3.82×** at identical per-GPU batch; loss/grads finite, no NCCL errors, no OOM, zero stalls.
- Selected formal config: bs16/GPU, global 64, grad-accum 1, bf16, AdamW 1e-5.

## M2 — Full data + 1K fast run: PASS

- Full `lerobot/libero`: **1693 episodes / 273,465 frames**, cached 1.9 GB; random-episode video decode verified.
- 1000 steps: first-50 mean loss **2.564** → last-50 **0.908**; runtime 307.7 s; peak 5.06 GB/rank; checkpoints self-contained; reload + real-batch forward PASS.

## M3 — 5K main run: PASS

- 5000 steps in 1389 s (252.8 samples/s), zero stalls, no NaN/Inf.
- Window mean loss: first100 **1.983** → ~1000 **0.915** → ~2500 **0.740** → last100 **0.665** (min smoothed 0.650).
- Checkpoints step_1000/2500/5000 saved with correct LIBERO normalizer stats (base default stats were found missing from `save_pretrained` output and were officially re-saved from dataset stats).

## M4 — Official closed-loop evaluation: PARTIAL

- Protocol: official LeRobot v0.6.0 evaluator, official `.pruned_init` states 0–9, seed 1000, official success criterion, 256×256 dual camera, receding chunk execution.
- **LIBERO-Spatial (FULL 100 episodes): SR = 4%** (4/100; per-task range 0–30%).
- object/goal/long suites: not completed — the node exposes a single EGL device, which crashed GPU1–3 render workers; root cause + fix (`CUDA_VISIBLE_DEVICES="N,0"` + `MUJOCO_EGL_DEVICE_ID=0`) identified but full runs deprioritized by early-stop audit.
- Inference latency (real batch, GPU): mean **8.6 ms** / p50 **3.0 ms** / p95 **4.3 ms** per env step; chunk-generation forward 0.28 s (once per 50 env steps).

## M5 — Checkpoint comparison: 0/15 (flagged)

base / step1000 / step2500 / step5000 on three Spatial tasks × 5 states: all 0/15. Flagged immediately: M4's task-2 full result was 3/10 at the same checkpoint, so a 0/5 subsample cannot refute fine-tuning (sampling ambiguity).

## M5B — Replanning-horizon test: CHUNK EFFECT = NONE

step5000, task 2, 10 official init states, chunk_size 50 fixed:

| Executed actions per replan | Successes | Success states |
|---|---|---|
| 50 (default) | 3/10 | [1,4,6] |
| 10 | 1/10 | [5] |
| 5 | 0/10 | [] |

More frequent replanning does **not** help; the 50-step action queue is not the bottleneck.

## M6 — Resume 5K→20K: PASS (training) / rollout comparison confounded

- +15,000 steps with optimizer-state resume in 3947.8 s (266.1 samples/s), no NaN/Inf.
- Window mean loss: ~10K **0.5947** → ~15K **0.5545** → ~20K **0.5349** (min smoothed 0.5171).
- Uncontrolled rollout reading at the time (5K 4/30, 10K 0/30, 20K 3/30) suggested "degrades" — **retired**: see M7/M7-R.

## M7 — Paired matrix + RNG audit: confound identified

- Success-state overlap across checkpoints was near-zero even where capabilities should overlap.
- Root cause: SmolVLA rollout samples flow-matching action noise with `torch.normal`; official eval seeds RNG once per run, so task execution **order changes each checkpoint's noise stream** → raw cross-checkpoint comparisons are not paired. Verdict at the time: RNG-comparability BLOCKED.

## M7-R — Controlled checkpoint rollout (final scaling conclusion)

Protocol: Spatial tasks {0,2,6} × official init states 0–9, per-episode `policy_seed = 20260831 + task_id*100 + init_state` synchronized across Python/NumPy/Torch/CUDA, plus `VLAFlowMatching.sample_noise` backed by per-element `torch.Generator` — identical noise streams for both checkpoints. 30 episodes per checkpoint.

| Checkpoint | task0 | task2 | task6 | total |
|---|---|---|---|---|
| 5K | 0/10 | 2/10 (states [2,4]) | 0/10 | **2/30** |
| 20K | 0/10 | 3/10 (states [1,4,7]) | 1/10 (state [9]) | **4/30** |

Paired transitions: retained 1 (task2 state 4), lost 1 (task2 state 2), newly gained 3 (task2 [1,7], task6 [9]).

## FINAL VERDICT

**WEAK POSITIVE TRAINING-SCALE SIGNAL.**

- Offline loss improves monotonically (0.665 → 0.535); controlled closed-loop success moves 2/30 → 4/30.
- The sample is small; **no statistical significance is claimed**, and training scale clearly has not solved the closed-loop bottleneck.
- The earlier M6 "DEGRADES" reading is **retired** — it was produced under RNG/task-order confounding and reversed direction once noise streams were controlled.

## Artifacts

- `results/m3_main_5k_metrics.jsonl`, `results/m6_resume_20k_metrics.jsonl` — per-10-step training metrics.
- `results/m7r_controlled_rollout.csv`, `results/m7r_paired_matrix.csv`, `results/m7r_summary.json` — controlled evaluation.
- `results/m4_all_episodes.csv`, `results/m4_summary.json`, `results/m4_latency.json`, `results/m5*`, `results/m5b*` — earlier diagnostics.
- `figures/` — A training loss, B DDP scaling, C controlled success, D paired transition heatmap.
