# Multi-GPU VLA Fine-Tuning, Closed-Loop Evaluation & Failure Diagnosis

End-to-end pipeline that fine-tunes **SmolVLA** (450M params) on **LIBERO** with 4×RTX 4090 DDP, evaluates it with the **official closed-loop rollout protocol**, and diagnoses why **offline training loss keeps improving while closed-loop success barely moves** — including the discovery and removal of a **stochastic-rollout RNG confound** in checkpoint comparisons.

## What this project is

> A real VLA training system → 4-GPU DDP scaling → standardized closed-loop evaluation → offline/online metric divergence identified → stochastic rollout evaluation confound diagnosed → paired RNG-controlled checkpoint evaluation → a trustworthy training-scale verdict.

It is **not** "SmolVLA reaches high success on LIBERO". The headline scientific outcome is a *negative-plus-methodological* one: with training loss dropping 1.98 → 0.53, controlled closed-loop success went only 2/30 → 4/30 — and the only reason that comparison is trustworthy at all is the per-episode paired RNG control designed in M7-R.

## Highlights

- **Native data/model contract** — 8D proprio state, 2×256×256 RGB cameras, 7D action, language task tokens; the official `lerobot/smolvla_libero` checkpoint was rejected because its 6D/3-camera contract cannot accept the native LIBERO observations without undocumented adapters.
- **4×RTX 4090 DDP** — bf16, global batch 64, 0.22–0.25 s/step, ~289 samples/s, **3.82× speedup** over single GPU at identical per-GPU batch, ~5 GB VRAM/rank (>75% headroom on 24 GB).
- **20K-step scaling study** — 5K→10K→15K→20K checkpoints, loss 0.665→0.535 with optimizer-state resume.
- **Official closed-loop evaluation** — LeRobot v0.6.0 evaluator, official LIBERO init states and success criterion; policy inference latency mean 8.6 ms / p95 4.3 ms per env step (action-chunk forward 0.28 s every 50 steps).
- **RNG confound diagnosis (key methodological contribution)** — SmolVLA's flow-matching rollout samples action noise via `torch.normal`; checkpoint comparisons made eval runs consume different RNG streams (task-order dependent), making raw comparisons non-paired. M7-R fixes this with `policy_seed = 20260831 + task_id*100 + init_state`, synchronized across Python/NumPy/Torch/CUDA plus a per-element `torch.Generator` inside `VLAFlowMatching.sample_noise` — identical noise streams for both checkpoints.
- **Final verdict: WEAK POSITIVE TRAINING-SCALE SIGNAL** — controlled 5K: 2/30 vs 20K: 4/30. Small sample; no significance claimed; the earlier "scaling degrades" reading is retired as RNG-confounded.

## Pipeline (M0 → M7-R)

| Stage | What | Outcome |
|---|---|---|
| M0 | Env + native contract gate (load, real batch, fwd/bwd/opt, tiny train) | PASS |
| M1 | 4-GPU DDP throughput/stability smoke (bs 4/8/16 per GPU) | PASS, 3.82× |
| M2 | Full dataset (1693 eps / 273,465 frames) + 1K fast run + ckpt reload | PASS |
| M3 | 5K main run, self-contained checkpoints | PASS |
| M4 | Official LIBERO rollout eval (Spatial full 100 eps: 4% SR) | PARTIAL (EGL device limits) |
| M5 | Checkpoint comparison (base/1K/2.5K/5K) | 0/15, sampling ambiguity flagged |
| M5B | Replanning-horizon test (execute 50/10/5 actions per chunk) | CHUNK EFFECT: NONE |
| M6 | Resume 5K→20K + uncontrolled scaling eval | loss ↓, success flat — **confounded** |
| M7 | Paired success matrix + RNG audit | Confound identified |
| M7-R | RNG-controlled 5K vs 20K rollout (60 eps) | WEAK POSITIVE signal |

## Repo layout

```
src/        training, DDP smoke, eval, RNG-control, aggregation scripts (all built on official LeRobot v0.6.0 APIs)
results/    compact JSON/CSV: per-step metrics, eval summaries, controlled rollout matrix
figures/    loss curve, DDP scaling, controlled success, paired transition heatmap
README.md / RESULTS.md / INTERVIEW_GUIDE.md
```

Key results in [RESULTS.md](RESULTS.md). Interview narration in [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md).

## Stack

PyTorch 2.7.1+cu126 · LeRobot v0.6.0 (pinned, unmodified) · SmolVLA base · lerobot/libero (AV1 video, PyAV backend) · torchrun/DDP/NCCL · MuJoCo/robosuite EGL rendering · 4×RTX 4090.

## Reproducibility notes

- All training/eval goes through official paths: `LeRobotDataset`, `make_policy`, `make_pre_post_processors`, official forward/loss, official `lerobot_eval` CLI. No core source patches anywhere.
- Checkpoints are made self-contained (model + config + optimizer + processor stats files) so eval never depends on hand-copied files.
- Eval protocol: official LIBERO `.pruned_init` init states, official success predicate, seed 1000; controlled runs add per-episode policy RNG (see M7-R).

## Future work (documented, not executed)

Task-conditioned offline action-error analysis; paired trajectory failure taxonomy; dataset/action-distribution diagnostics; targeted data balancing & failure-recovery data; larger-sample controlled closed-loop evaluation.
