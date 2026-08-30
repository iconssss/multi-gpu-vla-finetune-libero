# INTERVIEW GUIDE

Nine questions this project is built to answer, with the evidence to cite for each.

## 1. Why SmolVLA + LIBERO?

SmolVLA is a compact (450M) VLA with an official HF checkpoint, flow-matching action head, and language conditioning — small enough to fine-tune on 24 GB GPUs yet representative of modern VLA architecture (VLM backbone + action expert). LIBERO is the standard manipulation benchmark with a public demonstration dataset (`lerobot/libero`: 1693 episodes, 273k frames, 40 tasks) whose observation contract (8D proprio, 2 cameras, 7D delta-EE action, language) matches the model's capability envelope, plus deterministic official init states and success predicates for reproducible closed-loop evaluation.

## 2. Why drop the official `smolvla_libero` checkpoint?

Its config declares a 6D state and **three** cameras (`camera1/2/3`), while native LIBERO data is 8D state + **two** cameras. Its bundled preprocessor only renames two camera keys and offers no 8D→6D projection or third-camera synthesis. Using it would require hand-cropping state or fabricating a camera — undocumented adapters that silently change the data contract. I froze that route as BLOCKED and used `smolvla_base`, letting the official config path regenerate input features from real dataset metadata so the model natively accepts 8D + 2 cameras + 7D action.

## 3. How did you verify the native data/model contract?

Through the official API chain only: `LeRobotDataset` (PyAV video backend) → `PreTrainedConfig.from_pretrained` with `input_features={}` so features are regenerated from dataset metadata → `make_policy` → `make_pre_post_processors` → official training forward. Verified: input features (8D state, two 3×256×256 cameras), output 7D action, chunk_size 50 = n_action_steps 50, language tokenized to 48 tokens, finite loss (0.841), finite grads, real parameter updates, 8-step tiny train at 1.69 GiB — before any long training was allowed to start. Zero source patches (verified by clean git status on the pinned LeRobot checkout).

## 4. How was DDP configured, and why is speedup 3.82× and not 4×?

torchrun `--standalone --nproc-per-node 4`, PyTorch DDP over NCCL, one `DistributedSampler` shard per rank, bf16, per-GPU batch 16 (global 64), AdamW lr 1e-5, `num_workers=4` + `persistent_workers` + prefetch to keep video decode off the critical path. The measured 3.82× (92.0 vs 24.1 samples/s at identical per-GPU batch) is the honest number: the remaining ~5% is gradient all-reduce overlap plus per-rank dataloader jitter. Throughput per GPU *improves* with batch (24 → 72 samples/s/GPU from bs4→bs16) because the 450M model underutilizes a 4090 at small batches — which is also why 4×bs16 reached 289 samples/s with only 5 GB VRAM per card.

## 5. Why does falling offline loss not imply closed-loop success?

Flow-matching training loss measures one-step action prediction on demonstrator states. Closed-loop rollouts visit states the demonstrator never produced (covariate shift), and success requires 50-step open-loop chunks composed of actions that are individually plausible but jointly compounding errors. Empirically here: loss improved 1.98 → 0.665 (5K) → 0.535 (20K) while controlled Spatial success went 2/30 → 4/30. The replanning-horizon test (M5B) further showed the failure is not chunk staleness — replanning every 10 or 5 steps made success *worse* (3/10 → 1/10 → 0/10) — pointing at action-quality/observation-grounding limits rather than execution cadence.

## 6. Why was the original M6 checkpoint comparison unreliable?

SmolVLA's inference samples flow-matching initial noise with `torch.normal`. The official evaluator seeds RNG once per run, so the noise stream a checkpoint consumes depends on how many rollouts ran before it — i.e., on task execution order. M4 and M6 evaluated tasks in different orders, so "5K vs 20K" compared different noise realizations, not just different weights. The paired success-state matrix showed near-zero overlap across checkpoints, which is statistically implausible for related checkpoints and was the tell that the comparison wasn't paired. That reading ("scaling degrades") was retracted.

## 7. How do you make stochastic-policy checkpoint comparisons paired?

Freeze everything controllable per episode: identical official init state, identical env protocol, and identical policy-noise stream. Concretely (M7-R): `policy_seed = 20260831 + task_id*100 + init_state` applied at every episode start across Python/NumPy/Torch/CUDA RNG, plus replacing the noise source inside `VLAFlowMatching.sample_noise` with per-batch-element `torch.Generator`s seeded from the same formula — an inference-time hook, not a model/algorithm change. Both checkpoints then consume bit-identical noise sequences for the same (task, init state), so success differences are attributable to weights. A practical side lesson: when the eval driver crashed, it was because a custom eval entry script lacked an `if __name__ == "__main__"` guard — forkserver-based vector envs re-import the main module, so unguarded side effects (CUDA init, `main()`) run twice and break worker pipes.

## 8. Main failure-diagnosis conclusions of the project?

(1) Contract mismatch in the official LIBERO checkpoint — resolved by base + native features. (2) Missing processor stats in `save_pretrained` output — checkpoints must be made self-contained (config + model + optimizer + normalizer stats) or eval silently loads wrong normalization; fixed by re-saving the official processor state from dataset stats. (3) Infrastructure: single-EGL-device nodes need `CUDA_VISIBLE_DEVICES="N,0"` + `MUJOCO_EGL_DEVICE_ID=0` to place inference on GPU N while rendering on EGL device 0. (4) Chunk execution cadence is not the closed-loop bottleneck (M5B). (5) Offline loss ↓ ≠ closed-loop ↑; scale from 5K→20K gives at best a weak positive under controlled RNG (2/30→4/30, not significant). (6) Evaluation methodology itself was the biggest confound — controlled RNG changed the *direction* of the scaling conclusion.

## 9. With more time, what's the highest-value next step?

Task-conditioned offline action-error analysis (first-action/early-horizon MAE vs full-chunk MAE across 5K/10K/20K on identical fixed samples): it's cheap, fully deterministic, and discriminates "early-action quality never improved" from "full-chunk improved but closed loop didn't". Paired with a coarse failure taxonomy over the existing rollout videos (reach/grasp/timing/transport/oscillation) on the retained-vs-lost init states, it tells you whether to fix data (targeted recovery demonstrations, data balancing) or the objective. Then a larger-sample controlled closed-loop eval (the 30-episode scale here cannot resolve 2-point differences).
