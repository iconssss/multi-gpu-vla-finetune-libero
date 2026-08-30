import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

L = "/root/shared-nvme/project10-vla/logs"
O = "/root/shared-nvme/project10-vla/figures"
os.makedirs(O, exist_ok=True)

# A. training loss vs step (M3 0-5k + M6 5k-20k)
xs, ys = [], []
for f in ["m3_main_5k_metrics.jsonl", "m6_resume_20k_metrics.jsonl"]:
    for line in open(f"{L}/{f}"):
        r = json.loads(line)
        xs.append(r["step"]); ys.append(r["loss_mean"])
xs = np.array(xs); ys = np.array(ys)
o = np.argsort(xs); xs, ys = xs[o], ys[o]
k = 20
sm = np.convolve(ys, np.ones(k) / k, mode="valid")
plt.figure(figsize=(7, 4))
plt.plot(xs, ys, alpha=0.25, lw=0.7, color="tab:blue")
plt.plot(xs[k - 1:], sm, lw=1.8, color="tab:blue", label="20-window smoothed")
for s, lab in [(5000, "5K"), (10000, "10K"), (15000, "15K"), (20000, "20K")]:
    plt.axvline(s, color="gray", ls=":", lw=0.8)
    plt.text(s, plt.ylim()[1] * 0.95, " " + lab, fontsize=7, color="gray", va="top")
plt.xlabel("optimizer step"); plt.ylabel("flow-matching loss")
plt.title("SmolVLA fine-tuning on lerobot/libero (4x RTX4090 DDP, bf16, lr 1e-5)")
plt.legend(frameon=False); plt.tight_layout()
plt.savefig(f"{O}/A_training_loss.png", dpi=150); plt.close()

# B. DDP scaling / throughput (M1)
cfgs = ["1 GPU\nbs=4", "4 GPU\nbs=4", "4 GPU\nbs=8", "4 GPU\nbs=16"]
tput = [24.1, 92.0, 170.3, 289.3]
plt.figure(figsize=(6, 4))
bars = plt.bar(cfgs, tput, color=["tab:gray", "tab:blue", "tab:blue", "tab:blue"])
for b, v in zip(bars, tput):
    plt.text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f}", ha="center", fontsize=9)
plt.annotate("3.82x vs single GPU\n(same per-GPU batch)", xy=(1, 92), xytext=(0.35, 210),
             arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9)
plt.ylabel("throughput (samples/s)"); plt.ylim(0, 330)
plt.title("DDP throughput scaling (450M-param SmolVLA, bf16)")
plt.tight_layout(); plt.savefig(f"{O}/B_ddp_scaling.png", dpi=150); plt.close()

# C/D. controlled 5K vs 20K
def succ(ck, t):
    x = json.load(open(f"{L}/m7r/raw/{ck}_task{t}/eval_info.json"))
    return [bool(s) for s in next(i["metrics"]["successes"] for i in x["per_task"] if i["task_id"] == t)]

tasks = [0, 2, 6]
s5 = [sum(succ("5k", t)) for t in tasks]
s20 = [sum(succ("20k", t)) for t in tasks]
fig, ax = plt.subplots(figsize=(6, 4))
xp = np.arange(3); w = 0.36
b1 = ax.bar(xp - w / 2, s5, w, label="step 5K  (2/30)", color="tab:gray")
b2 = ax.bar(xp + w / 2, s20, w, label="step 20K  (4/30)", color="tab:blue")
for bs in (b1, b2):
    for b in bs:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{int(b.get_height())}/10", ha="center", fontsize=9)
ax.set_xticks(xp); ax.set_xticklabels([f"task {t}" for t in tasks])
ax.set_ylabel("successes / 10 init states"); ax.set_ylim(0, 5.2)
ax.set_title("Controlled closed-loop eval (paired policy RNG, LIBERO-Spatial)")
ax.legend(frameon=False); plt.tight_layout()
plt.savefig(f"{O}/C_controlled_success.png", dpi=150); plt.close()

from matplotlib.colors import ListedColormap
M = np.zeros((3, 10))
for r, t in enumerate(tasks):
    a, b = succ("5k", t), succ("20k", t)
    for c in range(10):
        M[r, c] = 3 if (a[c] and b[c]) else (1 if a[c] else (2 if b[c] else 0))
cmap = ListedColormap(["#eeeeee", "#f4a261", "#4a7bb7", "#2a9d8f"])
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
labels = {0: "both fail", 1: "5K only", 2: "20K only", 3: "both success"}
for r in range(3):
    for c in range(10):
        ax.text(c, r, ["x", "5K", "20K", "OK"][int(M[r, c])], ha="center", va="center", fontsize=8)
ax.set_xticks(range(10)); ax.set_yticks(range(3)); ax.set_yticklabels([f"task {t}" for t in tasks])
ax.set_xlabel("official init state id")
ax.set_title("Paired success-state transitions under identical policy-noise streams")
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(v)) for v in range(4)]
ax.legend(handles, [labels[v] for v in range(4)], ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{O}/D_paired_transition.png", dpi=150, bbox_inches="tight"); plt.close()
print("FIGURES_DONE", os.listdir(O))
