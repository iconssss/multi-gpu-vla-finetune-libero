import csv, json

R = "/root/shared-nvme/project10-vla/logs/m7r/raw"
OUT = "/root/shared-nvme/project10-vla/logs"
tasks = [0, 2, 6]; cks = ["5k", "20k"]
data = {}; rows = []
for ck in cks:
    for t in tasks:
        x = json.load(open(f"{R}/{ck}_task{t}/eval_info.json"))
        succ = next(i["metrics"]["successes"] for i in x["per_task"] if i["task_id"] == t)
        data[(ck, t)] = [bool(s) for s in succ]
        for i, s in enumerate(succ):
            rows.append({"checkpoint": ck, "run": "m3_main_5k/step_5000" if ck == "5k" else "m6_resume_20k/step_20000",
                         "task_id": t, "init_state": i, "policy_seed": 20260831 + t * 100 + i, "success": bool(s)})
paired = []
for t in tasks:
    for i in range(10):
        paired.append({"task_id": t, "init_state": i, "ck5k": data[("5k", t)][i], "ck20k": data[("20k", t)][i]})

def ss(ck, t): return [i for i in range(10) if data[(ck, t)][i]]
summary = {}
for ck in cks:
    summary[ck] = {"per_task": {str(t): {"successes": sum(data[(ck, t)]), "success_states": ss(ck, t)} for t in tasks},
                   "total": sum(sum(data[(ck, t)]) for t in tasks)}
trans = {}
for t in tasks:
    s5, s20 = set(ss("5k", t)), set(ss("20k", t))
    trans[str(t)] = {"retained": sorted(s5 & s20), "lost": sorted(s5 - s20), "new": sorted(s20 - s5)}
tot5 = summary["5k"]["total"]; tot20 = summary["20k"]["total"]
if tot20 >= tot5 + 2:
    verdict = "TRAINING_SCALE_POSITIVE"
elif tot20 < tot5:
    verdict = "CONTROLLED_DEGRADATION_OBSERVED"
else:
    verdict = "TRAINING_SCALE_NO_GAIN"
summary["transitions"] = trans
summary["overall"] = {"5k_total": tot5, "20k_total": tot20, "episodes_per_ckpt": 30,
                      "retained": sum(len(v["retained"]) for v in trans.values()),
                      "lost": sum(len(v["lost"]) for v in trans.values()),
                      "new": sum(len(v["new"]) for v in trans.values())}
summary["rng_protocol"] = {"policy_seed": "20260831 + task_id*100 + init_state",
                           "mechanism": "VLAFlowMatching.sample_noise patched to per-batch-element torch.Generator; identical for both checkpoints"}
summary["verdict"] = verdict
with open(f"{OUT}/m7r_controlled_rollout.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(f"{OUT}/m7r_paired_matrix.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(paired[0].keys())); w.writeheader(); w.writerows(paired)
with open(f"{OUT}/m7r_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
