import csv, json, os

D = "/root/shared-nvme/project10-vla/logs/m4_eval"
SUITES = ["spatial", "object", "goal", "long"]
rows = []
summary = {"seed": 1000, "protocol": "lerobot v0.6.0 official lerobot_eval, official init states (.pruned_init idx 0-9), official success criterion", "suites": {}}
for s in SUITES:
    p = f"{D}/{s}/eval_info.json"
    if not os.path.exists(p):
        summary["suites"][s] = {"missing": True}
        continue
    info = json.load(open(p))
    succ = 0; n = 0; task_sr = {}
    for t in info["per_task"]:
        tid = t["task_id"]; ss = t["metrics"]["successes"]; rs = t["metrics"]["sum_rewards"]
        tsucc = int(sum(ss)); task_sr[tid] = tsucc / len(ss)
        succ += tsucc; n += len(ss)
        for i, (ok, r) in enumerate(zip(ss, rs)):
            rows.append({"suite": s, "task_id": tid, "episode": i, "seed": 1000,
                         "success": bool(ok), "sum_reward": r})
    summary["suites"][s] = {"successes": succ, "episodes": n, "SR": succ / n if n else None,
                            "task_SR_min": min(task_sr.values()) if task_sr else None,
                            "task_SR_max": max(task_sr.values()) if task_sr else None,
                            "eval_s": info["overall"].get("eval_s")}
tot_s = sum(v.get("successes", 0) for v in summary["suites"].values())
tot_n = sum(v.get("episodes", 0) for v in summary["suites"].values())
summary["overall"] = {"successes": tot_s, "episodes": tot_n, "SR": tot_s / tot_n if tot_n else None}

with open(f"{D}/m4_all_episodes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["suite", "task_id", "episode", "seed", "success", "sum_reward"])
    w.writeheader(); w.writerows(rows)
with open(f"{D}/m4_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
