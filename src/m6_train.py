import json, os, time, random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.configs import PreTrainedConfig

RUNS="/root/shared-nvme/project10-vla/runs"; LOGS="/root/shared-nvme/project10-vla/logs"
RUN_NAME="m6_resume_20k"; RESUME=f"{RUNS}/m3_main_5k/step_5000"; START=5000; END=20000
B=16; LR=1e-5; SEED=42; SAVE_AT={10000,15000,20000}
DATA="/root/shared-nvme/project10-vla/cache/huggingface/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"

def seed(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def save_ckpt(policy,opt,pre,post,step,path):
 path=Path(path); path.mkdir(parents=True,exist_ok=True)
 policy.save_pretrained(path); pre.save_pretrained(path); post.save_pretrained(path); torch.save(opt.state_dict(),path/"optimizer.pt")
 files=["model.safetensors","config.json","optimizer.pt","train_meta.json","policy_preprocessor.json","policy_postprocessor.json","policy_preprocessor_step_5_normalizer_processor.safetensors","policy_postprocessor_step_0_unnormalizer_processor.safetensors"]
 (path/"train_meta.json").write_text(json.dumps({"step":step,"resume_from":RESUME,"optimizer_resumed":True,"lr":LR,"seed_lineage":SEED,"batch_per_gpu":B,"global_batch":64,"processor_files_self_contained":all((path/x).exists() for x in files[4:]),"files":files},indent=2))
 return all((path/x).exists() for x in files)

def main():
 rank=int(os.environ["RANK"]); local=int(os.environ["LOCAL_RANK"]); world=int(os.environ["WORLD_SIZE"]); seed(SEED); torch.cuda.set_device(local); dist.init_process_group("nccl")
 rd=Path(RUNS)/RUN_NAME
 if rank==0: rd.mkdir(parents=True,exist_ok=True)
 dist.barrier()
 ds=LeRobotDataset("lerobot/libero",root=DATA,video_backend="pyav",delta_timestamps={"action":[i/10 for i in range(50)]})
 cfg=PreTrainedConfig.from_pretrained(RESUME); cfg.pretrained_path=RESUME; cfg.device="cuda"
 policy=make_policy(cfg=cfg,ds_meta=ds.meta)
 preo={"device_processor":{"device":"cuda"},"normalizer_processor":{"stats":ds.meta.stats,"features":{**policy.config.input_features,**policy.config.output_features},"norm_map":policy.config.normalization_mapping},"rename_observations_processor":{"rename_map":{}}}
 posto={"unnormalizer_processor":{"stats":ds.meta.stats,"features":policy.config.output_features,"norm_map":policy.config.normalization_mapping}}
 pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=RESUME,preprocessor_overrides=preo,postprocessor_overrides=posto)
 net=DDP(policy,device_ids=[local]); net.train()
 opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=LR)
 opt.load_state_dict(torch.load(f"{RESUME}/optimizer.pt",map_location="cpu",weights_only=True))
 sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,seed=SEED,drop_last=False)
 dl=DataLoader(ds,batch_size=B,sampler=sampler,num_workers=4,drop_last=True,persistent_workers=True,prefetch_factor=2); it=iter(dl)
 mp=Path(LOGS)/f"{RUN_NAME}_metrics.jsonl"
 if rank==0: mp.write_text("")
 times=[]; losses=[]; steps=[]; t0=time.perf_counter(); nan=False; err=None; saved={}
 try:
  for step in range(START+1,END+1):
   try: batch=next(it)
   except StopIteration: sampler.set_epoch(step); it=iter(dl); batch=next(it)
   for k in ds.meta.camera_keys:
    if k in batch and batch[k].dtype==torch.uint8: batch[k]=batch[k].float()/255
   batch=pre(batch)
   if step==START+3: torch.cuda.reset_peak_memory_stats()
   ts=time.perf_counter(); opt.zero_grad(set_to_none=True); loss,_=net(batch)
   if not torch.isfinite(loss): nan=True; raise RuntimeError(f"nonfinite_loss_step_{step}")
   loss.backward()
   if step%50==0:
    for p in net.parameters():
     if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all(): nan=True; raise RuntimeError(f"nonfinite_grad_step_{step}")
   opt.step(); torch.cuda.synchronize(); dt=time.perf_counter()-ts
   if step>START+3: times.append(dt)
   if step%10==0:
    x=torch.tensor([loss.detach().item()],device="cuda",dtype=torch.float64); dist.all_reduce(x,op=dist.ReduceOp.AVG); losses.append(x.item()); steps.append(step)
    if rank==0:
     rec={"step":step,"loss_mean":x.item(),"lr":LR,"step_time_mean_recent_s":sum(times[-10:])/max(1,len(times[-10:])),"samples_per_s":64*min(10,len(times))/max(sum(times[-10:]),1e-9)}
     with mp.open("a") as f:f.write(json.dumps(rec)+"\n")
   if step in SAVE_AT:
    if rank==0: saved[step]=save_ckpt(net.module,opt,pre,post,step,rd/f"step_{step}")
    dist.barrier()
 except Exception as e: err=repr(e)[:500]
 ok=torch.tensor([float(err is None)],device="cuda",dtype=torch.float64); dist.all_reduce(ok,op=dist.ReduceOp.MIN)
 peak=torch.cuda.max_memory_allocated(); peaks=[torch.zeros(1,device="cuda",dtype=torch.float64) for _ in range(world)]; dist.all_gather(peaks,torch.tensor([peak],device="cuda",dtype=torch.float64))
 if rank==0:
  def mean(lo,hi):
   a=[v for s,v in zip(steps,losses) if lo<=s<=hi]; return sum(a)/len(a) if a else None
  smooth=min((sum(losses[i:i+10])/10 for i in range(max(0,len(losses)-9))),default=None)
  summary={"status":"PASS" if ok.item()==1 else "FAIL","err":err,"resume_from":RESUME,"optimizer_resumed":True,"start_step":START,"end_step":END,"steps_done":steps[-1] if steps else START,"train_s":time.perf_counter()-t0,"step_time_mean_s":sum(times)/max(1,len(times)),"samples_per_s":64/(sum(times)/max(1,len(times))),"loss_5k_window":None,"loss_10k_window":mean(9901,10000),"loss_15k_window":mean(14901,15000),"loss_20k_window":mean(19901,20000),"min_smoothed_loss_100w":smooth,"peak_vram_bytes_per_rank":[x.item() for x in peaks],"nan_inf":nan,"nccl":"PASS" if ok.item()==1 else "FAIL","checkpoints":saved}
  (Path(LOGS)/f"{RUN_NAME}_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
 dist.barrier(); dist.destroy_process_group()
if __name__=="__main__": main()
