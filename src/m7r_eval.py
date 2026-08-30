import os


def _run():
    import random
    import numpy as np
    import torch
    task_id = int(os.environ["M7R_TASK_ID"])
    base = 20260831 + task_id * 100
    random.seed(base); np.random.seed(base); torch.manual_seed(base); torch.cuda.manual_seed_all(base)
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
    _generators = {}

    def controlled_noise(self, shape, device):
        b = shape[0]; key = (str(device), b)
        if key not in _generators:
            _generators[key] = [torch.Generator(device=device).manual_seed(base + i) for i in range(b)]
        rows = [torch.normal(0.0, 1.0, size=(1, *shape[1:]), dtype=torch.float32, device=device, generator=g)
                for g in _generators[key]]
        return torch.cat(rows, dim=0)

    VLAFlowMatching.sample_noise = controlled_noise
    from lerobot.scripts.lerobot_eval import main
    main()


if __name__ == "__main__":
    _run()
