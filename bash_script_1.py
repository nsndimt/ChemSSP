import itertools
import os

import itertools
import os

if __name__ == "__main__":
    template = [
        "python",
        "fs_ner_train.py",
        "--task",
        "all_avg",
        "--bf16",
        "--tf32",
    ]
    
    target_source = {
        "Catalysis": ["SolidState", "PcMSP", "MSMention", "CHEMU", "WNUT"],
        "SolidState": ["Catalysis", "MSMention", "PcMSP", "CHEMU", "WNUT"],
        "PcMSP": ["Catalysis", "SolidState", "MSMention", "CHEMU", "WNUT"],
        "MSMention": ["Catalysis", "SolidState", "PcMSP", "CHEMU", "WNUT"],
        "CHEMU": ["Catalysis","SolidState", "PcMSP", "MSMention", "WNUT"],
        "WNUT": ["Catalysis","SolidState", "PcMSP", "MSMention", "CHEMU"],
        }
    
    wandb_proj = "exp1"
    model = "protospan"
    backbone = "s2orc-scibert-cased"

    bs = 1
    train_step = 1000
    Kshot = 10
    Qtest = 10


    Num_GPU = 4
    gpu_fp = []
    for gpu_id in range(Num_GPU):
        gpu_fp.append(open(f"exp1_train_{gpu_id}.sh", "w"))
    
    gpu_id = 0
    for seed in [12345, 23456, 34567, 45678, 56789]:
        for Kshot in [5, 10, 20, 40]:
            for target, source in target_source.items():
                cmd = template.copy()
                cmd.extend(["--wandb_proj", f'"{wandb_proj}"'])
                cmd.extend(["--model", f'"{model}"'])
                cmd.extend(["--backbone", f'"models/{backbone}"'])
                cmd.extend(["--seed", f'{seed}'])
                cmd.extend(["--bs", f"{bs}"])
                cmd.extend(["--train_step", f"{train_step}"])
                cmd.extend(["--Kshot", f'{Kshot}'])
                cmd.extend(["--Qtest", f'{Qtest}'])
                cmd.extend(["--source", f'"{",".join(source)}"'])
                cmd.extend(["--target", f'"{target}"'])
                ckpt_name = f'{model}_{backbone}_seed_{seed}_K{Kshot}_target_{target}_source_{",".join(source)}'
                assert len(ckpt_name) < 255
                ckpt_dir = f'"{os.path.join("checkpoint", ckpt_name)}"'
                assert not os.path.exists(ckpt_dir)
                cmd.extend(["--output_dir", ckpt_dir])
                cmd.extend(["--gpu", f'{gpu_id%2}'])
                gpu_fp[gpu_id].write(' '.join(cmd) + '\n')
                gpu_id = (gpu_id + 1) % Num_GPU
    
    for fp in gpu_fp:
        fp.close()