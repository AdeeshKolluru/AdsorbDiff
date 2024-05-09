import sys
import subprocess
import os
import argparse

# These are different commands to launch training for diffusion models.
def diffusion_training(gpus, model):
    pretrain = f"python -u -m torch.distributed.launch \
               --nproc_per_node={gpus} --master_port=1235 \
               main.py --mode train \
               --config-yml configs/denoising/eqv2_so3.yml \
               --distributed --amp --identifier pretrainis2rstrain200_fewshot_std0.1-10_so30.01-1.55_lr1e-4 \
               --optim.lr_initial=1.e-4"
    equiformerv2 = f"python -u -m torch.distributed.launch \
               --nproc_per_node={gpus} --master_port=1235 \
               main.py --mode train \
               --config-yml configs/denoising/eqv2_conditional.yml \
               --distributed --amp --identifier FTis2rstrain200_cond_std0.1-10_so30.01-1.55_wf4 \
               --optim.lr_initial=1.e-4"

    painn = f"python -u -m torch.distributed.launch \
               --nproc_per_node={gpus} --master_port=1234 \
               main.py --mode train \
               --config-yml configs/denoising/painn_so3.yml \
               --distributed  --identifier pretrainis2rs_sde_std0.1-10_so30.01-1.55_painn_new"
    gemnet_oc = f"python -u -m torch.distributed.launch \
               --nproc_per_node={gpus} --master_port 1234 \
               main.py --mode train \
               --config-yml configs/denoising/gemnet_so3.yml \
               --distributed  --identifier pretrainis2rs_sde_std0.1-10_so30.01-1.55_gemnet"
    return eval(model)


def sampling_and_relaxation(ngpus=1, nsite=1):
    out_path = f"/home/jovyan/shared-scratch/adeesh/denoising/valid_rerun/eqv2_conditional_FT"
    #ckpt_path = "/home/jovyan/repos/ocp-modeling/checkpoints/2024-01-08-13-05-04-pretrainis2rs_sde_std0.1-10_so30.01-1.55_painn/checkpoint.pt"
    ckpt_path = "/home/jovyan/shared-scratch/adeesh/adsorbdiff_ckpts/eqv2_pt_FT_conditional.pt"
    relax_ckpt_path = (
        "/home/jovyan/shared-scratch/adeesh/ckpts/gemnet_oc_base_s2ef_2M.pt"
    )
    val_id = "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/lmdbs_train_subsplits/val_nonrelaxed_update"
    val_ood = "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/lmdbs/valood50_R1I0.1"
    final_cmd = ""
    for step in range(nsite):
        if step != 0:
            final_cmd += " && \n"
        step_path = f"{out_path}/{step}"
        com_sde = f"python -u -m torch.distributed.launch \
                --nproc_per_node={ngpus} --master_port 1235 \
                main.py --mode run-relaxations \
                --config-yml configs/denoising/eqv2_conditional.yml \
                --task.relax_dataset.src={val_id} \
                --task.relax_opt.traj_dir={step_path} \
                --checkpoint {ckpt_path} \
                --distributed --amp  --model.sampling=True --seed {step} --debug"
        lmdb = f"python scripts/pred_traj_to_lmdb.py \
                --data-path {step_path} \
                --out-path {step_path}/final_struct_lmdb \
                --num-workers 4"
        com = f"python -u -m torch.distributed.launch \
                --nproc_per_node={ngpus} --master_port 1235 \
                main.py --mode run-relaxations \
                --config-yml configs/relaxation/gemnet_oc/gemnet_relax.yml \
                --checkpoint {relax_ckpt_path} \
                --task.relax_dataset.src={step_path}/final_struct_lmdb \
                --task.relax_opt.traj_dir={step_path}/relaxations \
                --distributed --amp --debug"
        cmd = com_sde + " && " + lmdb + " && " + com
        final_cmd += cmd
    return final_cmd


if __name__ == "__main__":

    # For training

    # command = diffusion_training(1, "painn")

    # To perform sampling and relaxation
    command = sampling_and_relaxation()

    with open("submit.sh", "w") as f:
        f.write(command)
    f.close()
    p = subprocess.Popen(["bash", "submit.sh"])
    p.wait()
