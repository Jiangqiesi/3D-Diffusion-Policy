# Examples:
# bash scripts/train_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 dexart_laptop 0322 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 metaworld_basketball 0602 0 0



DEBUG=False
save_ckpt=True

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
site_packages_dir="$(find "${workspace_root}/.venv/lib" -maxdepth 1 -type d -name 'python*' | head -n 1)/site-packages"
nvjitlink_lib="${site_packages_dir}/nvidia/nvjitlink/lib"
cusparse_lib="${site_packages_dir}/nvidia/cusparse/lib"
extra_ld_library_path=""
if [ -d "${nvjitlink_lib}" ]; then
  extra_ld_library_path="${nvjitlink_lib}:${extra_ld_library_path}"
fi
if [ -d "${cusparse_lib}" ]; then
  extra_ld_library_path="${cusparse_lib}:${extra_ld_library_path}"
fi
export LD_LIBRARY_PATH="${extra_ld_library_path}${LD_LIBRARY_PATH:-}"

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd "${workspace_root}/3D-Diffusion-Policy"


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}



                                
