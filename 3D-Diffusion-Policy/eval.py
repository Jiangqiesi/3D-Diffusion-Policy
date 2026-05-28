if __name__ == "__main__":
    import sys
    import os
    import pathlib

    PKG_DIR = pathlib.Path(__file__).resolve().parent
    ROOT_DIR = str(PKG_DIR.parent)
    sys.path.insert(0, str(PKG_DIR))
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    main()
