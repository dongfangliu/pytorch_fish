from .coupled_env import coupled_env
import gym_fish.envs.lib.pyflare as fl

import os


def get_full_path(asset_path):
    if asset_path.startswith("/"):
        full_path = asset_path
    else:
        full_path = os.path.join(os.path.dirname(__file__), asset_path)

    if not os.path.exists(full_path):
        raise IOError("File %s does not exist" % full_path)

    return full_path
class fish_env_basic(coupled_env):
    def __init__(self, fluid_json: str='assets/fluid/fluid_param_0.5.json', rigid_json: str='assets/rigids/rigids_4_30_new.json', gpuId: int=0, couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY) -> None:
        fluid_json = get_full_path(fluid_json)
        rigid_json = get_full_path(rigid_json)
        super().__init__(fluid_json, rigid_json, gpuId, couple_mode=couple_mode)
