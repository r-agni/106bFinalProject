# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os

import torch
from tensordict import TensorDict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, "cfg")


def init_simulation_app(cfg):
    # launch the simulator
    headless = bool(cfg.get("headless", True))
    default_display_options = 3094 if headless else 3286
    config = {
        "headless": headless,
        "anti_aliasing": int(cfg.get("anti_aliasing", 3)),
        "display_options": int(cfg.get("display_options", default_display_options)),
    }
    hide_ui = cfg.get("hide_ui", None)
    if hide_ui is not None:
        config["hide_ui"] = bool(hide_ui)
    viewer = cfg.get("viewer", None)
    resolution = viewer.get("resolution", None) if viewer is not None else None
    if resolution is not None:
        width, height = [int(v) for v in resolution]
        config.update(
            {
                "width": width,
                "height": height,
                "window_width": width,
                "window_height": height,
            }
        )
    for key in (
        "renderer",
        "samples_per_pixel_per_frame",
        "denoiser",
        "max_bounces",
        "max_specular_transmission_bounces",
        "max_volume_bounces",
        "subdiv_refinement_level",
    ):
        value = cfg.get(key, None)
        if value is not None:
            config[key] = value
    from isaacsim import SimulationApp
    simulation_app = SimulationApp(config)
    dlss_exec_mode = cfg.get("dlss_exec_mode", None)
    if dlss_exec_mode is not None:
        import carb

        settings = carb.settings.get_settings()
        dlss_exec_mode = int(dlss_exec_mode)
        settings.set("/rtx/post/dlss/execMode", dlss_exec_mode)
        settings.set("/rtx-defaults/post/dlss/execMode", dlss_exec_mode)
    return simulation_app

def _get_shapes(self: TensorDict):
    return {
        k: v.shape if isinstance(v, torch.Tensor) else v.shapes for k, v in self.items()
    }


def _get_devices(self: TensorDict):
    return {
        k: v.device if isinstance(v, torch.Tensor) else v.devices
        for k, v in self.items()
    }


TensorDict.shapes = property(_get_shapes)
TensorDict.devices = property(_get_devices)
