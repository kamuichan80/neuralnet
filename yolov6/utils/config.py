#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Minimal stand-in for YOLOv6's mmcv-based Config: loads a configs/*.py file's top-level
dict variables (e.g. `model = dict(...)`) into an attrdict-style object.
"""

import importlib.util
import os


class Config(dict):
    """Dict with attribute access, e.g. `cfg.model.backbone.base_channels`."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name, value):
        self[name] = value

    @staticmethod
    def fromfile(path):
        path = os.path.abspath(path)
        module_name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cfg = {
            name: value for name, value in vars(module).items()
            if not name.startswith('__') and isinstance(value, dict)
        }
        return Config(cfg)
