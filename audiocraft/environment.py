# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Runtime environment definition for AudioCraft.
"""

import socket
import os
import typing as tp
from pathlib import Path
from typing import Union


from omegaconf import OmegaConf

from .utils import cluster


class AudioCraftEnvironment:
    _instance: tp.Optional["AudioCraftEnvironment"] = None

    @staticmethod
    def get_user() -> str:
        return os.environ["USER"]

    @staticmethod
    def get_slurm_dir(slurm_dir: str) -> str:
        if slurm_dir == "":
            return ""
        else:
            return os.path.join(slurm_dir, AudioCraftEnvironment.get_user())

    @staticmethod
    def get_slurm_exclude() -> tp.Union[str, None]:
        return os.environ.get("AUDIOCRAFT_SLURM_EXCLUDE", default=None)

    @staticmethod
    def get_fair_cluster() -> bool:
        return cluster.get_cluster_type() in {
            cluster.ClusterType.FAIR,
            cluster.ClusterType.RSC,
        }

    @staticmethod
    def is_aws() -> bool:
        return cluster.get_cluster_type() == cluster.ClusterType.AWS

    def get_cache_dir(self) -> str:
        return self.config.cache_dir

    def get_config_dir(self) -> str:
        return f"{os.path.dirname(os.path.dirname(__file__))}/conf/cluster"

    @staticmethod
    def get_dora_dir() -> str:
        if AudioCraftEnvironment.get_fair_cluster() or AudioCraftEnvironment.is_aws():
            return f"/checkpoint/{AudioCraftEnvironment.get_user()}/experiments"
        else:
            return f"{os.path.expanduser('~')}/tmp/dora"

    def __init__(self) -> None:
        self.fully_qualified_domain_name = socket.getfqdn()
        self.cluster = cluster.get_cluster_type().value
        self.config = self._get_cluster_config()

    def _get_cluster_config(self) -> tp.Dict:
        if self.cluster == "windows":
            # Return a default configuration for Windows
            return {
                "account_number": None,
                "job_kwargs": {"cpus_per_gpu": 1, "num_gpus": 1},
                "name": "windows",
                "oss_upload_dir": None,
                "oss_download_dir": None,
                "ssh": {},
            }
        else:
            return self.config[self.cluster]
    @staticmethod
    def resolve_reference_path(path: Union[str, Path]) -> str:
        """
        Resolve the reference path for a given path.
        """
        if isinstance(path, Path):
            path_str = str(path)
        else:
            path_str = path

        if path_str.startswith("dora:"):
            path_str = path_str.replace("dora:", "", 1)
            base_path = AudioCraftEnvironment.get_dora_dir()
            return os.path.join(base_path, path_str)
        return str(path)
    @classmethod
    def instance(cls) -> "AudioCraftEnvironment":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_cache_dir(self) -> str:
        return self.config["cache_dir"]

    def get_ssh_config(self, cluster: str) -> tp.Dict:
        return OmegaConf.to_container(self.config["ssh"][cluster], resolve=True)