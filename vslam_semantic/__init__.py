# SPDX-License-Identifier: Apache-2.0
"""
vslam_semantic
==============

ROS 2 package that wires together cuVSLAM (Isaac ROS Visual SLAM), NanoOWL
(open-vocabulary OWL-ViT detection on Jetson TensorRT), a Pixhawk IMU bridge
via uXRCE-DDS, and a Redis-backed semantic graph builder for a UAV carrying
an Intel RealSense D415.

Public API
----------
Four node `main` functions for programmatic launching or testing:

    >>> from vslam_semantic import (
    ...     px4_imu_bridge_main,
    ...     nanoowl_inference_main,
    ...     semantic_graph_combiner_main,
    ...     redis_writer_main,
    ... )

Pure helper functions (no ROS dependencies, safe to import anywhere):

    >>> from vslam_semantic import (
    ...     quat_to_rotmat,
    ...     triangulate_bearing_rays,
    ...     first_attr,
    ... )

Imports are lazy: importing `vslam_semantic` does NOT pull in `rclpy`,
`nanoowl`, `redis`, or `px4_msgs`. Each `main` function only loads its
dependencies when it's actually called. This lets you, for example:

  * Run the IMU bridge on a machine that doesn't have NanoOWL installed
  * Unit-test the math helpers without a ROS environment
  * Import `vslam_semantic.__version__` from a deployment script
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Keep this in sync with package.xml and setup.py
__version__ = '0.1.0'

__author__ = 'Sharvesh'
__license__ = 'Apache-2.0'


# ---------------------------------------------------------------------------
# Public symbols. These are exported at package level via __getattr__ below
# so they are only loaded on first access (lazy import).
# ---------------------------------------------------------------------------
__all__ = [
    # Package metadata
    '__version__',
    '__author__',
    '__license__',
    # Node entry points (each pulls in rclpy + its own deps on first call)
    'px4_imu_bridge_main',
    'nanoowl_inference_main',
    'semantic_graph_combiner_main',
    'redis_writer_main',
    # Pure helpers (no ROS / numpy-only)
    'quat_to_rotmat',
    'triangulate_bearing_rays',
    'first_attr',
]


# Map of public name -> (module suffix, attribute name within that module).
# `module suffix` is appended to `vslam_semantic.` to form the import target.
#
# Helpers point at `_helpers` (numpy-only, no ROS deps) so they can be
# imported on machines that don't have rclpy installed.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Node mains - each pulls in rclpy + that node's specific deps on access
    'px4_imu_bridge_main':
        ('px4_imu_bridge_node', 'main'),
    'nanoowl_inference_main':
        ('nanoowl_inference_node', 'main'),
    'semantic_graph_combiner_main':
        ('semantic_graph_combiner_node', 'main'),
    'redis_writer_main':
        ('redis_writer_node', 'main'),
    # Pure helpers - numpy-only
    'quat_to_rotmat':
        ('_helpers', 'quat_to_rotmat'),
    'triangulate_bearing_rays':
        ('_helpers', 'triangulate_bearing_rays'),
    'first_attr':
        ('_helpers', 'first_attr'),
}


def __getattr__(name: str) -> Any:
    """PEP-562 lazy-import hook.

    Called by the interpreter for attribute accesses that aren't already
    defined at module level. We use it to defer the cost of importing
    `rclpy`, `nanoowl`, etc. until someone actually needs them.
    """
    if name in _LAZY_EXPORTS:
        module_suffix, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(f'{__name__}.{module_suffix}')
        value = getattr(module, attr_name)
        # Cache on the package module so subsequent accesses skip the lookup.
        globals()[name] = value
        return value
    raise AttributeError(f"module 'vslam_semantic' has no attribute {name!r}")


def __dir__() -> list[str]:
    """So tab-completion and ``dir(vslam_semantic)`` show the lazy names too."""
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


# For static analysers and IDE autocomplete: declare the names without
# actually importing them at runtime. TYPE_CHECKING is False at runtime,
# so this block never executes.
if TYPE_CHECKING:  # pragma: no cover
    from ._helpers import (  # noqa: F401
        first_attr,
        quat_to_rotmat,
        triangulate_bearing_rays,
    )
    from .nanoowl_inference_node import main as nanoowl_inference_main  # noqa: F401
    from .px4_imu_bridge_node import main as px4_imu_bridge_main  # noqa: F401
    from .redis_writer_node import main as redis_writer_main  # noqa: F401
    from .semantic_graph_combiner_node import main as semantic_graph_combiner_main  # noqa: F401
