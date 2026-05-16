#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
semantic_graph_combiner_node.py
================================

Fuses cuVSLAM pose output with NanoOWL detection output to maintain a
running 3D semantic graph of the environment.

Inputs
------
    /visual_slam/tracking/vo_pose_covariance  (PoseWithCovarianceStamped)
        Current robot pose in the `odom` frame from cuVSLAM. We prefer
        this over /visual_slam/tracking/odometry because the latter's
        covariance has been observed to contain zeros sporadically, which
        breaks downstream EKF fusion. The pose itself is identical between
        the two topics.

    /visual_slam/tracking/odometry            (nav_msgs/Odometry)
        Used only for the twist (linear + angular velocity), which the
        global planner may want as part of the latest snapshot.

    /nanoowl/detections                       (vision_msgs/Detection2DArray)
        NanoOWL inference output. Each detection has an image-plane bbox
        and a class label.

    /camera/infra1/camera_info                (sensor_msgs/CameraInfo)
        Latched. Used to back-project bbox centroids to camera bearings.

Outputs
-------
    /semantic_graph   (std_msgs/String)
        Serialized JSON of the current graph snapshot. The redis_writer
        node subscribes to this and writes it to Redis.

Graph data model
----------------
    {
      "schema_version": 1,
      "header": {
        "stamp_sec": float,            # ROS wall time in seconds
        "frame_id": "odom",
      },
      "robot": {
        "pose": [x, y, z, qx, qy, qz, qw],
        "twist": [vx, vy, vz, wx, wy, wz],
        "pose_covariance": [...36 floats...]
      },
      "nodes": [
        {
          "id": "n_42",
          "label": "fire extinguisher",
          "confidence": 0.83,
          "position_world": [x, y, z],        # estimated, may be approximate
          "bearing_from_robot": [bx, by, bz], # unit vector at last sighting
          "observation_count": 7,
          "last_seen_sec": float,
          "first_seen_sec": float,
          "observation_pose": [x, y, z, qx, qy, qz, qw]  # robot pose at last sight
        },
        ...
      ],
      "edges": [
        {"src": "n_42", "dst": "n_17", "type": "near", "distance": 0.92}
      ]
    }

Design notes
------------
  * Detections do not include depth. We approximate landmark positions by
    placing them along the bearing ray at a fixed default range (a
    parameter, `default_landmark_range_m`), and refine when the same label
    is observed from multiple poses (multi-view triangulation, simple
    least-squares of bearing rays). This is intentionally simple - the
    planner only needs "what's roughly where" - and is the appropriate
    level of effort given that NanoOWL outputs 2D boxes, not depth.
  * Spatial deduplication: when a new detection's nearest-neighbour
    landmark with the same label is within `spatial_merge_radius_m`, we
    merge into the existing node rather than creating a new one.
  * The graph is bounded by `max_graph_nodes`; oldest-by-last_seen nodes
    are evicted when the cap is exceeded.
  * Edges of type `near` are recomputed lazily on each publish for the K
    closest neighbour pairs (cheap; the graph is small).

This module also exposes the graph as an in-memory dict on `self.graph`
so the redis_writer_node could in principle import and use it directly,
but the topic-based decoupling makes the system easier to test.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

try:
    from vision_msgs.msg import Detection2DArray
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "vision_msgs is not installed. Install with "
        "'sudo apt install ros-humble-vision-msgs'."
    ) from exc


# ----------------------------------------------------------------------
# Internal data structures
# ----------------------------------------------------------------------

@dataclass
class SemanticNode:
    """One landmark in the graph."""
    node_id: str
    label: str
    confidence: float
    position_world: np.ndarray            # shape (3,)
    last_bearing_world: np.ndarray        # shape (3,), unit vector
    observation_count: int = 1
    first_seen_sec: float = 0.0
    last_seen_sec: float = 0.0
    observation_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))
    # Stored bearing rays from each observation pose, used for refinement.
    # Each entry is (origin_world (3,), bearing_world unit (3,)).
    _bearing_rays: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.node_id,
            "label": self.label,
            "confidence": float(self.confidence),
            "position_world": self.position_world.tolist(),
            "bearing_from_robot": self.last_bearing_world.tolist(),
            "observation_count": self.observation_count,
            "first_seen_sec": self.first_seen_sec,
            "last_seen_sec": self.last_seen_sec,
            "observation_pose": self.observation_pose.tolist(),
        }


# ----------------------------------------------------------------------
# Math helpers (re-exported from _helpers for backward compatibility)
# ----------------------------------------------------------------------
from ._helpers import quat_to_rotmat, triangulate_bearing_rays  # noqa: F401


# ----------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------

class SemanticGraphCombiner(Node):

    def __init__(self) -> None:
        super().__init__('semantic_graph_combiner')

        # Parameters
        self.declare_parameter('pose_topic', '/visual_slam/tracking/vo_pose_covariance')
        self.declare_parameter('odom_topic', '/visual_slam/tracking/odometry')
        self.declare_parameter('detections_topic', '/nanoowl/detections')
        self.declare_parameter('camera_info_topic', '/camera/infra1/camera_info')
        self.declare_parameter('graph_topic', '/semantic_graph')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('sync_slop_sec', 0.10)
        self.declare_parameter('spatial_merge_radius_m', 0.50)
        self.declare_parameter('confidence_threshold', 0.30)
        self.declare_parameter('max_graph_nodes', 2000)
        self.declare_parameter('default_landmark_range_m', 2.0)
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('edge_top_k', 8)
        self.declare_parameter('edge_max_distance_m', 3.0)

        self._pose_topic = self.get_parameter('pose_topic').value
        self._odom_topic = self.get_parameter('odom_topic').value
        self._det_topic = self.get_parameter('detections_topic').value
        self._cam_info_topic = self.get_parameter('camera_info_topic').value
        self._graph_topic = self.get_parameter('graph_topic').value
        self._world_frame = self.get_parameter('world_frame').value
        self._sync_slop = float(self.get_parameter('sync_slop_sec').value)
        self._merge_radius = float(self.get_parameter('spatial_merge_radius_m').value)
        self._conf_thresh = float(self.get_parameter('confidence_threshold').value)
        self._max_nodes = int(self.get_parameter('max_graph_nodes').value)
        self._default_range = float(self.get_parameter('default_landmark_range_m').value)
        self._edge_k = int(self.get_parameter('edge_top_k').value)
        self._edge_max_d = float(self.get_parameter('edge_max_distance_m').value)
        publish_rate = float(self.get_parameter('publish_rate_hz').value)

        # State
        self._latest_pose: Optional[PoseWithCovarianceStamped] = None
        self._latest_odom: Optional[Odometry] = None
        self._camera_info: Optional[CameraInfo] = None
        self._nodes: Dict[str, SemanticNode] = {}
        self._next_node_id = 0

        # QoS profiles - vo_pose_covariance comes in at the image rate
        # (60 Hz), so KEEP_LAST(10) is enough.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._sub_pose = self.create_subscription(
            PoseWithCovarianceStamped, self._pose_topic, self._on_pose, sensor_qos)
        self._sub_odom = self.create_subscription(
            Odometry, self._odom_topic, self._on_odom, sensor_qos)
        self._sub_det = self.create_subscription(
            Detection2DArray, self._det_topic, self._on_detections, sensor_qos)
        self._sub_cam = self.create_subscription(
            CameraInfo, self._cam_info_topic, self._on_camera_info, latched_qos)

        self._pub_graph = self.create_publisher(String, self._graph_topic, 5)

        self._publish_timer = self.create_timer(
            1.0 / max(publish_rate, 0.1), self._publish_graph)

        self.get_logger().info(
            f"Semantic graph combiner up. "
            f"Subscribing to: pose='{self._pose_topic}', "
            f"detections='{self._det_topic}', cam_info='{self._cam_info_topic}'. "
            f"Publishing on '{self._graph_topic}' at {publish_rate} Hz."
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        self._latest_pose = msg

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _on_detections(self, msg: Detection2DArray) -> None:
        if self._latest_pose is None or self._camera_info is None:
            return

        # Approximate-time sync: only fuse if the pose is fresh enough
        # relative to the detection. cuVSLAM at 60 Hz means pose is at
        # most ~16 ms old; NanoOWL inference on Orin is ~30-50 ms, so a
        # 100 ms slop comfortably covers both.
        det_t = self._stamp_to_sec(msg.header.stamp)
        pose_t = self._stamp_to_sec(self._latest_pose.header.stamp)
        if abs(det_t - pose_t) > self._sync_slop:
            self.get_logger().warn(
                f"Dropping detection batch: |t_det - t_pose| = "
                f"{abs(det_t - pose_t):.3f}s > slop {self._sync_slop:.3f}s",
                throttle_duration_sec=2.0,
            )
            return

        # Cache the robot pose at observation time.
        p = self._latest_pose.pose.pose
        robot_pos_world = np.array([p.position.x, p.position.y, p.position.z])
        R_world_cam = quat_to_rotmat(
            p.orientation.x, p.orientation.y,
            p.orientation.z, p.orientation.w,
        )
        observation_pose = np.array([
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y,
            p.orientation.z, p.orientation.w,
        ])

        # Camera intrinsics for back-projection.
        K = self._camera_info.k  # row-major 9-vec
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]
        if fx <= 0 or fy <= 0:
            return  # invalid info, skip

        now_sec = time.time()

        for det in msg.detections:
            if not det.results:
                continue

            # vision_msgs Detection2D.results is ObjectHypothesisWithPose;
            # different distros put the label on .hypothesis.class_id or
            # directly on .id - handle both for compatibility.
            best = det.results[0]
            score = float(getattr(best, 'score', 0.0)
                          or getattr(getattr(best, 'hypothesis', None),
                                     'score', 0.0))
            label = (getattr(getattr(best, 'hypothesis', None), 'class_id', None)
                     or getattr(best, 'id', None)
                     or 'unknown')
            if score < self._conf_thresh:
                continue

            # Bbox centroid in pixel coordinates.
            cx_px = det.bbox.center.position.x if hasattr(det.bbox.center, 'position') \
                else det.bbox.center.x
            cy_px = det.bbox.center.position.y if hasattr(det.bbox.center, 'position') \
                else det.bbox.center.y

            # Pinhole back-projection to a unit bearing in CAMERA OPTICAL
            # frame (REP-103: x right, y down, z forward).
            x_norm = (cx_px - cx) / fx
            y_norm = (cy_px - cy) / fy
            bearing_cam = np.array([x_norm, y_norm, 1.0])
            bearing_cam /= np.linalg.norm(bearing_cam)

            # Camera-optical -> base/world. cuVSLAM's base_frame is
            # camera_link, but vo_pose_covariance is published as the pose
            # of base_frame in world frame. We approximate the optical
            # frame as a fixed rotation of camera_link (REP-103):
            #     R_cam_link__cam_optical = [[0,0,1],[-1,0,0],[0,-1,0]]
            # so bearing in camera_link = R_cl_co @ bearing_cam.
            R_cl_co = np.array([
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ])
            bearing_link = R_cl_co @ bearing_cam
            bearing_world = R_world_cam @ bearing_link
            bn = np.linalg.norm(bearing_world)
            if bn < 1e-9:
                continue
            bearing_world /= bn

            # Initial position guess: along the bearing at default range.
            guess_world = robot_pos_world + bearing_world * self._default_range

            # Find existing landmark with matching label within merge radius.
            merged_id = None
            for nid, node in self._nodes.items():
                if node.label != label:
                    continue
                if np.linalg.norm(node.position_world - guess_world) < self._merge_radius:
                    merged_id = nid
                    break

            if merged_id is not None:
                node = self._nodes[merged_id]
                node._bearing_rays.append((robot_pos_world.copy(), bearing_world.copy()))
                # Refine position with all rays so far.
                node.position_world = triangulate_bearing_rays(
                    node._bearing_rays, fallback_range=self._default_range)
                node.last_bearing_world = bearing_world
                node.observation_count += 1
                node.last_seen_sec = now_sec
                node.observation_pose = observation_pose
                # Confidence: running max (good enough for planner gating).
                node.confidence = max(node.confidence, score)
            else:
                self._next_node_id += 1
                nid = f"n_{self._next_node_id}"
                node = SemanticNode(
                    node_id=nid,
                    label=str(label),
                    confidence=score,
                    position_world=guess_world,
                    last_bearing_world=bearing_world,
                    first_seen_sec=now_sec,
                    last_seen_sec=now_sec,
                    observation_pose=observation_pose,
                )
                node._bearing_rays.append((robot_pos_world.copy(), bearing_world.copy()))
                self._nodes[nid] = node

        self._enforce_capacity()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _enforce_capacity(self) -> None:
        if len(self._nodes) <= self._max_nodes:
            return
        # Evict by oldest last_seen_sec.
        keep = sorted(self._nodes.items(),
                      key=lambda kv: kv[1].last_seen_sec,
                      reverse=True)[:self._max_nodes]
        self._nodes = dict(keep)

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def _publish_graph(self) -> None:
        if self._latest_pose is None:
            return

        p = self._latest_pose.pose.pose
        pose_list = [p.position.x, p.position.y, p.position.z,
                     p.orientation.x, p.orientation.y,
                     p.orientation.z, p.orientation.w]
        cov_list = list(self._latest_pose.pose.covariance)

        if self._latest_odom is not None:
            t = self._latest_odom.twist.twist
            twist_list = [t.linear.x, t.linear.y, t.linear.z,
                          t.angular.x, t.angular.y, t.angular.z]
        else:
            twist_list = [0.0] * 6

        nodes_serialised = [node.to_dict() for node in self._nodes.values()]
        edges = self._compute_proximity_edges()

        snapshot = {
            "schema_version": 1,
            "header": {
                "stamp_sec": time.time(),
                "frame_id": self._world_frame,
            },
            "robot": {
                "pose": pose_list,
                "twist": twist_list,
                "pose_covariance": cov_list,
            },
            "nodes": nodes_serialised,
            "edges": edges,
        }

        out = String()
        out.data = json.dumps(snapshot, separators=(',', ':'))
        self._pub_graph.publish(out)

    def _compute_proximity_edges(self) -> List[dict]:
        ids = list(self._nodes.keys())
        if len(ids) < 2:
            return []
        positions = np.stack([self._nodes[i].position_world for i in ids])
        # Pairwise distances (only the strict upper triangle).
        edges: List[Tuple[float, str, str]] = []
        for i in range(len(ids)):
            diffs = positions[i + 1:] - positions[i]
            dists = np.linalg.norm(diffs, axis=1)
            for j_off, d in enumerate(dists):
                if d <= self._edge_max_d:
                    edges.append((float(d), ids[i], ids[i + 1 + j_off]))
        edges.sort()
        return [
            {"src": s, "dst": d_, "type": "near", "distance": dist}
            for dist, s, d_ in edges[: self._edge_k * len(ids)]
        ]


def main(args=None):
    rclpy.init(args=args)
    node = SemanticGraphCombiner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
