#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
redis_writer_node.py
====================

Subscribes to /semantic_graph (std_msgs/String, JSON-encoded) and writes
the latest snapshot into Redis where it is consumed by the global
planner and other navigation modules.

Redis layout
------------
    <prefix>:latest           STRING   - full JSON snapshot, overwritten each tick
    <prefix>:robot_pose       STRING   - JSON of just the robot pose+twist (cheap reads)
    <prefix>:node:<node_id>   HASH     - per-landmark hash for SCAN/MGET workflows
    <prefix>:nodes_index      ZSET     - node_id scored by last_seen_sec (for TTL sweeps)
    <prefix>:updated_at       STRING   - unix epoch seconds of last write
    <prefix>:pubsub channel   <prefix> - the JSON is also PUBLISHed on this channel
                                          so subscribers can react without polling.

Why both KV + pub/sub?
    Pub/sub gives instant reactivity to other modules (e.g. a planner that
    needs to replan as soon as a new obstacle is detected). The KV store
    gives a snapshot any new subscriber can pick up without waiting for
    the next publish. Together they cover both poll and push consumers.

This node is intentionally tolerant of Redis being unavailable: if the
connection drops it will log, back off, and keep retrying without
crashing the node. That's important because the combiner upstream is
still useful even when Redis is down (the topic still flows).
"""

from __future__ import annotations

import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String

try:
    import redis
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "redis-py is not installed. Install with 'pip install redis' "
        "(or 'pip3 install redis' in the Isaac ROS Docker container)."
    ) from exc


class RedisWriter(Node):

    def __init__(self) -> None:
        super().__init__('redis_writer')

        self.declare_parameter('graph_topic', '/semantic_graph')
        self.declare_parameter('redis_host', 'localhost')
        self.declare_parameter('redis_port', 6379)
        self.declare_parameter('redis_db', 0)
        self.declare_parameter('redis_password', '')
        self.declare_parameter('redis_key_prefix', 'semantic_graph')
        self.declare_parameter('write_full_snapshot', True)
        self.declare_parameter('write_per_node_hashes', True)
        self.declare_parameter('publish_pubsub', True)
        self.declare_parameter('snapshot_ttl_sec', 0)
        self.declare_parameter('reconnect_backoff_sec', 2.0)

        self._topic = self.get_parameter('graph_topic').value
        self._prefix = self.get_parameter('redis_key_prefix').value
        self._write_full = bool(self.get_parameter('write_full_snapshot').value)
        self._write_nodes = bool(self.get_parameter('write_per_node_hashes').value)
        self._publish_pubsub = bool(self.get_parameter('publish_pubsub').value)
        self._ttl = int(self.get_parameter('snapshot_ttl_sec').value)
        self._reconnect_backoff = float(self.get_parameter('reconnect_backoff_sec').value)

        self._redis_kwargs = dict(
            host=self.get_parameter('redis_host').value,
            port=int(self.get_parameter('redis_port').value),
            db=int(self.get_parameter('redis_db').value),
            decode_responses=True,
            socket_keepalive=True,
            socket_connect_timeout=2.0,
        )
        password = self.get_parameter('redis_password').value
        if password:
            self._redis_kwargs['password'] = password

        self._redis: Optional[redis.Redis] = None
        self._last_reconnect_attempt = 0.0
        self._connect()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._sub = self.create_subscription(
            String, self._topic, self._on_graph, qos)

        self._write_count = 0
        self._error_count = 0
        self._stat_timer = self.create_timer(10.0, self._log_stats)

        self.get_logger().info(
            f"Redis writer up. Topic='{self._topic}', "
            f"redis={self._redis_kwargs['host']}:{self._redis_kwargs['port']}, "
            f"prefix='{self._prefix}'."
        )

    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        try:
            self._redis = redis.Redis(**self._redis_kwargs)
            self._redis.ping()
            self.get_logger().info("Connected to Redis.")
            return True
        except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
            self._redis = None
            self.get_logger().warn(f"Redis connection failed: {e}")
            return False

    def _ensure_connected(self) -> bool:
        if self._redis is not None:
            return True
        now = time.time()
        if now - self._last_reconnect_attempt < self._reconnect_backoff:
            return False
        self._last_reconnect_attempt = now
        return self._connect()

    # ------------------------------------------------------------------
    def _on_graph(self, msg: String) -> None:
        if not self._ensure_connected():
            self._error_count += 1
            return

        # Parse once - we need the dict for per-node writes and the raw
        # string for the snapshot value.
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON on {self._topic}: {e}")
            self._error_count += 1
            return

        try:
            self._write(msg.data, payload)
            self._write_count += 1
        except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
            self.get_logger().warn(f"Redis write failed, will reconnect: {e}")
            self._redis = None
            self._error_count += 1

    def _write(self, raw_json: str, payload: dict) -> None:
        assert self._redis is not None
        prefix = self._prefix
        now_sec = time.time()

        pipe = self._redis.pipeline(transaction=False)

        if self._write_full:
            key_latest = f"{prefix}:latest"
            pipe.set(key_latest, raw_json)
            if self._ttl > 0:
                pipe.expire(key_latest, self._ttl)

        # Robot pose as a separate small key for cheap polling.
        robot = payload.get("robot", {})
        if robot:
            key_pose = f"{prefix}:robot_pose"
            pipe.set(key_pose, json.dumps(robot, separators=(',', ':')))
            if self._ttl > 0:
                pipe.expire(key_pose, self._ttl)

        # Per-node hashes + index.
        if self._write_nodes:
            index_key = f"{prefix}:nodes_index"
            # Snapshot the current set of node IDs so we can prune ones
            # that disappeared from the graph (capacity eviction upstream).
            current_ids = set()
            for node in payload.get("nodes", []):
                nid = node.get("id")
                if not nid:
                    continue
                current_ids.add(nid)
                node_key = f"{prefix}:node:{nid}"
                # Flatten the dict to strings for HSET.
                flat = {k: json.dumps(v) if not isinstance(v, str) else v
                        for k, v in node.items()}
                pipe.hset(node_key, mapping=flat)
                if self._ttl > 0:
                    pipe.expire(node_key, self._ttl)
                pipe.zadd(index_key, {nid: float(node.get("last_seen_sec", now_sec))})

            # Prune the index by reading after the pipeline executes.
            # Doing this inside the same pipeline would require knowing
            # the existing IDs; we do a cheap diff below.

        pipe.set(f"{prefix}:updated_at", f"{now_sec:.6f}")

        if self._publish_pubsub:
            pipe.publish(prefix, raw_json)

        pipe.execute()

        # Best-effort prune of orphaned node keys. Done outside the pipe
        # to keep the hot path simple. Skipped if too many nodes to make
        # the SCAN cost worthwhile (the capacity bound in the combiner
        # keeps this bounded anyway).
        if self._write_nodes and len(payload.get("nodes", [])) > 0:
            try:
                index_key = f"{prefix}:nodes_index"
                stored_ids = set(self._redis.zrange(index_key, 0, -1))
                stale = stored_ids - current_ids
                if stale:
                    del_pipe = self._redis.pipeline(transaction=False)
                    for nid in stale:
                        del_pipe.delete(f"{prefix}:node:{nid}")
                        del_pipe.zrem(index_key, nid)
                    del_pipe.execute()
            except (redis.ConnectionError, redis.TimeoutError):
                # Not fatal - we'll catch it on the next write.
                pass

    # ------------------------------------------------------------------
    def _log_stats(self) -> None:
        self.get_logger().info(
            f"Redis writer: {self._write_count} writes, "
            f"{self._error_count} errors in last 10s"
        )
        self._write_count = 0
        self._error_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = RedisWriter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
