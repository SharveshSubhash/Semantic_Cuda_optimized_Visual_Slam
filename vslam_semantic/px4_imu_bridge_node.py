#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
px4_imu_bridge_node.py
======================

Subscribes to /fmu/out/sensor_combined (px4_msgs/msg/SensorCombined) published
by the uXRCE-DDS agent and republishes the readings as sensor_msgs/msg/Imu on
/imu/data_raw, ready to be consumed by isaac_ros_visual_slam (or any other
ROS 2 component).

Frame conversion - this matters a lot:
    PX4 reports sensor data in the FRD body frame (Forward-Right-Down).
    ROS / REP-103 uses FLU (Forward-Left-Up).
    The mapping is therefore:
        x_FLU  =  x_FRD
        y_FLU  = -y_FRD
        z_FLU  = -z_FRD
    Failing to convert is the single most common mistake when bridging
    PX4 to ROS, and produces an upside-down gravity vector that makes
    cuVSLAM's IMU initialization fail silently.

Timestamp:
    SensorCombined.timestamp is microseconds since PX4 boot. Naively
    stamping the ROS message with this value puts it ~50 years in the
    past (epoch 0) and synchronisers reject it. We instead stamp with
    the ROS clock at reception, which is good enough for cuVSLAM's
    image_jitter_threshold (any sub-frame-period offset is absorbed).
    For tighter sync, run UXRCE_DDS_SYNCT=1 on the FCU and use the
    PX4 timesync_status topic to compute the offset; that's left as a
    follow-up since the simple approach is robust in flight.

Covariance:
    The SensorCombined message has no covariance field, so we synthesise
    diagonal covariance matrices from the user-supplied noise density and
    random walk parameters (these end up only being informational - the
    cuVSLAM node has its own gyro_noise_density / accel_noise_density
    parameters that it trusts).
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Imu

try:
    from px4_msgs.msg import SensorCombined
except ImportError as exc:  # pragma: no cover - dev-time check
    raise ImportError(
        "px4_msgs is not installed. Build it from "
        "https://github.com/PX4/px4_msgs into your ROS 2 workspace."
    ) from exc


# QoS used by the uXRCE-DDS agent for /fmu/out/* topics.
# It publishes with BEST_EFFORT + KEEP_LAST(5) + VOLATILE; subscribers using
# RELIABLE will silently see zero messages, which is why most people new to
# PX4-ROS 2 see an empty topic on first try.
PX4_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)


class Px4ImuBridge(Node):

    def __init__(self) -> None:
        super().__init__('px4_imu_bridge')

        self.declare_parameter('input_topic', '/fmu/out/sensor_combined')
        self.declare_parameter('output_topic', '/imu/data_raw')
        self.declare_parameter('imu_frame_id', 'imu_link')
        self.declare_parameter('gyro_noise_density', 0.00018)
        self.declare_parameter('gyro_random_walk', 1.0e-5)
        self.declare_parameter('accel_noise_density', 0.002)
        self.declare_parameter('accel_random_walk', 3.0e-3)

        self._in_topic = self.get_parameter('input_topic').value
        self._out_topic = self.get_parameter('output_topic').value
        self._frame_id = self.get_parameter('imu_frame_id').value

        # Synthesised variances. Variance = noise_density^2 (assumes 1 Hz
        # bandwidth normalisation; this is informational only for cuVSLAM).
        gnd = float(self.get_parameter('gyro_noise_density').value)
        and_ = float(self.get_parameter('accel_noise_density').value)
        self._gyro_var = gnd * gnd
        self._accel_var = and_ * and_

        self._sub = self.create_subscription(
            SensorCombined,
            self._in_topic,
            self._on_sensor_combined,
            PX4_PUB_QOS,
        )
        self._pub = self.create_publisher(Imu, self._out_topic, 50)

        self._msg_count = 0
        self._log_timer = self.create_timer(5.0, self._log_rate)
        self.get_logger().info(
            f"PX4 IMU bridge up: '{self._in_topic}' -> '{self._out_topic}' "
            f"(frame_id='{self._frame_id}')"
        )

    # ------------------------------------------------------------------
    def _on_sensor_combined(self, msg: SensorCombined) -> None:
        # Reject the message if the gyro integration period is zero - that
        # happens at startup before the FMU has produced its first sample,
        # and feeding it into cuVSLAM trips the jitter detector.
        if msg.gyro_integral_dt == 0 or msg.accelerometer_integral_dt == 0:
            return

        imu = Imu()
        imu.header.stamp = self.get_clock().now().to_msg()
        imu.header.frame_id = self._frame_id

        # FRD -> FLU conversion: invert Y and Z on both vectors.
        # gyro_rad is rad/s; accelerometer_m_s2 is m/s^2. Both arrays are
        # length 3, indices [0]=X, [1]=Y, [2]=Z in the FRD body frame.
        imu.angular_velocity.x = float(msg.gyro_rad[0])
        imu.angular_velocity.y = float(-msg.gyro_rad[1])
        imu.angular_velocity.z = float(-msg.gyro_rad[2])

        imu.linear_acceleration.x = float(msg.accelerometer_m_s2[0])
        imu.linear_acceleration.y = float(-msg.accelerometer_m_s2[1])
        imu.linear_acceleration.z = float(-msg.accelerometer_m_s2[2])

        # No orientation is provided by SensorCombined (PX4 publishes the
        # filtered attitude on a separate topic). Conform to REP-145 by
        # setting the orientation covariance [0] element to -1 to signal
        # "orientation not available".
        imu.orientation_covariance[0] = -1.0

        imu.angular_velocity_covariance[0] = self._gyro_var
        imu.angular_velocity_covariance[4] = self._gyro_var
        imu.angular_velocity_covariance[8] = self._gyro_var

        imu.linear_acceleration_covariance[0] = self._accel_var
        imu.linear_acceleration_covariance[4] = self._accel_var
        imu.linear_acceleration_covariance[8] = self._accel_var

        # Drop frames containing NaNs (rare, but happens during FMU reboots).
        if not (math.isfinite(imu.angular_velocity.x)
                and math.isfinite(imu.linear_acceleration.x)):
            return

        self._pub.publish(imu)
        self._msg_count += 1

    def _log_rate(self) -> None:
        rate = self._msg_count / 5.0
        self.get_logger().info(
            f"IMU bridge rate: {rate:.1f} Hz "
            f"(expected ~200 Hz with IMU_INTEG_RATE=200)"
        )
        self._msg_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = Px4ImuBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
