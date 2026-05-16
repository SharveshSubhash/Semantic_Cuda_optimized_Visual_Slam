# SPDX-License-Identifier: Apache-2.0
#
# Launches the full cuVSLAM stack tuned for an Intel RealSense D415 on a Jetson,
# with the option of fusing IMU from a Pixhawk via the uXRCE-DDS bridge.
#
# Pipeline:
#   realsense2_camera (infra1 + infra2, emitter OFF)
#     -> isaac_ros_visual_slam (stereo, IMU fusion optional)
#         -> /visual_slam/tracking/vo_pose_covariance  (consumed downstream)
#
# Why these choices (verified against NVIDIA docs and Intel D400 datasheet):
#   * The D415 has NO onboard IMU (only the D435i and D455 do). The example
#     launch file in bandofpv/VSLAM-UAV was written for a D435i, so all of its
#     gyro/accel/unite_imu_method/IMU-fusion parameters must be removed or
#     replaced. We do both: by default we run pure stereo cuVSLAM, and if
#     `use_px4_imu` is true we bridge /fmu/out/sensor_combined into
#     sensor_msgs/Imu and feed THAT into cuVSLAM.
#   * The IR projector ("emitter") must be off. With it on, cuVSLAM tracks the
#     projected dots as features and drifts wildly. NVIDIA's troubleshooting
#     page calls this out explicitly. The emitter is needed for the depth
#     stream, but cuVSLAM does not consume depth - only raw rectified IR.
#   * tracking_mode = 0 forces stereo-only tracking. The launch file shipped
#     by NVIDIA defaults to a mode that expects an IMU; without IMU it must
#     be set to 0 explicitly.
#   * We expose `enable_image_denoising = False` because the IR streams from
#     the D415 are already low-noise enough at 60 Hz, and denoising adds
#     latency that hurts the jitter threshold.
#
# Launch arguments:
#   use_px4_imu  (default: false)  - Bridge /fmu/out/sensor_combined into
#                                    sensor_msgs/Imu and enable IMU fusion in
#                                    cuVSLAM. Requires the uXRCE-DDS agent
#                                    running and the static transform from
#                                    camera_link to imu_link to be published.
#   start_combiner (default: true) - Start the semantic graph combiner node.
#   start_redis_writer (default: true) - Start the Redis writer node.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    use_px4_imu_arg = DeclareLaunchArgument(
        'use_px4_imu', default_value='false',
        description='Bridge Pixhawk IMU (uXRCE-DDS sensor_combined) into cuVSLAM.'
    )
    start_combiner_arg = DeclareLaunchArgument(
        'start_combiner', default_value='true',
        description='Start the semantic_graph_combiner node.'
    )
    start_redis_writer_arg = DeclareLaunchArgument(
        'start_redis_writer', default_value='true',
        description='Start the Redis writer node.'
    )
    start_nanoowl_arg = DeclareLaunchArgument(
        'start_nanoowl', default_value='true',
        description='Start the NanoOWL inference node.'
    )
    enable_color_arg = DeclareLaunchArgument(
        'enable_color', default_value='false',
        description=(
            'Enable the D415 RGB stream. NanoOWL will use color if true, '
            'otherwise it uses the left IR stream (with the emitter off). '
            'cuVSLAM ignores color either way.'
        )
    )
    nanoowl_engine_arg = DeclareLaunchArgument(
        'nanoowl_engine',
        default_value='/opt/nanoowl/data/owl_image_encoder_patch32.engine',
        description='Path to the NanoOWL TensorRT image-encoder engine.'
    )
    nanoowl_prompt_arg = DeclareLaunchArgument(
        'nanoowl_prompt',
        default_value=(
            'a person, a chair, a table, a fire extinguisher, a door, '
            'a window, a laptop, a backpack, a cup, a bottle'
        ),
        description='Comma-separated initial open-vocabulary prompt for NanoOWL.'
    )

    use_px4_imu = LaunchConfiguration('use_px4_imu')
    start_combiner = LaunchConfiguration('start_combiner')
    start_redis_writer = LaunchConfiguration('start_redis_writer')
    start_nanoowl = LaunchConfiguration('start_nanoowl')
    enable_color = LaunchConfiguration('enable_color')
    nanoowl_engine = LaunchConfiguration('nanoowl_engine')
    nanoowl_prompt = LaunchConfiguration('nanoowl_prompt')

    # -----------------------------------------------------------------
    # RealSense D415 camera node
    # -----------------------------------------------------------------
    # Only infra1 + infra2 are streamed. Color, depth and pointcloud are
    # disabled because:
    #   - cuVSLAM does not consume them
    #   - they steal USB3 bandwidth and reduce IR FPS
    #   - depth_module.emitter_enabled MUST be 0 for cuVSLAM (see above)
    #
    # The D415 supports 848x480 @ 90 fps on the IR streams over USB3, but
    # cuVSLAM is tuned for 60 fps with image_jitter_threshold_ms = 33.33,
    # so we use the (640x480, 60 Hz) profile which is the sweet spot for
    # Jetson Orin throughput.
    realsense_camera_node = Node(
        name='camera',
        namespace='camera',
        package='realsense2_camera',
        executable='realsense2_camera_node',
        parameters=[{
            'enable_infra1': True,
            'enable_infra2': True,
            # We use a PythonExpression so a single launch invocation can
            # flip color on or off via the `enable_color` arg.
            'enable_color': PythonExpression(["'", enable_color, "' == 'true'"]),
            'enable_depth': False,
            'depth_module.emitter_enabled': 0,
            'depth_module.profile': '640x480x60',
            'rgb_camera.profile': '640x480x30',
            'enable_gyro': False,
            'enable_accel': False,
        }],
        output='screen',
    )

    # -----------------------------------------------------------------
    # PX4 IMU bridge (conditional)
    # -----------------------------------------------------------------
    # Subscribes to /fmu/out/sensor_combined (px4_msgs.SensorCombined) and
    # republishes as sensor_msgs/Imu on /imu/data_raw, with frame_id set to
    # `imu_link`. The uXRCE-DDS agent must be running and the FCU configured
    # for UXRCE_DDS_CFG. Frame conversion FRD -> FLU is handled inside the
    # bridge node.
    px4_imu_bridge_node = Node(
        package='vslam_semantic',
        executable='px4_imu_bridge_node',
        name='px4_imu_bridge',
        output='screen',
        condition=IfCondition(use_px4_imu),
        parameters=[{
            'input_topic': '/fmu/out/sensor_combined',
            'output_topic': '/imu/data_raw',
            'imu_frame_id': 'imu_link',
            # Noise model values from the Pixhawk 4 (ICM-20689) datasheet.
            # These are exposed so they can be overridden by allan_ros2
            # results if you've calibrated your own unit.
            'gyro_noise_density': 0.00018,
            'gyro_random_walk': 1.0e-5,
            'accel_noise_density': 0.002,
            'accel_random_walk': 3.0e-3,
        }],
    )

    # -----------------------------------------------------------------
    # cuVSLAM (isaac_ros_visual_slam)
    # -----------------------------------------------------------------
    # tracking_mode:
    #   0 = stereo only        <- our default for D415
    #   1 = stereo + IMU       <- used when use_px4_imu == true
    #
    # We define two ComposableNodes with the only difference being the IMU
    # parameters / remappings, and gate them on `use_px4_imu`.
    common_vslam_params = {
        'enable_image_denoising': False,
        'rectified_images': True,
        'base_frame': 'camera_link',
        'map_frame': 'map',
        'odom_frame': 'odom',
        'enable_slam_visualization': True,
        'enable_landmarks_view': True,
        'enable_observations_view': True,
        'publish_odom_to_base_tf': True,
        'publish_map_to_odom_tf': True,
        'image_jitter_threshold_ms': 33.34,  # ~1 frame at 60 Hz
        'camera_optical_frames': [
            'camera_infra1_optical_frame',
            'camera_infra2_optical_frame',
        ],
    }

    vslam_stereo_only = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            **common_vslam_params,
            'enable_imu_fusion': False,
        }],
        remappings=[
            ('visual_slam/image_0', '/camera/infra1/image_rect_raw'),
            ('visual_slam/camera_info_0', '/camera/infra1/camera_info'),
            ('visual_slam/image_1', '/camera/infra2/image_rect_raw'),
            ('visual_slam/camera_info_1', '/camera/infra2/camera_info'),
        ],
    )

    vslam_with_imu = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            **common_vslam_params,
            'enable_imu_fusion': True,
            'imu_frame': 'imu_link',
            # Pixhawk IMU specs (override these with allan_ros2 results
            # if you've calibrated your own unit).
            'gyro_noise_density': 0.00018,
            'gyro_random_walk': 1.0e-5,
            'accel_noise_density': 0.002,
            'accel_random_walk': 3.0e-3,
            'calibration_frequency': 200.0,
            'imu_jitter_threshold_ms': 6.0,   # 200 Hz IMU -> 5 ms period
        }],
        remappings=[
            ('visual_slam/image_0', '/camera/infra1/image_rect_raw'),
            ('visual_slam/camera_info_0', '/camera/infra1/camera_info'),
            ('visual_slam/image_1', '/camera/infra2/image_rect_raw'),
            ('visual_slam/camera_info_1', '/camera/infra2/camera_info'),
            ('visual_slam/imu', '/imu/data_raw'),
        ],
    )

    # We use two containers gated on opposite conditions; exactly one of
    # them loads its ComposableNode at launch time.
    vslam_container_stereo = ComposableNodeContainer(
        name='visual_slam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[vslam_stereo_only],
        output='screen',
        condition=UnlessCondition(use_px4_imu),
    )
    vslam_container_imu = ComposableNodeContainer(
        name='visual_slam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[vslam_with_imu],
        output='screen',
        condition=IfCondition(use_px4_imu),
    )

    # -----------------------------------------------------------------
    # NanoOWL inference
    # -----------------------------------------------------------------
    # The image_topic is conditional: if `enable_color` is true we feed
    # the RGB stream; otherwise we feed the left IR stream. Either way
    # the combiner's back-projection uses /camera/infra1/camera_info, so
    # if you switch to color you should also remap the combiner's
    # camera_info_topic param to /camera/color/camera_info.
    nanoowl_image_topic = PythonExpression([
        "'/camera/color/image_raw' if '", enable_color,
        "' == 'true' else '/camera/infra1/image_rect_raw'"
    ])

    nanoowl_inference_node = Node(
        package='vslam_semantic',
        executable='nanoowl_inference_node',
        name='nanoowl_inference',
        output='screen',
        condition=IfCondition(start_nanoowl),
        parameters=[{
            'image_topic': nanoowl_image_topic,
            'query_topic': '/nanoowl/input_query',
            'detections_topic': '/nanoowl/detections',
            'vis_image_topic': '/nanoowl/output_image',
            'publish_vis_image': True,
            'image_encoder_engine': nanoowl_engine,
            'initial_prompt': nanoowl_prompt,
            'confidence_threshold': 0.10,
            'nms_iou_threshold': 0.30,
            'target_height': 480,
            'inference_period_sec': 0.10,  # 10 Hz
        }],
    )

    # -----------------------------------------------------------------
    # Semantic graph combiner + Redis writer
    # -----------------------------------------------------------------
    combiner_camera_info_topic = PythonExpression([
        "'/camera/color/camera_info' if '", enable_color,
        "' == 'true' else '/camera/infra1/camera_info'"
    ])

    semantic_combiner_node = Node(
        package='vslam_semantic',
        executable='semantic_graph_combiner_node',
        name='semantic_graph_combiner',
        output='screen',
        condition=IfCondition(start_combiner),
        parameters=[{
            'pose_topic': '/visual_slam/tracking/vo_pose_covariance',
            'odom_topic': '/visual_slam/tracking/odometry',
            'detections_topic': '/nanoowl/detections',
            'camera_info_topic': combiner_camera_info_topic,
            'graph_topic': '/semantic_graph',
            'world_frame': 'odom',
            'sync_slop_sec': 0.10,
            'spatial_merge_radius_m': 0.50,
            'confidence_threshold': 0.30,
            'max_graph_nodes': 2000,
        }],
    )

    redis_writer_node = Node(
        package='vslam_semantic',
        executable='redis_writer_node',
        name='redis_writer',
        output='screen',
        condition=IfCondition(start_redis_writer),
        parameters=[{
            'graph_topic': '/semantic_graph',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0,
            'redis_key_prefix': 'semantic_graph',
            'write_full_snapshot': True,
            'snapshot_ttl_sec': 0,  # 0 = no expiry
        }],
    )

    return LaunchDescription([
        use_px4_imu_arg,
        start_combiner_arg,
        start_redis_writer_arg,
        start_nanoowl_arg,
        enable_color_arg,
        nanoowl_engine_arg,
        nanoowl_prompt_arg,
        realsense_camera_node,
        px4_imu_bridge_node,
        vslam_container_stereo,
        vslam_container_imu,
        nanoowl_inference_node,
        semantic_combiner_node,
        redis_writer_node,
    ])
