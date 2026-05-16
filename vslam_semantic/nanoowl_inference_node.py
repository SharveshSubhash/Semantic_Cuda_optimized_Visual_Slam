#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
nanoowl_inference_node.py
==========================

Runs NanoOWL (OWL-ViT optimised with TensorRT) inside a ROS 2 node and
publishes detections in the form the semantic_graph_combiner expects.

Why this exists alongside NVIDIA's ROS2-NanoOWL
-----------------------------------------------
NVIDIA's ROS2-NanoOWL is a fine reference, but it has three properties
that are awkward for our pipeline:

  1. The `class_id` field of every emitted Detection2D is the *index*
     of the prompt in the prompt list, not the prompt text. The
     semantic graph combiner merges landmarks by label string, so we'd
     have to maintain a separate prompt-index -> label dictionary on
     the subscriber side that drifts whenever the prompt is updated.
  2. The input topic must be RGB sensor_msgs/Image. With a D415 in
     our pipeline we have IR (mono8) and (optionally) color streams.
     This node handles either: it converts IR to a 3-channel image
     in-place, so we can run OWL detection against the same image
     stream that cuVSLAM consumes (perfect time-alignment, no
     additional bandwidth).
  3. NVIDIA's node loads the prompt at startup. We support a
     dynamically-updatable prompt topic (`/nanoowl/input_query`)
     **and** an initial prompt parameter, so the planner can change
     the open vocabulary on the fly (e.g. "describe what's ahead"
     vs "is there a fire extinguisher in the room").

Topic interface (matches what the combiner expects)
---------------------------------------------------
Subscribes:
    /camera/infra1/image_rect_raw   sensor_msgs/Image  (mono8 from D415,
                                                        or rgb8/bgr8)
    /nanoowl/input_query            std_msgs/String    (optional, dynamic
                                                        prompt update; format
                                                        is a comma-separated
                                                        list, e.g. "a person,
                                                        a fire extinguisher,
                                                        a chair")

Publishes:
    /nanoowl/detections             vision_msgs/Detection2DArray
        - one Detection2D per box
        - bbox.center.position.{x,y} in image pixel coords
        - bbox.size_{x,y} in pixels
        - results[0].hypothesis.class_id = the LABEL TEXT (e.g. "a chair")
        - results[0].hypothesis.score    = confidence in [0, 1]
    /nanoowl/output_image           sensor_msgs/Image (optional, for debug)

Performance knobs
-----------------
  * `target_height` downsizes the image before inference. OWL-ViT runs
    on 768x768 internally regardless, but a smaller input means less
    CPU work in cv_bridge + colour conversion. On Jetson Orin Nano,
    240p IR -> 768x768 OWL inference holds ~10-12 fps; on AGX Orin,
    full 640x480 IR holds ~25 fps.
  * `inference_period_sec` rate-limits inference - we don't need it at
    60 Hz, and the combiner only publishes at 5 Hz. The default of
    0.1 (10 Hz) is a good balance.
"""

from __future__ import annotations

import inspect
import threading
import time
from typing import List, Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OpenCV (python3-opencv) is required. Install with "
        "'sudo apt install python3-opencv'."
    ) from exc

try:
    from vision_msgs.msg import (
        BoundingBox2D,
        Detection2D,
        Detection2DArray,
        ObjectHypothesis,
        ObjectHypothesisWithPose,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "vision_msgs is required. Install with "
        "'sudo apt install ros-humble-vision-msgs'."
    ) from exc

# NanoOWL is installed by following the steps in the README. We import
# lazily so this module can still be imported (for unit tests etc.) on
# machines that do not have a Jetson TensorRT environment.
_NANOOWL_IMPORT_ERROR: Optional[ImportError] = None
try:
    from nanoowl.owl_predictor import OwlPredictor
    from PIL import Image as PILImage
except ImportError as exc:
    _NANOOWL_IMPORT_ERROR = exc
    OwlPredictor = None
    PILImage = None


# Re-export the version-tolerant attribute reader from _helpers under its
# original private name so the rest of this module keeps working unchanged.
from ._helpers import first_attr as _first_attr  # noqa: F401


class NanoOwlInferenceNode(Node):

    def __init__(self) -> None:
        super().__init__('nanoowl_inference')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('image_topic', '/camera/infra1/image_rect_raw')
        self.declare_parameter('query_topic', '/nanoowl/input_query')
        self.declare_parameter('detections_topic', '/nanoowl/detections')
        self.declare_parameter('vis_image_topic', '/nanoowl/output_image')
        self.declare_parameter('publish_vis_image', True)
        # Default prompt is a comma-separated list of OWL-ViT phrases.
        self.declare_parameter(
            'initial_prompt',
            'a person, a chair, a table, a fire extinguisher, a door, '
            'a window, a laptop, a backpack, a cup, a bottle')
        self.declare_parameter('confidence_threshold', 0.10)
        self.declare_parameter('nms_iou_threshold', 0.30)
        self.declare_parameter(
            'image_encoder_engine',
            '/opt/nanoowl/data/owl_image_encoder_patch32.engine')
        self.declare_parameter('model_name', 'google/owlvit-base-patch32')
        self.declare_parameter('target_height', 480)
        self.declare_parameter('inference_period_sec', 0.10)  # 10 Hz
        self.declare_parameter('log_period_sec', 5.0)

        self._image_topic = self.get_parameter('image_topic').value
        self._query_topic = self.get_parameter('query_topic').value
        self._det_topic = self.get_parameter('detections_topic').value
        self._vis_topic = self.get_parameter('vis_image_topic').value
        self._publish_vis = bool(self.get_parameter('publish_vis_image').value)
        self._conf_thresh = float(self.get_parameter('confidence_threshold').value)
        self._nms_iou = float(self.get_parameter('nms_iou_threshold').value)
        self._engine_path = self.get_parameter('image_encoder_engine').value
        self._model_name = self.get_parameter('model_name').value
        self._target_h = int(self.get_parameter('target_height').value)
        self._inference_period = float(self.get_parameter('inference_period_sec').value)
        self._log_period = float(self.get_parameter('log_period_sec').value)

        # ------------------------------------------------------------------
        # Load NanoOWL
        # ------------------------------------------------------------------
        if _NANOOWL_IMPORT_ERROR is not None:
            self.get_logger().error(
                f"Cannot import NanoOWL: {_NANOOWL_IMPORT_ERROR}. "
                "Follow the install steps in the README.")
            raise SystemExit(1)

        self.get_logger().info(
            f"Loading NanoOWL: model='{self._model_name}', "
            f"engine='{self._engine_path}' (this may take ~10 s)…")
        self._predictor = OwlPredictor(
            self._model_name,
            image_encoder_engine=self._engine_path,
        )

        # Text encoding is the expensive operation when the prompt
        # changes (~500 ms on Orin Nano), so cache it. We also probe the
        # OwlPredictor.predict() signature once - the NanoOWL API has
        # drifted over the past year (some versions accept
        # `text_encodings=`, some accept `pad_square=`, some neither),
        # and we want to support all of them without per-frame try/except.
        sig = inspect.signature(self._predictor.predict)
        self._supports_text_encodings = 'text_encodings' in sig.parameters
        self._supports_pad_square = 'pad_square' in sig.parameters
        self.get_logger().info(
            f"OwlPredictor.predict() params: {list(sig.parameters)}")

        initial_prompt = str(self.get_parameter('initial_prompt').value)
        self._prompt_lock = threading.Lock()
        self._labels: List[str] = []
        self._text_encodings = None
        self._set_prompt(initial_prompt)

        # ------------------------------------------------------------------
        # State for rate-limited inference
        # ------------------------------------------------------------------
        self._bridge = CvBridge()
        self._last_image: Optional[Image] = None
        self._last_image_lock = threading.Lock()
        self._last_inference_time = 0.0
        self._inference_count = 0
        self._dropped_count = 0

        # ------------------------------------------------------------------
        # Pub/Sub
        # ------------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,  # keep last 2 - we drop older frames anyway
        )

        self._sub_image = self.create_subscription(
            Image, self._image_topic, self._on_image, sensor_qos)
        self._sub_query = self.create_subscription(
            String, self._query_topic, self._on_query, 5)

        self._pub_det = self.create_publisher(
            Detection2DArray, self._det_topic, 5)
        self._pub_vis: Optional[rclpy.publisher.Publisher] = None
        if self._publish_vis:
            self._pub_vis = self.create_publisher(Image, self._vis_topic, 2)

        # Inference is driven by a timer so we decouple from input frame
        # rate. The timer pulls the latest image (dropping older ones).
        self._inference_timer = self.create_timer(
            self._inference_period, self._run_inference_if_ready)
        self._log_timer = self.create_timer(self._log_period, self._log_stats)

        self.get_logger().info(
            f"NanoOWL inference up. image='{self._image_topic}', "
            f"detections='{self._det_topic}', "
            f"target_height={self._target_h}px, "
            f"inference rate ~{1.0/self._inference_period:.1f} Hz"
        )

    # ------------------------------------------------------------------
    # Prompt management
    # ------------------------------------------------------------------
    def _set_prompt(self, prompt_str: str) -> None:
        """Update the open-vocabulary prompt, re-encoding text features."""
        # Accept comma-separated, optionally surrounded by brackets a-la
        # the NanoOWL CLI ("[a person, a chair]").
        cleaned = prompt_str.strip().lstrip('[').rstrip(']')
        labels = [s.strip() for s in cleaned.split(',') if s.strip()]
        if not labels:
            self.get_logger().warn(f"Empty prompt received: '{prompt_str}'")
            return

        encodings = None
        # encode_text exists in current NanoOWL but didn't always - keep
        # going without cached encodings if it's missing or fails. In
        # that case we pay the encode cost on every frame, which is
        # ~5 ms - not great, but functional.
        if hasattr(self._predictor, 'encode_text') and self._supports_text_encodings:
            try:
                encodings = self._predictor.encode_text(labels)
            except Exception as e:
                self.get_logger().warn(
                    f"encode_text failed ({e}); will re-encode each frame")

        with self._prompt_lock:
            self._labels = labels
            self._text_encodings = encodings
        self.get_logger().info(
            f"Active prompt set to {len(labels)} labels: {labels}")

    def _on_query(self, msg: String) -> None:
        self._set_prompt(msg.data)

    # ------------------------------------------------------------------
    # Image ingest (drop policy: keep latest only)
    # ------------------------------------------------------------------
    def _on_image(self, msg: Image) -> None:
        with self._last_image_lock:
            if self._last_image is not None:
                self._dropped_count += 1
            self._last_image = msg

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    def _run_inference_if_ready(self) -> None:
        with self._last_image_lock:
            msg = self._last_image
            self._last_image = None
        if msg is None:
            return

        with self._prompt_lock:
            labels = list(self._labels)
            encodings = self._text_encodings
        if not labels:
            return

        # 1. Convert ROS Image -> 3-channel RGB numpy array
        try:
            rgb = self._ros_image_to_rgb(msg)
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}",
                                    throttle_duration_sec=2.0)
            return

        # 2. Resize (optional). OWL-ViT will rescale to 768x768 anyway;
        # we resize beforehand to reduce numpy/CV-bridge work. We
        # preserve aspect ratio.
        in_h, in_w = rgb.shape[:2]
        if self._target_h > 0 and in_h > self._target_h:
            scale = self._target_h / float(in_h)
            new_w = max(1, int(round(in_w * scale)))
            new_h = self._target_h
            resized = cv2.resize(rgb, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            new_h, new_w = in_h, in_w
            resized = rgb

        # 3. Run OWL inference. NanoOWL's predict() takes a PIL.Image.
        # We build kwargs based on what the installed version supports
        # (probed once at startup) so this stays compatible across
        # NanoOWL releases.
        pil = PILImage.fromarray(resized)
        predict_kwargs = dict(
            image=pil,
            text=labels,
            threshold=self._conf_thresh,
        )
        if self._supports_text_encodings and encodings is not None:
            predict_kwargs['text_encodings'] = encodings
        if self._supports_pad_square:
            predict_kwargs['pad_square'] = False

        t0 = time.time()
        try:
            output = self._predictor.predict(**predict_kwargs)
        except Exception as e:
            self.get_logger().error(f"OWL inference failed: {e}",
                                    throttle_duration_sec=2.0)
            return
        infer_ms = (time.time() - t0) * 1000.0
        self._inference_count += 1
        self._last_inference_time = infer_ms

        # 4. Build Detection2DArray. The OwlDecodeOutput dataclass varies
        # slightly between NanoOWL versions - we read by attribute with
        # a couple of fallback names. The current main has:
        #     - boxes        Tensor[N, 4] in pixel coords (x1,y1,x2,y2)
        #     - scores       Tensor[N]
        #     - labels       Tensor[N]  (indices into `text`)
        det_array = Detection2DArray()
        det_array.header = msg.header  # preserve frame_id + stamp

        boxes_t = _first_attr(output, ['boxes', 'pred_boxes'])
        scores_t = _first_attr(output, ['scores', 'pred_scores'])
        label_idx_t = _first_attr(output, ['labels', 'class_indices',
                                            'pred_labels'])
        if boxes_t is None or scores_t is None or label_idx_t is None:
            # Empty result - publish empty array so the combiner sees the heartbeat
            self._pub_det.publish(det_array)
            return

        # Move to CPU numpy once (cheap thanks to small N).
        boxes = boxes_t.detach().cpu().numpy() if hasattr(boxes_t, 'detach') \
            else np.asarray(boxes_t)
        scores = scores_t.detach().cpu().numpy() if hasattr(scores_t, 'detach') \
            else np.asarray(scores_t)
        lbl_idx = label_idx_t.detach().cpu().numpy() if hasattr(label_idx_t, 'detach') \
            else np.asarray(label_idx_t)

        # Guard: if any of the arrays came back zero-length, publish empty.
        if boxes.size == 0 or scores.size == 0:
            self._pub_det.publish(det_array)
            return

        # 5. Class-agnostic NMS to deduplicate overlapping boxes. We rely
        # on cv2.dnn.NMSBoxes which expects (x, y, w, h).
        keep_idx = list(range(len(scores)))
        if len(scores) > 1 and self._nms_iou < 1.0:
            xywh = [[float(b[0]), float(b[1]),
                     float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
            kept = cv2.dnn.NMSBoxes(xywh, scores.tolist(),
                                    self._conf_thresh, self._nms_iou)
            if isinstance(kept, np.ndarray):
                keep_idx = kept.flatten().tolist()
            elif kept is None or len(kept) == 0:
                keep_idx = []
            else:
                # OpenCV < 4.7 returns nested lists.
                keep_idx = [int(k[0]) if hasattr(k, '__iter__') else int(k)
                            for k in kept]

        # 6. Rescale boxes back to the ORIGINAL input image coords (this
        # is what the combiner uses for back-projection - it must
        # match camera_info K, which is for the original resolution).
        inv_scale = 1.0 / scale if scale != 0 else 1.0

        for i in keep_idx:
            x1, y1, x2, y2 = boxes[i]
            cx_resized = float((x1 + x2) * 0.5)
            cy_resized = float((y1 + y2) * 0.5)
            w_resized = float(max(1.0, x2 - x1))
            h_resized = float(max(1.0, y2 - y1))

            cx = cx_resized * inv_scale
            cy = cy_resized * inv_scale
            w = w_resized * inv_scale
            h = h_resized * inv_scale

            label_text = labels[int(lbl_idx[i])] \
                if 0 <= int(lbl_idx[i]) < len(labels) else "unknown"

            det = Detection2D()
            det.header = msg.header
            det.bbox = BoundingBox2D()
            # vision_msgs in Humble uses `center.position` (Pose2D-style);
            # earlier distros used `center.x`. We set both via attribute
            # discovery to stay portable across point releases.
            if hasattr(det.bbox.center, 'position'):
                det.bbox.center.position.x = cx
                det.bbox.center.position.y = cy
            else:
                det.bbox.center.x = cx
                det.bbox.center.y = cy
            det.bbox.size_x = w
            det.bbox.size_y = h

            hyp = ObjectHypothesisWithPose()
            # `hypothesis` style is used in Iron+; older distros put
            # class_id / score directly on the ObjectHypothesisWithPose.
            if hasattr(hyp, 'hypothesis'):
                hyp.hypothesis = ObjectHypothesis()
                hyp.hypothesis.class_id = label_text
                hyp.hypothesis.score = float(scores[i])
            else:
                hyp.id = label_text
                hyp.score = float(scores[i])
            det.results.append(hyp)
            det_array.detections.append(det)

        self._pub_det.publish(det_array)

        # 7. Optional debug visualization
        if self._pub_vis is not None:
            vis = self._draw_detections(resized.copy(), boxes, scores,
                                        lbl_idx, labels, keep_idx)
            vis_msg = self._bridge.cv2_to_imgmsg(vis, encoding='rgb8')
            vis_msg.header = msg.header
            self._pub_vis.publish(vis_msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ros_image_to_rgb(self, msg: Image) -> np.ndarray:
        """Convert any incoming Image encoding to an HxWx3 uint8 RGB array."""
        enc = msg.encoding.lower()
        if enc in ('rgb8',):
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        if enc in ('bgr8',):
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if enc in ('mono8', '8uc1'):
            mono = self._bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
        if enc in ('mono16', '16uc1'):
            # The D415 IR streams can be requested as 16-bit; OWL needs
            # 8-bit. Normalise to 0-255 with simple linear stretch.
            mono16 = self._bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
            mono8 = cv2.normalize(mono16, None, 0, 255,
                                  cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.cvtColor(mono8, cv2.COLOR_GRAY2RGB)
        # Fallback: let cv_bridge try its best, then coerce to RGB.
        arr = self._bridge.imgmsg_to_cv2(msg)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        raise ValueError(f"Unsupported image encoding: '{msg.encoding}'")

    @staticmethod
    def _draw_detections(img: np.ndarray, boxes: np.ndarray,
                         scores: np.ndarray, label_idx: np.ndarray,
                         labels: List[str], keep: List[int]) -> np.ndarray:
        for i in keep:
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            label_text = labels[int(label_idx[i])] \
                if 0 <= int(label_idx[i]) < len(labels) else "?"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label_text} {scores[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1),
                          (0, 255, 0), -1)
            cv2.putText(img, text, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img

    def _log_stats(self) -> None:
        rate = self._inference_count / max(self._log_period, 1e-3)
        self.get_logger().info(
            f"NanoOWL: {rate:.1f} inferences/s "
            f"(last={self._last_inference_time:.1f} ms), "
            f"{self._dropped_count} dropped frames in last {self._log_period:.0f}s"
        )
        self._inference_count = 0
        self._dropped_count = 0


def main(args=None):
    rclpy.init(args=args)
    try:
        node = NanoOwlInferenceNode()
    except SystemExit:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
