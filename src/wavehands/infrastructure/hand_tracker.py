from dataclasses import dataclass
from typing import List
import math

import cv2
import mediapipe as mp
import numpy as np

from wavehands.config import HandTrackerConfig
from wavehands.domain.models import HandPointer, Point2D


@dataclass
class HandTrackerResult:
    pointers: List[HandPointer]
    raw_landmarks: List[object]


class MediaPipeHandTracker:
    def __init__(self, config: HandTrackerConfig) -> None:
        self._config = config
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_num_hands,
            model_complexity=config.model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> HandTrackerResult:
        frame_for_inference = frame_bgr
        if self._config.inference_width > 0 and frame_bgr.shape[1] > self._config.inference_width:
            scale = self._config.inference_width / float(frame_bgr.shape[1])
            inf_h = max(1, int(frame_bgr.shape[0] * scale))
            frame_for_inference = cv2.resize(frame_bgr, (self._config.inference_width, inf_h), interpolation=cv2.INTER_LINEAR)

        frame_rgb = cv2.cvtColor(frame_for_inference, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)
        pointers: List[HandPointer] = []
        landmarks: List[object] = []

        if not result.multi_hand_landmarks:
            return HandTrackerResult(pointers=pointers, raw_landmarks=landmarks)

        inf_h, inf_w, _ = frame_for_inference.shape
        out_h, out_w, _ = frame_bgr.shape
        scale_x = out_w / float(inf_w)
        scale_y = out_h / float(inf_h)
        for idx, hand_lm in enumerate(result.multi_hand_landmarks):
            pinch_point = self._pinch_point_if_active(hand_lm, inf_w=inf_w, inf_h=inf_h)
            if pinch_point is None:
                continue

            point = Point2D(
                x=int(pinch_point.x * scale_x),
                y=int(pinch_point.y * scale_y),
            )
            hand_label = "Unknown"
            if result.multi_handedness and idx < len(result.multi_handedness):
                hand_label = result.multi_handedness[idx].classification[0].label
            pointers.append(HandPointer(point=point, label=f"{hand_label} pinch"))
            landmarks.append(hand_lm)

        return HandTrackerResult(pointers=pointers, raw_landmarks=landmarks)

    def draw(self, frame_bgr: np.ndarray, tracker_result: HandTrackerResult) -> None:
        for idx, hand_lm in enumerate(tracker_result.raw_landmarks):
            if self._config.draw_landmarks:
                self._mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_lm,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )
            if idx < len(tracker_result.pointers):
                pointer = tracker_result.pointers[idx]
                cv2.circle(frame_bgr, (pointer.point.x, pointer.point.y), 7, (0, 255, 255), -1)
                cv2.putText(
                    frame_bgr,
                    pointer.label,
                    (pointer.point.x + 9, pointer.point.y - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                )

    def close(self) -> None:
        self._hands.close()

    def _pinch_point_if_active(self, hand_lm: object, inf_w: int, inf_h: int) -> Point2D | None:
        thumb_tip = hand_lm.landmark[self._mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_lm.landmark[self._mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = hand_lm.landmark[self._mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = hand_lm.landmark[self._mp_hands.HandLandmark.PINKY_MCP]

        thumb_x = float(thumb_tip.x) * inf_w
        thumb_y = float(thumb_tip.y) * inf_h
        index_x = float(index_tip.x) * inf_w
        index_y = float(index_tip.y) * inf_h
        index_mcp_x = float(index_mcp.x) * inf_w
        index_mcp_y = float(index_mcp.y) * inf_h
        pinky_mcp_x = float(pinky_mcp.x) * inf_w
        pinky_mcp_y = float(pinky_mcp.y) * inf_h

        pinch_distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
        palm_width = math.hypot(index_mcp_x - pinky_mcp_x, index_mcp_y - pinky_mcp_y)
        threshold = max(6.0, palm_width * self._config.pinch_threshold_ratio)

        if pinch_distance > threshold:
            return None

        return Point2D(x=int((thumb_x + index_x) * 0.5), y=int((thumb_y + index_y) * 0.5))
