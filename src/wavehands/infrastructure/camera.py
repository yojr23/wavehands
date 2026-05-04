from typing import Optional

import cv2
import numpy as np

from wavehands.config import CameraConfig


class CameraStream:
    def __init__(self, config: CameraConfig) -> None:
        self._cap = cv2.VideoCapture(config.device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        if not self._cap.isOpened():
            raise RuntimeError("No se pudo abrir la camara.")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        self._cap.release()
