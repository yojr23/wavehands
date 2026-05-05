from typing import Optional

import cv2
import numpy as np


def draw_status(
    frame: np.ndarray,
    fps: float,
    selected_note: Optional[str],
    selected_chord: Optional[str],
    frequency_hz: Optional[float],
    hands_detected: int,
    interaction_mode: str,
    selected_scale: str,
    selected_instrument: str,
    sustain_enabled: bool,
    loop_mode: str,
    loop_layers: int,
) -> None:
    cv2.putText(frame, f"FPS: {fps:5.1f}", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 240, 100), 2)
    cv2.putText(frame, f"Manos: {hands_detected}", (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.putText(frame, f"Modo: {interaction_mode}", (14, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.putText(frame, f"Sustain: {'ON' if sustain_enabled else 'OFF'}", (14, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.putText(
        frame,
        f"Loop: {loop_mode.upper()}  Capas:{loop_layers}",
        (14, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
    )
    cv2.putText(frame, f"Escala: {selected_scale}", (14, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.putText(
        frame,
        f"Instrumento: {selected_instrument}",
        (14, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
    )

    note_label = selected_note if selected_note is not None else "--"
    cv2.putText(frame, f"Nota: {note_label}", (14, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2)
    chord_label = selected_chord if selected_chord is not None else "--"
    cv2.putText(frame, f"Acorde: {chord_label}", (14, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 190, 110), 2)

    freq_label = f"{frequency_hz:6.2f} Hz" if frequency_hz is not None else "--"
    cv2.putText(frame, f"Frecuencia: {freq_label}", (14, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1)
