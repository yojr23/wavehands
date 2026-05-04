import time
import ctypes
import platform
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from wavehands.application.loop_station import LoopStationService
from wavehands.application.selection_service import HoverSelectionService
from wavehands.config import AppConfig, default_config
from wavehands.domain.models import HandPointer, Point2D
from wavehands.domain.notes import (
    CHORD_NAMES,
    NOTE_NAMES,
    chord_intervals_from_index,
    chord_name_from_index,
    note_frequency_from_index,
    note_name_from_index,
)
from wavehands.infrastructure.audio.mono_synth import MonoSynthEngine
from wavehands.infrastructure.camera import CameraStream
from wavehands.infrastructure.hand_tracker import MediaPipeHandTracker
from wavehands.presentation.controls import ControlPanel
from wavehands.presentation.note_wheel import NoteWheel, WheelPalette
from wavehands.presentation.renderer import draw_status
from wavehands.utils.metrics import RuntimeMetrics, configure_logging


class WaveHandsApp:
    WINDOW_NAME = "WaveHands Synth MVP"

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        configure_logging()
        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(2)
        except cv2.error:
            pass
        self.config = config or default_config()
        self.camera = CameraStream(self.config.camera)
        self.tracker = MediaPipeHandTracker(self.config.tracker)
        self.note_selector = HoverSelectionService(self.config.selection)
        self.chord_selector = HoverSelectionService(self.config.selection)
        self.loop_station = LoopStationService()
        self.synth = MonoSynthEngine(self.config.audio)

        note_palette = WheelPalette(
            selected_fill=(33, 190, 236),
            hover_fill=(22, 122, 171),
            pulse=(255, 228, 120),
            center_fill=(22, 40, 55),
        )
        chord_palette = WheelPalette(
            selected_fill=(190, 93, 233),
            hover_fill=(130, 72, 165),
            pointer_ring=(194, 110, 230),
            pulse=(255, 170, 238),
            center_fill=(46, 24, 54),
        )
        self.note_wheel = NoteWheel(Point2D(0, 0), 100, NOTE_NAMES, palette=note_palette)
        self.chord_wheel = NoteWheel(Point2D(0, 0), 80, CHORD_NAMES, palette=chord_palette)

        self.base_frame_w = self.config.camera.width
        self.base_frame_h = self.config.camera.height
        self.window_w = self.base_frame_w + self.config.ui.panel_width
        self.window_h = self.base_frame_h
        self.screen_w, self.screen_h = self._detect_screen_size()

        self.controls = ControlPanel(panel_x=self.base_frame_w, panel_width=self.config.ui.panel_width)

        self._last_time = time.time()
        self._last_trigger_signature: Optional[Tuple[int, int, int, bool]] = None
        self._previous_sustain = False
        self._note_pulse_started_at = 0.0
        self._chord_pulse_started_at = 0.0
        self._canvas = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)
        self._last_layout_signature: Optional[tuple[int, int, int]] = None
        self._window_query_period_frames = 5
        self._frame_counter = 0
        self._last_volume_revision = self.controls.volume_revision
        self._record_state = "idle"
        self._record_writer: Optional[cv2.VideoWriter] = None
        self._record_temp_path: Optional[Path] = None
        self._record_fps = 30.0
        self.metrics = RuntimeMetrics(interval_seconds=self.config.metrics.log_interval_seconds)

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        try:
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        except cv2.error:
            pass
        cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(self.WINDOW_NAME, self.controls.on_mouse)
        self._is_fullscreen = True
        self.synth.set_volume(self.controls.volume_slider.value)
        self.controls.set_record_state(self._record_state)

    def run(self) -> None:
        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    break

                frame = cv2.flip(frame, 1)
                self._frame_counter += 1
                if self._frame_counter % self._window_query_period_frames == 0:
                    self.window_w, self.window_h = self._current_window_size()
                panel_w = self._panel_width(self.window_w)
                camera_w = max(220, self.window_w - panel_w)
                camera_h = max(140, self.window_h)
                if frame.shape[1] != camera_w or frame.shape[0] != camera_h:
                    frame = cv2.resize(frame, (camera_w, camera_h), interpolation=cv2.INTER_LINEAR)
                tracker_result = self.tracker.detect(frame)
                self.tracker.draw(frame, tracker_result)
                self.metrics.tick_frame(has_hands=bool(tracker_result.pointers))

                layout_signature = (camera_w, camera_h, panel_w)
                if self._last_layout_signature != layout_signature:
                    self._layout_ui(camera_w=camera_w, camera_h=camera_h, panel_w=panel_w)
                    self._last_layout_signature = layout_signature

                note_pointer = self._select_pointer_for_wheel(tracker_result.pointers, self.note_wheel)
                chord_pointer = self._select_pointer_for_wheel(tracker_result.pointers, self.chord_wheel)
                candidate_note_index = self.note_wheel.point_to_note_index(note_pointer.point) if note_pointer is not None else None
                candidate_chord_index = self.chord_wheel.point_to_note_index(chord_pointer.point) if chord_pointer is not None else None

                now = time.time()
                settings = self.controls.to_settings()
                volume_revision = self.controls.volume_revision
                if volume_revision != self._last_volume_revision:
                    self._last_volume_revision = volume_revision
                    self.synth.set_volume(settings.volume)
                record_toggle_requested = self.controls.consume_record_toggle()
                record_stop_requested = self.controls.consume_record_stop()

                if self._previous_sustain and not settings.sustain:
                    self.synth.stop_note()
                self._previous_sustain = settings.sustain

                note_selection_result = self.note_selector.update(candidate_note_index, now)
                chord_selection_result = self.chord_selector.update(candidate_chord_index, now)
                selected_chord_index = self.chord_selector.state.selected_index
                chord_intervals = chord_intervals_from_index(selected_chord_index) if selected_chord_index is not None else (0,)

                if note_selection_result.just_selected and note_selection_result.selected_index is not None:
                    frequency_hz = note_frequency_from_index(note_selection_result.selected_index, settings.octave_shift)
                    self.synth.trigger_note(
                        frequency_hz=frequency_hz,
                        duration_seconds=settings.note_duration_seconds,
                        sustain=settings.sustain,
                        chord_intervals=chord_intervals,
                    )
                    self.loop_station.record_note_event(
                        frequency_hz=frequency_hz,
                        duration_seconds=settings.note_duration_seconds,
                        velocity=settings.volume,
                        now=now,
                    )
                    self._note_pulse_started_at = now
                    self._last_trigger_signature = (
                        note_selection_result.selected_index,
                        settings.octave_shift,
                        selected_chord_index if selected_chord_index is not None else -1,
                        settings.sustain,
                    )
                    self.metrics.counters.note_changes += 1

                selected_note_index = self.note_selector.state.selected_index
                if (
                    chord_selection_result.just_selected
                    and selected_note_index is not None
                    and selected_chord_index is not None
                ):
                    frequency_hz = note_frequency_from_index(selected_note_index, settings.octave_shift)
                    self.synth.trigger_note(
                        frequency_hz=frequency_hz,
                        duration_seconds=settings.note_duration_seconds,
                        sustain=settings.sustain,
                        chord_intervals=chord_intervals_from_index(selected_chord_index),
                    )
                    self._last_trigger_signature = (
                        selected_note_index,
                        settings.octave_shift,
                        selected_chord_index,
                        settings.sustain,
                    )
                    self.metrics.counters.chord_changes += 1
                    self._chord_pulse_started_at = now

                # Si sustain esta activo y cambian parametros de rango, se actualiza sin esperar nueva seleccion.
                if settings.sustain and selected_note_index is not None:
                    signature = (
                        selected_note_index,
                        settings.octave_shift,
                        selected_chord_index if selected_chord_index is not None else -1,
                        settings.sustain,
                    )
                    if signature != self._last_trigger_signature:
                        frequency_hz = note_frequency_from_index(selected_note_index, settings.octave_shift)
                        self.synth.trigger_note(
                            frequency_hz=frequency_hz,
                            duration_seconds=settings.note_duration_seconds,
                            sustain=True,
                            chord_intervals=chord_intervals,
                        )
                        self._last_trigger_signature = signature

                if not settings.sustain and selected_note_index is None:
                    self.synth.stop_note()

                for event in self.loop_station.poll_due_events(now):
                    self.synth.trigger_loop_note(
                        frequency_hz=event.frequency_hz,
                        duration_seconds=event.duration_seconds,
                        velocity=event.velocity,
                    )

                fps = 1.0 / max(now - self._last_time, 1e-6)
                self._last_time = now

                note_hover_progress = self._hover_progress(self.note_selector, now)
                chord_hover_progress = self._hover_progress(self.chord_selector, now)
                canvas = self._ensure_canvas()
                canvas.fill(0)
                canvas[:, :camera_w] = frame

                pointer_inside_note_wheel = candidate_note_index is not None
                self.note_wheel.draw(
                    canvas,
                    hovered_index=self.note_selector.state.hovered_index,
                    selected_index=selected_note_index,
                    hover_progress=note_hover_progress,
                    pulse_strength=self._note_pulse_strength(now),
                    pointer_inside=pointer_inside_note_wheel,
                )
                pointer_inside_chord_wheel = candidate_chord_index is not None
                self.chord_wheel.draw(
                    canvas,
                    hovered_index=self.chord_selector.state.hovered_index,
                    selected_index=self.chord_selector.state.selected_index,
                    hover_progress=chord_hover_progress,
                    pulse_strength=self._chord_pulse_strength(now),
                    pointer_inside=pointer_inside_chord_wheel,
                )
                panel_x = camera_w
                self.controls.draw(canvas, panel_x)
                if record_toggle_requested:
                    self._toggle_recording(self.window_w, self.window_h)
                if record_stop_requested:
                    self._stop_recording(prompt_name=True)
                if self._record_state == "recording":
                    self._write_record_frame(canvas)

                selected_note = note_name_from_index(selected_note_index) if selected_note_index is not None else None
                selected_chord = chord_name_from_index(selected_chord_index) if selected_chord_index is not None else None
                selected_freq = (
                    note_frequency_from_index(selected_note_index, settings.octave_shift)
                    if selected_note_index is not None
                    else None
                )
                interaction_mode = "2 manos" if len(tracker_result.pointers) >= 2 else "1 mano"
                interaction_mode = f"{interaction_mode} | REC:{self._record_state}"

                draw_status(
                    canvas,
                    fps=fps,
                    selected_note=selected_note,
                    selected_chord=selected_chord,
                    frequency_hz=selected_freq,
                    hands_detected=len(tracker_result.pointers),
                    interaction_mode=interaction_mode,
                    sustain_enabled=settings.sustain,
                    loop_mode=self.loop_station.state.mode,
                    loop_layers=len(self.loop_station.state.layers),
                )

                if self.config.metrics.enabled:
                    self.metrics.maybe_log(
                        selected_note=selected_note,
                        selected_chord=selected_chord,
                        loop_mode=self.loop_station.state.mode,
                        audio_metrics=self.synth.metrics_snapshot(),
                    )

                cv2.imshow(self.WINDOW_NAME, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("f"):
                    self._toggle_fullscreen()
        finally:
            self._stop_recording(prompt_name=False)
            self.synth.close()
            self.tracker.close()
            self.camera.release()
            cv2.destroyAllWindows()

    def _select_pointer_for_wheel(self, pointers: list[HandPointer], wheel: NoteWheel) -> Optional[HandPointer]:
        active_pointer: Optional[HandPointer] = None
        active_distance_sq: Optional[int] = None

        for pointer in pointers:
            candidate = wheel.point_to_note_index(pointer.point)
            if candidate is None:
                continue
            distance_sq = (pointer.point.x - wheel.center.x) ** 2 + (pointer.point.y - wheel.center.y) ** 2
            if active_distance_sq is None or distance_sq < active_distance_sq:
                active_distance_sq = distance_sq
                active_pointer = pointer

        return active_pointer

    def _hover_progress(self, selector: HoverSelectionService, now: float) -> float:
        if selector.state.hovered_index is None:
            return 0.0
        hover_elapsed = now - selector.state.hover_started_at
        return max(0.0, min(1.0, hover_elapsed / self.config.selection.hover_seconds))

    def _note_pulse_strength(self, now: float) -> float:
        if self._note_pulse_started_at == 0.0:
            return 0.0
        age = now - self._note_pulse_started_at
        if age > 0.4:
            return 0.0
        return 1.0 - (age / 0.4)

    def _chord_pulse_strength(self, now: float) -> float:
        if self._chord_pulse_started_at == 0.0:
            return 0.0
        age = now - self._chord_pulse_started_at
        if age > 0.4:
            return 0.0
        return 1.0 - (age / 0.4)

    def _current_window_size(self) -> tuple[int, int]:
        if self._is_fullscreen:
            return self.screen_w, self.screen_h
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(self.WINDOW_NAME)
            if win_w > 0 and win_h > 0:
                return win_w, win_h
        except cv2.error:
            pass
        return self.window_w, self.window_h

    def _panel_width(self, window_width: int) -> int:
        preferred = max(220, min(420, int(window_width * 0.32)))
        max_allowed = max(180, window_width - 220)
        return min(preferred, max_allowed)

    def _layout_ui(self, camera_w: int, camera_h: int, panel_w: int) -> None:
        note_radius = max(52, int(min(camera_w * 0.20, camera_h * 0.30)))
        note_center = Point2D(x=int(camera_w * 0.30), y=int(camera_h * 0.56))
        self.note_wheel.set_geometry(note_center, note_radius)

        chord_radius = max(44, int(min(camera_w * 0.17, camera_h * 0.24)))
        chord_center = Point2D(x=int(camera_w * 0.72), y=int(camera_h * 0.36))

        # Evita que ambos circulos se monten en resoluciones complicadas.
        center_dist = ((note_center.x - chord_center.x) ** 2 + (note_center.y - chord_center.y) ** 2) ** 0.5
        min_gap = 26.0
        max_combined_radius = max(60.0, center_dist - min_gap)
        if note_radius + chord_radius > max_combined_radius:
            scale = max_combined_radius / float(note_radius + chord_radius)
            note_radius = int(note_radius * scale)
            chord_radius = int(chord_radius * scale)
            self.note_wheel.set_geometry(note_center, note_radius)

        self.chord_wheel.set_geometry(chord_center, chord_radius)

        panel_x = camera_w
        self.controls.layout(panel_x=panel_x, panel_width=panel_w, canvas_height=camera_h)

    def _ensure_canvas(self) -> np.ndarray:
        if self._canvas.shape[0] != self.window_h or self._canvas.shape[1] != self.window_w:
            self._canvas = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)
        return self._canvas

    def _toggle_fullscreen(self) -> None:
        if self._is_fullscreen:
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            try:
                cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
            except cv2.error:
                pass
            cv2.resizeWindow(self.WINDOW_NAME, 1400, 900)
            self._is_fullscreen = False
            self._last_layout_signature = None
            return
        try:
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        except cv2.error:
            pass
        cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._is_fullscreen = True
        self._last_layout_signature = None

    def _detect_screen_size(self) -> tuple[int, int]:
        if platform.system() == "Darwin":
            try:
                core_graphics = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
                core_graphics.CGMainDisplayID.restype = ctypes.c_uint32
                core_graphics.CGDisplayPixelsWide.argtypes = [ctypes.c_uint32]
                core_graphics.CGDisplayPixelsWide.restype = ctypes.c_size_t
                core_graphics.CGDisplayPixelsHigh.argtypes = [ctypes.c_uint32]
                core_graphics.CGDisplayPixelsHigh.restype = ctypes.c_size_t

                display_id = core_graphics.CGMainDisplayID()
                width = int(core_graphics.CGDisplayPixelsWide(display_id))
                height = int(core_graphics.CGDisplayPixelsHigh(display_id))
                if width > 0 and height > 0:
                    return width, height
            except Exception:
                pass
        return self.window_w, self.window_h

    def _toggle_recording(self, width: int, height: int) -> None:
        if self._record_state == "idle":
            self._start_recording(width, height)
            return
        if self._record_state == "recording":
            self._record_state = "paused"
            self.controls.set_record_state(self._record_state)
            return
        if self._record_state == "paused":
            self._record_state = "recording"
            self.controls.set_record_state(self._record_state)

    def _start_recording(self, width: int, height: int) -> None:
        downloads = Path.home() / "Downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        temp_path = downloads / f"wavehands_tmp_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_path), fourcc, self._record_fps, (int(width), int(height)))
        if not writer.isOpened():
            return
        self._record_writer = writer
        self._record_temp_path = temp_path
        self._record_state = "recording"
        self.controls.set_record_state(self._record_state)

    def _stop_recording(self, prompt_name: bool) -> None:
        if self._record_writer is None:
            self._record_state = "idle"
            self.controls.set_record_state(self._record_state)
            return
        self._record_writer.release()
        self._record_writer = None

        temp_path = self._record_temp_path
        self._record_temp_path = None
        self._record_state = "idle"
        self.controls.set_record_state(self._record_state)
        if temp_path is None or not temp_path.exists():
            return

        name = ""
        if prompt_name:
            try:
                name = input("\nNombre para guardar el video (sin .mp4): ").strip()
            except EOFError:
                name = ""
        if not name:
            name = f"wavehands_take_{time.strftime('%Y%m%d_%H%M%S')}"

        safe_name = "".join(ch for ch in name if ch.isalnum() or ch in ("_", "-", " ")).strip().replace(" ", "_")
        if not safe_name:
            safe_name = f"wavehands_take_{int(time.time())}"

        final_path = temp_path.with_name(f"{safe_name}.mp4")
        idx = 1
        while final_path.exists():
            final_path = temp_path.with_name(f"{safe_name}_{idx}.mp4")
            idx += 1

        try:
            temp_path.replace(final_path)
            print(f"[WaveHands] Video guardado en: {final_path}")
        except OSError:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _write_record_frame(self, frame: np.ndarray) -> None:
        if self._record_writer is None:
            return
        self._record_writer.write(frame)
