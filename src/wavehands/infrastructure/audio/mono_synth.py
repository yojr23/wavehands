import queue
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from wavehands.config import AudioConfig
from wavehands.infrastructure.audio.c_voice_mixer import CVoiceMixer


class MonoSynthEngine:
    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._sample_rate = config.sample_rate

        self._target_freq = 261.63
        self._target_amp = 0.0
        self._volume = 0.5
        self._instrument_name = "Sine"
        self._note_off_at: Optional[float] = None
        self._note_off_at_perf: Optional[float] = None
        self._is_active = False

        self._c_mixer = CVoiceMixer()
        self._root_mix = 0.58
        self._harmonic_mix = 1.05

        # Persistent state scalars as 1-element arrays to avoid per-callback ctypes objects.
        self._state_freq = np.asarray([261.63], dtype=np.float32)
        self._state_amp = np.asarray([0.0], dtype=np.float32)
        self._state_phase = np.asarray([0.0], dtype=np.float32)

        self._harmonic_ratios_np = np.zeros(0, dtype=np.float32)
        self._harmonic_phases_np = np.zeros(0, dtype=np.float32)

        self._callback_count = 0
        self._xrun_count = 0
        self._callback_cpu_avg = 0.0

        self._output_buffer = np.zeros(max(1, config.block_size), dtype=np.float32)

        self._loop_voice_count = 0
        self._loop_freqs = np.zeros(8, dtype=np.float32)
        self._loop_phases = np.zeros(8, dtype=np.float32)
        self._loop_elapsed = np.zeros(8, dtype=np.int32)
        self._loop_durations = np.zeros(8, dtype=np.int32)
        self._loop_velocities = np.zeros(8, dtype=np.float32)

        self._control_queue: queue.SimpleQueue[tuple[str, tuple]] = queue.SimpleQueue()
        self._record_capture_active = False
        self._record_capture_paused = False
        self._record_chunks: list[np.ndarray] = []
        self._record_lock = threading.Lock()

        self._freq_lerp = 0.0025
        self._amp_attack_lerp = self._seconds_to_lerp(config.attack_seconds)
        self._amp_release_lerp = self._seconds_to_lerp(config.release_seconds)
        self._attack_samples = int(0.008 * self._sample_rate)
        self._release_samples = int(0.055 * self._sample_rate)

        self._stream = sd.OutputStream(
            channels=1,
            samplerate=self._sample_rate,
            dtype="float32",
            blocksize=config.block_size,
            latency="low",
            callback=self._audio_callback,
        )
        self._stream.start()

    def set_volume(self, volume: float) -> None:
        self._control_queue.put(("set_volume", (max(0.0, min(1.0, volume)),)))

    def set_instrument(self, instrument_name: str) -> None:
        self._control_queue.put(("set_instrument", (str(instrument_name),)))

    def trigger_note(
        self,
        frequency_hz: float,
        duration_seconds: float,
        sustain: bool,
        chord_intervals: Optional[tuple[int, ...]] = None,
    ) -> None:
        cmd_intervals = tuple(chord_intervals) if chord_intervals else (0,)
        self._control_queue.put(
            (
                "trigger_note",
                (
                    max(20.0, frequency_hz),
                    max(0.05, duration_seconds),
                    bool(sustain),
                    cmd_intervals,
                ),
            )
        )

    def stop_note(self) -> None:
        self._control_queue.put(("stop_note", ()))

    def trigger_loop_note(self, frequency_hz: float, duration_seconds: float, velocity: float) -> None:
        self._control_queue.put(
            (
                "trigger_loop_note",
                (
                    max(20.0, frequency_hz),
                    max(0.05, duration_seconds),
                    max(0.0, min(1.0, velocity)),
                ),
            )
        )

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def start_record_capture(self) -> None:
        with self._record_lock:
            self._record_chunks = []
            self._record_capture_paused = False
            self._record_capture_active = True

    def pause_record_capture(self) -> None:
        with self._record_lock:
            if self._record_capture_active:
                self._record_capture_paused = True

    def resume_record_capture(self) -> None:
        with self._record_lock:
            if self._record_capture_active:
                self._record_capture_paused = False

    def stop_record_capture(self, wav_path: Path) -> bool:
        with self._record_lock:
            self._record_capture_active = False
            self._record_capture_paused = False
            chunks = self._record_chunks
            self._record_chunks = []
        if not chunks:
            return False

        try:
            audio = np.concatenate(chunks, axis=0)
            np.clip(audio, -1.0, 1.0, out=audio)
            pcm = (audio * 32767.0).astype(np.int16)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self._sample_rate)
                wav_file.writeframes(pcm.tobytes())
            return True
        finally:
            chunks.clear()

    def metrics_snapshot(self) -> dict[str, float]:
        return {
            "callback_count": float(self._callback_count),
            "xrun_count": float(self._xrun_count),
            "callback_cpu_avg": float(self._callback_cpu_avg),
            "loop_voice_count": float(self._loop_voice_count),
        }

    def _seconds_to_lerp(self, seconds: float) -> float:
        samples = max(1.0, seconds * self._sample_rate)
        return min(1.0, 1.0 / samples * 64.0)

    def _audio_callback(self, outdata: np.ndarray, frames: int, _time_info: object, status: sd.CallbackFlags) -> None:
        cb_start = time.perf_counter()
        if status:
            self._xrun_count += 1

        self._callback_count += 1
        self._apply_pending_commands()

        now_perf = time.perf_counter()
        if self._note_off_at_perf is not None and now_perf >= self._note_off_at_perf:
            self._target_amp = 0.0
            self._is_active = False
            self._note_off_at = None
            self._note_off_at_perf = None

        self._ensure_output_buffer(frames)
        output = self._output_buffer[:frames]
        output.fill(0.0)

        harmonic_count = int(self._harmonic_phases_np.shape[0])

        use_c_main_mixer = self._c_mixer.available and self._instrument_name in ("Sine", "Piano")
        if use_c_main_mixer:
            instrument_id = 0 if self._instrument_name == "Sine" else 1
            self._c_mixer.mix_main_voice(
                frames=frames,
                sample_rate=self._sample_rate,
                target_freq=self._target_freq,
                target_amp=self._target_amp,
                volume=self._volume,
                master_gain=self._config.master_gain,
                freq_lerp=self._freq_lerp,
                amp_attack_lerp=self._amp_attack_lerp,
                amp_release_lerp=self._amp_release_lerp,
                root_mix=self._root_mix,
                harmonic_mix=self._harmonic_mix,
                current_freq=self._state_freq,
                current_amp=self._state_amp,
                main_phase=self._state_phase,
                instrument_type=instrument_id,
                harmonic_count=harmonic_count,
                harmonic_ratios=self._harmonic_ratios_np if harmonic_count > 0 else None,
                harmonic_phases=self._harmonic_phases_np if harmonic_count > 0 else None,
                out_buffer=output,
            )
        else:
            self._mix_main_voice_python(
                output=output,
                frames=frames,
                target_freq=self._target_freq,
                target_amp=self._target_amp,
                volume=self._volume,
                instrument=self._instrument_name,
            )

        if self._loop_voice_count > 0:
            if self._c_mixer.available:
                voice_count = self._loop_voice_count
                self._c_mixer.mix(
                    frequencies=self._loop_freqs[:voice_count],
                    phases=self._loop_phases[:voice_count],
                    elapsed=self._loop_elapsed[:voice_count],
                    durations=self._loop_durations[:voice_count],
                    velocities=self._loop_velocities[:voice_count],
                    frames=frames,
                    sample_rate=self._sample_rate,
                    volume=self._volume,
                    master_gain=self._config.master_gain,
                    attack_samples=self._attack_samples,
                    release_samples=self._release_samples,
                    out_buffer=output,
                )
            else:
                self._mix_loop_voices_python(output=output, frames=frames, volume=self._volume)
            self._compact_loop_voices()

        np.tanh(output, out=output)

        cb_elapsed = time.perf_counter() - cb_start
        buffer_seconds = frames / float(self._sample_rate)
        cpu_ratio = cb_elapsed / max(buffer_seconds, 1e-6)
        self._callback_cpu_avg = (self._callback_cpu_avg * 0.98) + (cpu_ratio * 0.02)

        outdata[:, 0] = output
        if self._record_capture_active and not self._record_capture_paused:
            chunk = output.copy()
            with self._record_lock:
                if self._record_capture_active and not self._record_capture_paused:
                    self._record_chunks.append(chunk)

    def _ensure_output_buffer(self, frames: int) -> None:
        if self._output_buffer.shape[0] < frames:
            self._output_buffer = np.zeros(frames, dtype=np.float32)

    def _ensure_loop_capacity(self, required: int) -> None:
        if required <= self._loop_freqs.shape[0]:
            return

        new_capacity = max(required, self._loop_freqs.shape[0] * 2, 8)

        next_freqs = np.zeros(new_capacity, dtype=np.float32)
        next_phases = np.zeros(new_capacity, dtype=np.float32)
        next_elapsed = np.zeros(new_capacity, dtype=np.int32)
        next_durations = np.zeros(new_capacity, dtype=np.int32)
        next_velocities = np.zeros(new_capacity, dtype=np.float32)

        count = self._loop_voice_count
        next_freqs[:count] = self._loop_freqs[:count]
        next_phases[:count] = self._loop_phases[:count]
        next_elapsed[:count] = self._loop_elapsed[:count]
        next_durations[:count] = self._loop_durations[:count]
        next_velocities[:count] = self._loop_velocities[:count]

        self._loop_freqs = next_freqs
        self._loop_phases = next_phases
        self._loop_elapsed = next_elapsed
        self._loop_durations = next_durations
        self._loop_velocities = next_velocities

    def _append_loop_voice(self, frequency_hz: float, duration_samples: int, velocity: float) -> None:
        index = self._loop_voice_count
        self._ensure_loop_capacity(index + 1)
        self._loop_freqs[index] = np.float32(frequency_hz)
        self._loop_phases[index] = np.float32(0.0)
        self._loop_elapsed[index] = np.int32(0)
        self._loop_durations[index] = np.int32(duration_samples)
        self._loop_velocities[index] = np.float32(velocity)
        self._loop_voice_count = index + 1

    def _compact_loop_voices(self) -> None:
        count = self._loop_voice_count
        if count == 0:
            return

        alive = 0
        for idx in range(count):
            if self._loop_elapsed[idx] >= self._loop_durations[idx]:
                continue
            if alive != idx:
                self._loop_freqs[alive] = self._loop_freqs[idx]
                self._loop_phases[alive] = self._loop_phases[idx]
                self._loop_elapsed[alive] = self._loop_elapsed[idx]
                self._loop_durations[alive] = self._loop_durations[idx]
                self._loop_velocities[alive] = self._loop_velocities[idx]
            alive += 1
        self._loop_voice_count = alive

    def _mix_main_voice_python(
        self,
        output: np.ndarray,
        frames: int,
        target_freq: float,
        target_amp: float,
        volume: float,
        instrument: str,
    ) -> None:
        sr = float(self._sample_rate)
        root_gain = self._root_mix if self._harmonic_phases_np.size > 0 else 1.0
        harmonic_gain = self._harmonic_mix / max(1, self._harmonic_phases_np.size)

        current_freq = float(self._state_freq[0])
        current_amp = float(self._state_amp[0])
        phase = float(self._state_phase[0])
        attack_lerp = self._amp_attack_lerp
        release_lerp = self._amp_release_lerp
        if instrument == "Piano":
            attack_lerp = min(1.0, attack_lerp * 1.8)
            release_lerp = min(1.0, release_lerp * 1.2)

        for i in range(frames):
            current_freq += (target_freq - current_freq) * self._freq_lerp
            amp_lerp = attack_lerp if target_amp > current_amp else release_lerp
            current_amp += (target_amp - current_amp) * amp_lerp

            root = np.sin(2.0 * np.pi * phase) * current_amp * volume * self._config.master_gain * root_gain
            phase += current_freq / sr
            if phase >= 1.0:
                phase -= 1.0

            harmonic_mix = 0.0
            if self._harmonic_phases_np.size > 0:
                for h_idx in range(self._harmonic_phases_np.size):
                    ratio = float(self._harmonic_ratios_np[h_idx])
                    bright_boost = 1.08 if ratio > 1.8 else 1.0
                    harmonic_mix += (
                        np.sin(2.0 * np.pi * float(self._harmonic_phases_np[h_idx]))
                        * current_amp
                        * volume
                        * self._config.master_gain
                        * harmonic_gain
                        * bright_boost
                    )
                    self._harmonic_phases_np[h_idx] += np.float32((current_freq * ratio) / sr)
                    if self._harmonic_phases_np[h_idx] >= 1.0:
                        self._harmonic_phases_np[h_idx] -= np.float32(1.0)

            sample = root + harmonic_mix
            if instrument == "Piano":
                second = np.sin(2.0 * np.pi * ((phase * 2.0) % 1.0)) * current_amp * volume * self._config.master_gain * 0.24
                third = np.sin(2.0 * np.pi * ((phase * 3.0) % 1.0)) * current_amp * volume * self._config.master_gain * 0.14
                sample = (sample * 0.88) + second + third
            output[i] += sample

        self._state_freq[0] = np.float32(current_freq)
        self._state_amp[0] = np.float32(current_amp)
        self._state_phase[0] = np.float32(phase)

    def _mix_loop_voices_python(self, output: np.ndarray, frames: int, volume: float) -> None:
        sr = float(self._sample_rate)
        for voice_idx in range(self._loop_voice_count):
            elapsed = int(self._loop_elapsed[voice_idx])
            duration = int(self._loop_durations[voice_idx])
            if elapsed >= duration:
                continue

            phase = float(self._loop_phases[voice_idx])
            freq = float(self._loop_freqs[voice_idx])
            velocity = float(self._loop_velocities[voice_idx])

            for sample_idx in range(frames):
                if elapsed >= duration:
                    break
                remaining = duration - elapsed
                if self._attack_samples > 0 and elapsed < self._attack_samples:
                    envelope = elapsed / float(max(1, self._attack_samples))
                elif self._release_samples > 0 and remaining < self._release_samples:
                    envelope = remaining / float(max(1, self._release_samples))
                else:
                    envelope = 1.0

                output[sample_idx] += np.sin(2.0 * np.pi * phase) * envelope * velocity * volume * self._config.master_gain
                phase += freq / sr
                if phase >= 1.0:
                    phase -= 1.0
                elapsed += 1

            self._loop_phases[voice_idx] = np.float32(phase)
            self._loop_elapsed[voice_idx] = np.int32(elapsed)

    def _apply_pending_commands(self) -> None:
        while True:
            try:
                cmd, payload = self._control_queue.get_nowait()
            except queue.Empty:
                break

            if cmd == "set_volume":
                self._volume = float(payload[0])
                continue

            if cmd == "set_instrument":
                name = str(payload[0]).strip() if payload else "Sine"
                if name not in ("Sine", "Piano", "Drums"):
                    name = "Sine"
                self._instrument_name = name
                self._target_amp = 0.0
                self._is_active = False
                self._note_off_at = None
                self._note_off_at_perf = None
                self._harmonic_ratios_np = np.zeros(0, dtype=np.float32)
                self._harmonic_phases_np = np.zeros(0, dtype=np.float32)
                continue

            if cmd == "stop_note":
                self._target_amp = 0.0
                self._is_active = False
                self._note_off_at = None
                self._note_off_at_perf = None
                continue

            if cmd == "trigger_note":
                frequency_hz, duration_seconds, sustain, chord_intervals = payload
                if self._instrument_name == "Drums":
                    base_freq = float(frequency_hz)
                    if base_freq < 220.0:
                        drum_freq = 72.0
                        drum_duration = 0.24
                        drum_velocity = 1.0
                    elif base_freq < 440.0:
                        drum_freq = 182.0
                        drum_duration = 0.16
                        drum_velocity = 0.9
                    else:
                        drum_freq = 620.0
                        drum_duration = 0.08
                        drum_velocity = 0.7
                    self._append_loop_voice(
                        frequency_hz=drum_freq,
                        duration_samples=max(1, int(drum_duration * self._sample_rate)),
                        velocity=max(0.0, min(1.0, drum_velocity)),
                    )
                    self._target_amp = 0.0
                    self._is_active = False
                    self._note_off_at = None
                    self._note_off_at_perf = None
                    self._harmonic_ratios_np = np.zeros(0, dtype=np.float32)
                    self._harmonic_phases_np = np.zeros(0, dtype=np.float32)
                    continue

                now = time.time()
                self._target_freq = float(frequency_hz)
                self._is_active = True
                self._target_amp = self._volume
                self._note_off_at = None if sustain else now + float(duration_seconds)
                self._note_off_at_perf = None if sustain else time.perf_counter() + float(duration_seconds)

                harmonics = max(0, len(tuple(chord_intervals)) - 1)
                if harmonics == 0:
                    self._harmonic_ratios_np = np.zeros(0, dtype=np.float32)
                    self._harmonic_phases_np = np.zeros(0, dtype=np.float32)
                    continue

                detuned = np.zeros(harmonics, dtype=np.float32)
                for idx, interval in enumerate(tuple(chord_intervals)[1:]):
                    ratio = float(np.power(2.0, float(interval) / 12.0))
                    cents = 4.0 if idx % 2 == 0 else -4.0
                    detune = float(np.power(2.0, cents / 1200.0))
                    detuned[idx] = np.float32(ratio * detune)

                if self._harmonic_phases_np.shape[0] != harmonics:
                    self._harmonic_phases_np = np.zeros(harmonics, dtype=np.float32)
                    self._harmonic_ratios_np = detuned
                else:
                    self._harmonic_ratios_np[:] = detuned
                continue

            if cmd == "trigger_loop_note":
                frequency_hz, duration_seconds, velocity = payload
                duration_samples = max(1, int(float(duration_seconds) * self._sample_rate))
                self._append_loop_voice(
                    frequency_hz=float(frequency_hz),
                    duration_samples=duration_samples,
                    velocity=float(velocity),
                )
