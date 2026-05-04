import ctypes
import queue
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from wavehands.config import AudioConfig
from wavehands.infrastructure.audio.c_voice_mixer import CVoiceMixer


@dataclass
class _LoopVoice:
    frequency_hz: float
    duration_samples: int
    velocity: float
    phase: float = 0.0
    elapsed_samples: int = 0


class MonoSynthEngine:
    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._sample_rate = config.sample_rate
        self._phase = 0.0

        self._target_freq = 261.63
        self._current_freq = 261.63
        self._target_amp = 0.0
        self._current_amp = 0.0
        self._volume = 0.5
        self._note_off_at: Optional[float] = None
        self._is_active = False
        self._loop_voices: list[_LoopVoice] = []
        self._c_mixer = CVoiceMixer()
        self._chord_intervals: tuple[int, ...] = (0,)
        self._chord_phases: list[float] = []
        self._harmonic_ratios: tuple[float, ...] = ()
        self._harmonic_phases_np = np.zeros(0, dtype=np.float64)
        self._harmonic_ratios_np = np.zeros(0, dtype=np.float64)
        self._root_mix = 0.58
        self._harmonic_mix = 1.05
        self._callback_count = 0
        self._xrun_count = 0
        self._callback_cpu_avg = 0.0
        self._output_buffer = np.zeros(max(1, config.block_size), dtype=np.float32)
        self._loop_freqs = np.zeros(0, dtype=np.float64)
        self._loop_phases = np.zeros(0, dtype=np.float64)
        self._loop_elapsed = np.zeros(0, dtype=np.int32)
        self._loop_durations = np.zeros(0, dtype=np.int32)
        self._loop_velocities = np.zeros(0, dtype=np.float64)
        self._note_off_at_perf: Optional[float] = None
        self._control_queue: queue.SimpleQueue[tuple[str, tuple]] = queue.SimpleQueue()
        self._record_capture_active = False
        self._record_capture_paused = False
        self._record_chunks: list[np.ndarray] = []

        self._stream = sd.OutputStream(
            channels=1,
            samplerate=self._sample_rate,
            dtype="float32",
            blocksize=config.block_size,
            latency="low",
            callback=self._audio_callback,
        )
        self._stream.start()

        self._freq_lerp = 0.0025
        self._amp_attack_lerp = self._seconds_to_lerp(config.attack_seconds)
        self._amp_release_lerp = self._seconds_to_lerp(config.release_seconds)

    def set_volume(self, volume: float) -> None:
        self._control_queue.put(("set_volume", (max(0.0, min(1.0, volume)),)))

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
        self._record_chunks = []
        self._record_capture_paused = False
        self._record_capture_active = True

    def pause_record_capture(self) -> None:
        if self._record_capture_active:
            self._record_capture_paused = True

    def resume_record_capture(self) -> None:
        if self._record_capture_active:
            self._record_capture_paused = False

    def stop_record_capture(self, wav_path: Path) -> bool:
        self._record_capture_active = False
        self._record_capture_paused = False
        if not self._record_chunks:
            return False

        try:
            audio = np.concatenate(self._record_chunks, axis=0)
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
            self._record_chunks = []

    def metrics_snapshot(self) -> dict[str, float]:
        return {
            "callback_count": float(self._callback_count),
            "xrun_count": float(self._xrun_count),
            "callback_cpu_avg": float(self._callback_cpu_avg),
            "loop_voice_count": float(len(self._loop_voices)),
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

        target_freq = self._target_freq
        target_amp = self._target_amp
        volume = self._volume
        chord_phases = list(self._chord_phases)
        harmonic_ratios_np = self._harmonic_ratios_np
        loop_voices = self._loop_voices

        self._ensure_audio_buffers(frames, len(loop_voices))
        output = self._output_buffer[:frames]
        output.fill(0.0)

        attack_samples = int(0.008 * self._sample_rate)
        release_samples = int(0.055 * self._sample_rate)

        harmonic_count = len(chord_phases)
        if harmonic_count > 0:
            self._harmonic_phases_np[:harmonic_count] = np.asarray(chord_phases, dtype=np.float64)
            harmonics_phase_ptr = self._harmonic_phases_np
            harmonics_ratio_ptr = harmonic_ratios_np
        else:
            harmonics_phase_ptr = None
            harmonics_ratio_ptr = None

        if self._c_mixer.available:
            current_freq = ctypes.c_double(self._current_freq)
            current_amp = ctypes.c_double(self._current_amp)
            main_phase = ctypes.c_double(self._phase)
            self._current_freq, self._current_amp, self._phase = self._c_mixer.mix_main_voice(
                frames=frames,
                sample_rate=self._sample_rate,
                target_freq=target_freq,
                target_amp=target_amp,
                volume=volume,
                master_gain=self._config.master_gain,
                freq_lerp=self._freq_lerp,
                amp_attack_lerp=self._amp_attack_lerp,
                amp_release_lerp=self._amp_release_lerp,
                root_mix=self._root_mix,
                harmonic_mix=self._harmonic_mix,
                current_freq=current_freq,
                current_amp=current_amp,
                main_phase=main_phase,
                harmonic_count=harmonic_count,
                harmonic_ratios=harmonics_ratio_ptr,
                harmonic_phases=harmonics_phase_ptr,
                out_buffer=output,
            )
            if harmonic_count > 0:
                chord_phases = self._harmonic_phases_np[:harmonic_count].tolist()
        else:
            self._mix_main_voice_python(
                output=output,
                frames=frames,
                target_freq=target_freq,
                target_amp=target_amp,
                volume=volume,
                chord_phases=chord_phases,
                harmonic_count=harmonic_count,
            )

        # Loop voices are mixed by C helper when available (fallback to Python).
        if loop_voices:
            if self._c_mixer.available:
                voice_count = len(loop_voices)
                for idx, voice in enumerate(loop_voices):
                    self._loop_freqs[idx] = voice.frequency_hz
                    self._loop_phases[idx] = voice.phase
                    self._loop_elapsed[idx] = voice.elapsed_samples
                    self._loop_durations[idx] = voice.duration_samples
                    self._loop_velocities[idx] = voice.velocity

                self._c_mixer.mix(
                    frequencies=self._loop_freqs[:voice_count],
                    phases=self._loop_phases[:voice_count],
                    elapsed=self._loop_elapsed[:voice_count],
                    durations=self._loop_durations[:voice_count],
                    velocities=self._loop_velocities[:voice_count],
                    frames=frames,
                    sample_rate=self._sample_rate,
                    volume=volume,
                    master_gain=self._config.master_gain,
                    attack_samples=attack_samples,
                    release_samples=release_samples,
                    out_buffer=output,
                )

                alive_voices: list[_LoopVoice] = []
                for idx, voice in enumerate(loop_voices):
                    voice.phase = float(self._loop_phases[idx])
                    voice.elapsed_samples = int(self._loop_elapsed[idx])
                    if voice.elapsed_samples < voice.duration_samples:
                        alive_voices.append(voice)
                loop_voices = alive_voices
            else:
                sr = float(self._sample_rate)
                loop_output = np.zeros(frames, dtype=np.float32)
                alive_voices: list[_LoopVoice] = []
                for voice in loop_voices:
                    if voice.elapsed_samples >= voice.duration_samples:
                        continue
                    for i in range(frames):
                        if voice.elapsed_samples >= voice.duration_samples:
                            break
                        remaining = voice.duration_samples - voice.elapsed_samples
                        if attack_samples > 0 and voice.elapsed_samples < attack_samples:
                            envelope = voice.elapsed_samples / float(max(1, attack_samples))
                        elif release_samples > 0 and remaining < release_samples:
                            envelope = remaining / float(max(1, release_samples))
                        else:
                            envelope = 1.0
                        loop_output[i] += np.sin(2.0 * np.pi * voice.phase) * envelope * voice.velocity * volume * self._config.master_gain
                        voice.phase += voice.frequency_hz / sr
                        if voice.phase >= 1.0:
                            voice.phase -= 1.0
                        voice.elapsed_samples += 1
                    if voice.elapsed_samples < voice.duration_samples:
                        alive_voices.append(voice)
                output += loop_output
                loop_voices = alive_voices

        np.tanh(output, out=output)

        self._loop_voices = loop_voices
        self._chord_phases = chord_phases
        cb_elapsed = time.perf_counter() - cb_start
        buffer_seconds = frames / float(self._sample_rate)
        cpu_ratio = cb_elapsed / max(buffer_seconds, 1e-6)
        self._callback_cpu_avg = (self._callback_cpu_avg * 0.98) + (cpu_ratio * 0.02)

        outdata[:, 0] = output
        if self._record_capture_active and not self._record_capture_paused:
            self._record_chunks.append(np.copy(output))

    def _ensure_audio_buffers(self, frames: int, voices: int) -> None:
        if self._output_buffer.shape[0] < frames:
            self._output_buffer = np.zeros(frames, dtype=np.float32)
        if self._loop_freqs.shape[0] < voices:
            self._loop_freqs = np.zeros(voices, dtype=np.float64)
            self._loop_phases = np.zeros(voices, dtype=np.float64)
            self._loop_elapsed = np.zeros(voices, dtype=np.int32)
            self._loop_durations = np.zeros(voices, dtype=np.int32)
            self._loop_velocities = np.zeros(voices, dtype=np.float64)

    def _mix_main_voice_python(
        self,
        output: np.ndarray,
        frames: int,
        target_freq: float,
        target_amp: float,
        volume: float,
        chord_phases: list[float],
        harmonic_count: int,
    ) -> None:
        sr = float(self._sample_rate)
        root_gain = self._root_mix if harmonic_count > 0 else 1.0
        harmonic_gain = self._harmonic_mix / harmonic_count if harmonic_count > 0 else 0.0
        for i in range(frames):
            self._current_freq += (target_freq - self._current_freq) * self._freq_lerp
            amp_lerp = self._amp_attack_lerp if target_amp > self._current_amp else self._amp_release_lerp
            self._current_amp += (target_amp - self._current_amp) * amp_lerp

            root = np.sin(2.0 * np.pi * self._phase) * self._current_amp * volume * self._config.master_gain * root_gain
            self._phase += self._current_freq / sr
            if self._phase >= 1.0:
                self._phase -= 1.0

            harmonic_mix = 0.0
            if harmonic_count > 0:
                for h_idx in range(harmonic_count):
                    freq = self._current_freq * self._harmonic_ratios[h_idx]
                    bright_boost = 1.08 if self._harmonic_ratios[h_idx] > 1.8 else 1.0
                    harmonic_mix += (
                        np.sin(2.0 * np.pi * chord_phases[h_idx])
                        * self._current_amp
                        * volume
                        * self._config.master_gain
                        * harmonic_gain
                        * bright_boost
                    )
                    chord_phases[h_idx] += freq / sr
                    if chord_phases[h_idx] >= 1.0:
                        chord_phases[h_idx] -= 1.0

            output[i] += root + harmonic_mix

    def _apply_pending_commands(self) -> None:
        while True:
            try:
                cmd, payload = self._control_queue.get_nowait()
            except queue.Empty:
                break

            if cmd == "set_volume":
                self._volume = float(payload[0])
                continue

            if cmd == "stop_note":
                self._target_amp = 0.0
                self._is_active = False
                self._note_off_at = None
                self._note_off_at_perf = None
                continue

            if cmd == "trigger_note":
                frequency_hz, duration_seconds, sustain, chord_intervals = payload
                now = time.time()
                self._target_freq = float(frequency_hz)
                self._is_active = True
                self._target_amp = self._volume
                self._note_off_at = None if sustain else now + float(duration_seconds)
                self._note_off_at_perf = None if sustain else time.perf_counter() + float(duration_seconds)
                self._chord_intervals = tuple(chord_intervals)
                harmonics = max(0, len(self._chord_intervals) - 1)
                base_ratios = [float(np.power(2.0, interval / 12.0)) for interval in self._chord_intervals[1:]]
                detuned_ratios: list[float] = []
                for idx, ratio in enumerate(base_ratios):
                    cents = 4.0 if idx % 2 == 0 else -4.0
                    detune = float(np.power(2.0, cents / 1200.0))
                    detuned_ratios.append(ratio * detune)
                self._harmonic_ratios = tuple(detuned_ratios)
                if harmonics != len(self._chord_phases):
                    self._chord_phases = [0.0 for _ in range(harmonics)]
                    self._harmonic_phases_np = np.zeros(harmonics, dtype=np.float64)
                    self._harmonic_ratios_np = (
                        np.asarray(self._harmonic_ratios, dtype=np.float64) if harmonics > 0 else np.zeros(0, dtype=np.float64)
                    )
                elif harmonics > 0:
                    self._harmonic_ratios_np[:] = np.asarray(self._harmonic_ratios, dtype=np.float64)
                continue

            if cmd == "trigger_loop_note":
                frequency_hz, duration_seconds, velocity = payload
                duration_samples = max(1, int(float(duration_seconds) * self._sample_rate))
                self._loop_voices.append(
                    _LoopVoice(
                        frequency_hz=float(frequency_hz),
                        duration_samples=duration_samples,
                        velocity=float(velocity),
                    )
                )
