from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

import numpy as np


class CVoiceMixer:
    def __init__(self) -> None:
        self._lib = self._load_library()
        self.available = self._lib is not None
        if self._lib is not None:
            self._configure_signatures()

    def mix(
        self,
        frequencies: np.ndarray,
        phases: np.ndarray,
        elapsed: np.ndarray,
        durations: np.ndarray,
        velocities: np.ndarray,
        frames: int,
        sample_rate: int,
        volume: float,
        master_gain: float,
        attack_samples: int,
        release_samples: int,
        out_buffer: np.ndarray,
    ) -> None:
        if not self._lib:
            return

        voice_count = int(len(frequencies))
        if voice_count == 0:
            return

        self._lib.mix_loop_voices(
            ctypes.c_int(voice_count),
            frequencies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            phases.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            elapsed.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            durations.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            velocities.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(frames),
            ctypes.c_int(sample_rate),
            ctypes.c_float(volume),
            ctypes.c_float(master_gain),
            ctypes.c_int(attack_samples),
            ctypes.c_int(release_samples),
            out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def mix_main_voice(
        self,
        frames: int,
        sample_rate: int,
        target_freq: float,
        target_amp: float,
        volume: float,
        master_gain: float,
        freq_lerp: float,
        amp_attack_lerp: float,
        amp_release_lerp: float,
        root_mix: float,
        harmonic_mix: float,
        current_freq: np.ndarray,
        current_amp: np.ndarray,
        main_phase: np.ndarray,
        instrument_type: int,
        harmonic_count: int,
        harmonic_ratios: Optional[np.ndarray],
        harmonic_phases: Optional[np.ndarray],
        out_buffer: np.ndarray,
    ) -> None:
        if not self._lib:
            return

        if harmonic_count > 0:
            if harmonic_ratios is None or harmonic_phases is None:
                return
            ratios_ptr = harmonic_ratios.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            phases_ptr = harmonic_phases.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            ratios_ptr = ctypes.POINTER(ctypes.c_float)()
            phases_ptr = ctypes.POINTER(ctypes.c_float)()

        self._lib.mix_main_voice(
            ctypes.c_int(frames),
            ctypes.c_int(sample_rate),
            ctypes.c_float(target_freq),
            ctypes.c_float(target_amp),
            ctypes.c_float(volume),
            ctypes.c_float(master_gain),
            ctypes.c_float(freq_lerp),
            ctypes.c_float(amp_attack_lerp),
            ctypes.c_float(amp_release_lerp),
            ctypes.c_float(root_mix),
            ctypes.c_float(harmonic_mix),
            current_freq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            current_amp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            main_phase.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(instrument_type),
            ctypes.c_int(harmonic_count),
            ratios_ptr,
            phases_ptr,
            out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def _configure_signatures(self) -> None:
        if not self._lib:
            return
        self._lib.mix_main_voice.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.mix_main_voice.restype = None

        self._lib.mix_loop_voices.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.mix_loop_voices.restype = None

    def _load_library(self) -> Optional[ctypes.CDLL]:
        root = Path(__file__).resolve().parents[4]
        build = root / "build"
        candidates = (
            build / "libvoice_mixer.dylib",
            build / "libvoice_mixer.so",
            build / "voice_mixer.dll",
        )
        for lib_path in candidates:
            if lib_path.exists():
                try:
                    return ctypes.CDLL(str(lib_path))
                except OSError:
                    continue
        return None
