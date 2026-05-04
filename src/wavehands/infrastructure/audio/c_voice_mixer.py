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
            frequencies.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            phases.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            elapsed.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            durations.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            velocities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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
        current_freq: ctypes.c_double,
        current_amp: ctypes.c_double,
        main_phase: ctypes.c_double,
        harmonic_count: int,
        harmonic_ratios: Optional[np.ndarray],
        harmonic_phases: Optional[np.ndarray],
        out_buffer: np.ndarray,
    ) -> tuple[float, float, float]:
        if not self._lib:
            return float(current_freq.value), float(current_amp.value), float(main_phase.value)

        if harmonic_count > 0:
            if harmonic_ratios is None or harmonic_phases is None:
                return float(current_freq.value), float(current_amp.value), float(main_phase.value)
            ratios_ptr = harmonic_ratios.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            phases_ptr = harmonic_phases.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        else:
            ratios_ptr = ctypes.POINTER(ctypes.c_double)()
            phases_ptr = ctypes.POINTER(ctypes.c_double)()

        self._lib.mix_main_voice(
            ctypes.c_int(frames),
            ctypes.c_int(sample_rate),
            ctypes.c_double(target_freq),
            ctypes.c_double(target_amp),
            ctypes.c_double(volume),
            ctypes.c_double(master_gain),
            ctypes.c_double(freq_lerp),
            ctypes.c_double(amp_attack_lerp),
            ctypes.c_double(amp_release_lerp),
            ctypes.c_double(root_mix),
            ctypes.c_double(harmonic_mix),
            ctypes.byref(current_freq),
            ctypes.byref(current_amp),
            ctypes.byref(main_phase),
            ctypes.c_int(harmonic_count),
            ratios_ptr,
            phases_ptr,
            out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return float(current_freq.value), float(current_amp.value), float(main_phase.value)

    def _configure_signatures(self) -> None:
        if not self._lib:
            return
        self._lib.mix_main_voice.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.mix_main_voice.restype = None

        self._lib.mix_loop_voices.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
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
