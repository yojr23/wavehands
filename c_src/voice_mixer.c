#include <math.h>
#include <stddef.h>

// Mixes main live voice (root + optional harmonics) in-place into out_buffer.
// Updates current_freq/current_amp/main_phase and harmonic_phases.
void mix_main_voice(
    int frames,
    int sample_rate,
    double target_freq,
    double target_amp,
    double volume,
    double master_gain,
    double freq_lerp,
    double amp_attack_lerp,
    double amp_release_lerp,
    double root_mix,
    double harmonic_mix,
    double* current_freq,
    double* current_amp,
    double* main_phase,
    int harmonic_count,
    const double* harmonic_ratios,
    double* harmonic_phases,
    float* out_buffer
) {
    if (
        frames <= 0 || sample_rate <= 0 || current_freq == NULL || current_amp == NULL ||
        main_phase == NULL || out_buffer == NULL
    ) {
        return;
    }

    const double two_pi = 6.28318530717958647692;
    const double inv_sr = 1.0 / (double)sample_rate;
    const double root_gain = (harmonic_count > 0) ? root_mix : 1.0;
    const double harmonic_gain = (harmonic_count > 0) ? (harmonic_mix / (double)harmonic_count) : 0.0;

    double cfreq = *current_freq;
    double camp = *current_amp;
    double phase = *main_phase;

    for (int i = 0; i < frames; i++) {
        cfreq += (target_freq - cfreq) * freq_lerp;
        const double amp_lerp = (target_amp > camp) ? amp_attack_lerp : amp_release_lerp;
        camp += (target_amp - camp) * amp_lerp;

        double sample = sin(two_pi * phase) * camp * volume * master_gain * root_gain;
        phase += cfreq * inv_sr;
        if (phase >= 1.0) {
            phase -= floor(phase);
        }

        if (harmonic_count > 0 && harmonic_ratios != NULL && harmonic_phases != NULL) {
            double hmix = 0.0;
            for (int h = 0; h < harmonic_count; h++) {
                const double hfreq = cfreq * harmonic_ratios[h];
                const double bright_boost = (harmonic_ratios[h] > 1.8) ? 1.08 : 1.0;
                hmix += sin(two_pi * harmonic_phases[h]) * camp * volume * master_gain * harmonic_gain * bright_boost;
                harmonic_phases[h] += hfreq * inv_sr;
                if (harmonic_phases[h] >= 1.0) {
                    harmonic_phases[h] -= floor(harmonic_phases[h]);
                }
            }
            sample += hmix;
        }

        out_buffer[i] += (float)sample;
    }

    *current_freq = cfreq;
    *current_amp = camp;
    *main_phase = phase;
}

// Mixes loop voices in-place into out_buffer.
// All arrays must have voice_count length.
void mix_loop_voices(
    int voice_count,
    const double* frequencies,
    double* phases,
    int* elapsed_samples,
    const int* duration_samples,
    const double* velocities,
    int frames,
    int sample_rate,
    float volume,
    float master_gain,
    int attack_samples,
    int release_samples,
    float* out_buffer
) {
    if (voice_count <= 0 || frames <= 0 || sample_rate <= 0) {
        return;
    }

    const double two_pi = 6.28318530717958647692;
    const double inv_sr = 1.0 / (double)sample_rate;

    for (int v = 0; v < voice_count; v++) {
        int elapsed = elapsed_samples[v];
        const int duration = duration_samples[v];
        if (elapsed >= duration) {
            continue;
        }

        double phase = phases[v];
        const double freq = frequencies[v];
        const double velocity = velocities[v];

        for (int i = 0; i < frames; i++) {
            if (elapsed >= duration) {
                break;
            }

            const int remaining = duration - elapsed;
            double envelope = 1.0;
            if (attack_samples > 0 && elapsed < attack_samples) {
                envelope = (double)elapsed / (double)attack_samples;
            } else if (release_samples > 0 && remaining < release_samples) {
                envelope = (double)remaining / (double)release_samples;
            }

            const double sample = sin(two_pi * phase) * envelope * velocity * volume * master_gain;
            out_buffer[i] += (float)sample;

            phase += freq * inv_sr;
            if (phase >= 1.0) {
                phase -= floor(phase);
            }
            elapsed++;
        }

        phases[v] = phase;
        elapsed_samples[v] = elapsed;
    }
}
