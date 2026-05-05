#include <cmath>
#include <cstddef>

extern "C" {

void mix_main_voice(
    int frames,
    int sample_rate,
    float target_freq,
    float target_amp,
    float volume,
    float master_gain,
    float freq_lerp,
    float amp_attack_lerp,
    float amp_release_lerp,
    float root_mix,
    float harmonic_mix,
    float* current_freq,
    float* current_amp,
    float* main_phase,
    int instrument_type,
    int harmonic_count,
    const float* harmonic_ratios,
    float* harmonic_phases,
    float* out_buffer
) {
    if (
        frames <= 0 || sample_rate <= 0 || current_freq == nullptr || current_amp == nullptr ||
        main_phase == nullptr || out_buffer == nullptr
    ) {
        return;
    }

    const float two_pi = 6.28318530717958647692f;
    const float inv_sr = 1.0f / static_cast<float>(sample_rate);
    const float root_gain = (harmonic_count > 0) ? root_mix : 1.0f;
    const float harmonic_gain = (harmonic_count > 0) ? (harmonic_mix / static_cast<float>(harmonic_count)) : 0.0f;

    float cfreq = *current_freq;
    float camp = *current_amp;
    float phase = *main_phase;

    const int INST_SINE = 0;
    const int INST_PIANO = 1;

    for (int i = 0; i < frames; i++) {
        cfreq += (target_freq - cfreq) * freq_lerp;
        const float amp_lerp = (target_amp > camp) ? amp_attack_lerp : amp_release_lerp;
        camp += (target_amp - camp) * amp_lerp;

        float sample = 0.0f;
        if (instrument_type == INST_PIANO) {
            // A richer piano-like voicing with body and hammer noise.
            const float root = std::sinf(two_pi * phase);
            const float body_2 = std::sinf(two_pi * std::fmod(phase * 2.0005f, 1.0f));
            const float body_3 = std::sinf(two_pi * std::fmod(phase * 3.995f, 1.0f));
            const float body_5 = std::sinf(two_pi * std::fmod(phase * 5.012f, 1.0f));
            float attack_punch = 0.0f;
            if (target_amp > camp) {
                attack_punch = std::sinf(two_pi * std::fmod(phase * 8.0f, 1.0f)) * 0.08f;
            }
            sample = (
                root * 0.44f +
                body_2 * 0.22f +
                body_3 * 0.14f +
                body_5 * 0.08f +
                attack_punch
            ) * camp * volume * master_gain * root_gain;

            // Add subtle inharmonic string character for piano-like warmth.
            sample += std::sinf(two_pi * std::fmod(phase * 1.997f, 1.0f)) * camp * volume * master_gain * 0.04f;
        } else {
            sample = std::sinf(two_pi * phase) * camp * volume * master_gain * root_gain;
        }

        phase += cfreq * inv_sr;
        if (phase >= 1.0f) {
            phase -= 1.0f;
        }

        if (harmonic_count > 0 && harmonic_ratios != nullptr && harmonic_phases != nullptr) {
            float hmix = 0.0f;
            for (int h = 0; h < harmonic_count; h++) {
                const float hfreq = cfreq * harmonic_ratios[h];
                const float bright_boost = (harmonic_ratios[h] > 1.8f) ? 1.08f : 1.0f;
                hmix += std::sinf(two_pi * harmonic_phases[h]) * camp * volume * master_gain * harmonic_gain * bright_boost;
                harmonic_phases[h] += hfreq * inv_sr;
                if (harmonic_phases[h] >= 1.0f) {
                    harmonic_phases[h] -= 1.0f;
                }
            }
            sample += hmix;
        }

        out_buffer[i] += sample;
    }

    *current_freq = cfreq;
    *current_amp = camp;
    *main_phase = phase;
}

void mix_loop_voices(
    int voice_count,
    const float* frequencies,
    float* phases,
    int* elapsed_samples,
    const int* duration_samples,
    const float* velocities,
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

    const float two_pi = 6.28318530717958647692f;
    const float inv_sr = 1.0f / static_cast<float>(sample_rate);

    for (int v = 0; v < voice_count; v++) {
        int elapsed = elapsed_samples[v];
        const int duration = duration_samples[v];
        if (elapsed >= duration) {
            continue;
        }

        float phase = phases[v];
        const float freq = frequencies[v];
        const float velocity = velocities[v];

        for (int i = 0; i < frames; i++) {
            if (elapsed >= duration) {
                break;
            }

            const int remaining = duration - elapsed;
            float envelope = 1.0f;
            if (attack_samples > 0 && elapsed < attack_samples) {
                envelope = static_cast<float>(elapsed) / static_cast<float>(attack_samples);
            } else if (release_samples > 0 && remaining < release_samples) {
                envelope = static_cast<float>(remaining) / static_cast<float>(release_samples);
            }

            const float sample = std::sinf(two_pi * phase) * envelope * velocity * volume * master_gain;
            out_buffer[i] += sample;

            phase += freq * inv_sr;
            if (phase >= 1.0f) {
                phase -= 1.0f;
            }
            elapsed++;
        }

        phases[v] = phase;
        elapsed_samples[v] = elapsed;
    }
}

} // extern "C"
