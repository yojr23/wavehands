# WaveHands MVP

Sintetizador monofonico controlado con manos usando MediaPipe y OpenCV.

## Objetivo de esta version

- Seleccionar notas con la mano en un circulo de 7 notas: `Do, Re, Mi, Fa, Sol, La, Si`.
- Configurar `volumen`, `longitud` y `rango` con mouse.
- Reproduccion de audio en tiempo real (seno) con `sounddevice`.
- Pedal virtual de sustain por gesto (mantiene nota aunque retires la mano).
- Loop station por gesto: grabar, reproducir en bucle y overdub.

## Requisitos

- Python `3.11.10` (fijado con `.python-version`)
- Webcam
- Salida de audio del sistema

## Instalacion

```bash
pyenv local 3.11.10
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Compilar helper C (recomendado para menor latencia)

```bash
./scripts/build_c_helpers.sh
```

## Ejecucion

```bash
source .venv/bin/activate
python app.py
```

- Inicia en pantalla completa por defecto.
- `f` alterna pantalla completa / ventana.

- Salir con `ESC` o `q`.

## Interaccion

- Mueve la punta del indice sobre el circulo de notas.
- Mantener `hover` ~0.5s selecciona la nota.
- Con `Sustain` apagado, la nota dura lo definido por `Longitud`.
- Con `Sustain` encendido, la nota se mantiene hasta cambiar de nota.
- El circulo de acordes/subnotas modifica el color armonico del sonido en tiempo real.
- `Sustain` se controla por checkbox en el panel derecho.
- Grabacion de video en panel derecho:
  - `Grabar` inicia
  - `Pausar/Reanudar` alterna durante la toma
  - `Detener` cierra la toma y abre un cuadro dentro de la app para escribir el nombre.
- Exportacion con audio:
  - Si `ffmpeg` esta instalado, se guarda un `.mp4` con audio integrado.
  - Si `ffmpeg` no esta instalado, se guarda el `.mp4` de video y un `.wav` separado en `~/Downloads`.
- La ventana ahora es responsive: puedes redimensionarla y la UI se adapta (con mapeo correcto de mouse).

## Arquitectura

```text
app.py
src/wavehands/
  application/
    app_controller.py
    loop_station.py
    selection_service.py
  domain/
    looping.py
    models.py
    notes.py
  infrastructure/
    camera.py
    hand_tracker.py
    audio/mono_synth.py
    audio/c_voice_mixer.py
  presentation/
    note_wheel.py
    gesture_pedals.py
    controls.py
    renderer.py
  config.py
```

## Rendimiento

- Resolucion de proceso optimizada por defecto: `640x360`.
- Inferencia de manos en ancho reducido (`256px`) para subir FPS.
- Buffer de audio bajo: `64` muestras (latencia menor).
- Mezcla de voces del loop acelerada en C (`c_src/voice_mixer.c`) con fallback Python.
- Seleccion rapida de nota/acorde (`hover` corto y `cooldown` bajo).

## Logs y metricas

El programa imprime metricas periodicas en terminal (cada ~2s):

- `fps`
- `% frames con manos detectadas`
- `nota/acorde actuales`
- `numero de cambios de nota/acorde`
- `toggles de sustain` y cambios de loop
- `audio_cb_rate`, `audio_xruns`, `audio_cpu` y `voices`

## Notas

- El MVP actual es monofonico por diseno.
- `mingus` y `midiutil` estan instalados para la siguiente iteracion (acordes/grabacion MIDI).
