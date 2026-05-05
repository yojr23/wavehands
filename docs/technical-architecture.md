# WaveHands: Arquitectura Tecnica

## 1. Vision general

WaveHands es una aplicacion de ejecucion en tiempo real que combina:

- Entrada de video (webcam).
- Deteccion de manos y gesto de pinza (MediaPipe).
- Seleccion musical por UI radial + hover.
- Sintesis de audio de baja latencia (sounddevice + aceleracion C opcional).
- Render de interfaz en OpenCV.
- Grabacion de video y captura de audio.

El objetivo principal es mantener una respuesta rapida de interaccion (vision + audio) con una arquitectura modular por capas.

## 2. Estructura por capas

### `src/wavehands/domain`

Contiene entidades y reglas puras:

- `models.py`: tipos base (`Point2D`, `HandPointer`, `PerformanceSettings`, etc.).
- `notes.py`: teoria musical (notas, escalas, alteraciones, mapeos a semitonos/frecuencia).
- `looping.py`: dataclasses del loop (`LoopEvent`, `LoopLayer`).

Caracteristica: sin dependencia de IO (camara, audio, GUI).

### `src/wavehands/application`

Orquesta casos de uso:

- `app_controller.py`: ciclo principal, estado global, coordinacion UI/audio/camara.
- `selection_service.py`: logica de seleccion por hover estable.
- `loop_station.py`: estado y reglas del loop (record, play, overdub, polling de eventos).

Caracteristica: capa de coordinacion.

### `src/wavehands/infrastructure`

Adaptadores a tecnologias externas:

- `camera.py`: lectura de webcam via OpenCV.
- `hand_tracker.py`: wrapper de MediaPipe Hands.
- `audio/mono_synth.py`: motor de audio tiempo real.
- `audio/c_voice_mixer.py`: puente ctypes a biblioteca C.
- `c_src/voice_mixer.c`: mezcla DSP acelerada (voz principal + voces de loop).

Caracteristica: IO, librerias nativas y recursos del sistema.

### `src/wavehands/presentation`

Render y controles visuales:

- `controls.py`: sliders/selectores/toggles.
- `note_wheel.py`: rueda radial de notas/acordes.
- `renderer.py`: HUD de estado (fps, nota, escala, instrumento, etc.).

Caracteristica: presentacion y UX.

## 3. Flujo principal en runtime

## 3.1 Pipeline de vision

1. Hilo de captura:
- Lee frame de camara.
- Aplica flip horizontal.
- Publica el frame mas reciente en cola de baja latencia.

2. Hilo de tracking:
- Consume frame.
- Ejecuta MediaPipe.
- Extrae punteros de pinza.
- Dibuja overlays solo si el modo visual de camara esta activo.
- Publica frame trackeado + punteros.

3. Hilo principal (UI/control):
- Consume ultimo resultado disponible.
- Reescala punteros al tamano actual de ventana.
- Calcula seleccion de nota y acorde.
- Dispara eventos de audio y loop.
- Renderiza lienzo final y panel.

## 3.2 Flujo musical

1. Usuario define contexto musical:
- Tonica.
- Escala.
- Modo de alteraciones (`#` o `b`).
- Instrumento.

2. La rueda de notas se actualiza dinamicamente:
- Etiquetas visibles segun escala activa.
- Cantidad de sectores variable (5, 6, 7, 12, etc. segun escala).

3. Seleccion por hover:
- Estabilidad por frames.
- Tiempo minimo de hover.
- Cooldown de re-disparo.

4. Mapeo a audio:
- `indice rueda -> offset semitonos -> frecuencia Hz`.
- Aplicacion de acorde como intervalos armonicos.

## 4. Audio y rendimiento

## 4.1 Motor de audio (`MonoSynthEngine`)

- Usa `sounddevice.OutputStream` en callback.
- Mezcla bloque a bloque en float32 mono.
- Tiene cola de control lock-free (`SimpleQueue`) para comandos:
  - `set_volume`
  - `set_instrument`
  - `trigger_note`
  - `stop_note`
  - `trigger_loop_note`

## 4.2 Aceleracion C

- `c_src/voice_mixer.c` implementa mezcla DSP de:
  - Voz principal con armonicos.
  - Voces del loop con envelope de ataque/release.
- `c_voice_mixer.py` carga libreria dinamica y define firmas ctypes.
- Si C no esta disponible, el motor cae a implementacion Python.

## 4.3 Optimizaciones ya aplicadas

- Pipeline con colas de tamano pequeno para priorizar frames recientes.
- Inferencia en ancho reducido configurable (`inference_width`).
- Fondo neutro cacheado en modo sin camara.
- Evitar dibujo de tracker cuando la camara visual esta apagada.
- Menos copias de frame en cola de grabacion cuando ya hubo `resize`.

## 5. Grabacion y exportacion

## 5.1 Flujo

- Writer thread para video (evita bloquear loop principal).
- Captura de audio en paralelo desde el callback.
- Finalizacion en thread de background.
- Prompt de nombre dentro de la app.

## 5.2 Salida

- Si hay `ffmpeg`: mux video + audio a mp4 final.
- Si no hay `ffmpeg`: video mp4 + wav separado.

## 6. Configuracion

`config.py` concentra defaults:

- Camara (resolucion/dispositivo).
- Tracker (confianzas, max manos, ancho inferencia).
- Audio (sample rate, block size, ADSR base, gain).
- UI (ancho panel).
- Metricas.

## 7. Riesgos tecnicos actuales

- `app_controller.py` esta sobrecargado (orquestacion + estado + grabacion + render).
- Cobertura de tests aun baja para regresion en tiempo real.
- Parte del DSP instrumental (piano/drums) sigue en Python y puede migrarse a C.

## 8. Siguientes mejoras tecnicas recomendadas

1. Dividir `app_controller` en servicios:
- `render_pipeline`
- `performance_engine`
- `recording_service`

2. Extender C DSP:
- Portar modelado de `Piano` y `Drums` a C.

3. Instrumentar benchmarks:
- frame-time p50/p95
- callback cpu avg
- xruns por minuto

4. Agregar tests:
- unit tests para `notes.py` y `loop_station.py`
- pruebas de integracion para mapeo seleccion->frecuencia.
