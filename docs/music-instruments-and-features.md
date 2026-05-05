# WaveHands: Musica, Instrumentos y Features

## 1. Modelo musical actual

WaveHands funciona como un instrumento gestual basado en:

- Seleccion de nota por posicion en rueda.
- Seleccion de color armonico (acorde) en segunda rueda.
- Reproduccion instantanea de nota/acorde en audio sintetizado.
- Modo loop para construir capas.

## 2. Escalas y alteraciones

## 2.1 Tonica

Se puede elegir tonica cromatica:

- `Do`, `Do#`, `Re`, `Re#`, `Mi`, `Fa`, `Fa#`, `Sol`, `Sol#`, `La`, `La#`, `Si`.

## 2.2 Alteraciones

Modo visual para nombres:

- `Sostenidos (#)` (ej: `Do#`, `Re#`, `Fa#`).
- `Bemoles (b)` (ej: `Reb`, `Mib`, `Solb`).

Ambos modos apuntan al mismo semitono real.

## 2.3 Escalas disponibles

- Mayor
- Menor (natural)
- Pentatonica Mayor
- Pentatonica Menor
- Blues
- Cromatica

La rueda de notas adapta dinamicamente:

- Etiquetas.
- Numero de sectores.
- Mapeo musical.

## 3. Mapeo de seleccion a sonido

Pipeline logico:

1. `indice de sector` seleccionado.
2. `offset de semitono` segun escala activa.
3. `semitono absoluto` segun tonica.
4. `frecuencia` por formula temperada (A4=440Hz).
5. Aplicacion de `octave_shift`.

## 4. Instrumentos actuales

## 4.1 Sine

- Sonido base mas limpio.
- Usa mezcla principal optimizada (C cuando disponible).
- Ideal para afinacion y pruebas de latencia.

## 4.2 Piano

- Ataque mas rapido.
- Mayor contenido armonico.
- Decaimiento mas musical que `Sine`.
- Implementacion principal actual en ruta Python DSP.

## 4.3 Drums

- Disparo one-shot percusivo.
- Mapeo por rango de frecuencia:
  - Bajo -> golpe tipo kick.
  - Medio -> golpe tipo snare.
  - Alto -> golpe tipo hat.

No es sample-based aun; es sintesis percusiva simplificada.

## 5. Loop station

Estados actuales:

- `idle`
- `recording`
- `playing`
- `overdubbing`

Capacidades:

- Grabar capa base.
- Reproducir en bucle.
- Agregar nuevas capas en overdub.

Cada evento guarda:

- offset temporal
- frecuencia
- duracion
- velocidad

## 6. Features actuales de UX musical

- Sustain ON/OFF.
- Rango de octava.
- Duracion de nota.
- Volumen master.
- Modo visual con camara o fondo neutro.
- HUD con escala e instrumento activo.

## 7. Limitaciones actuales

- Salida mono.
- No hay cuantizacion ritmica de eventos.
- No hay export MIDI de la performance.
- Drums y piano aun pueden mejorar su realismo timbrico.

## 8. Propuesta de nuevas features (priorizada)

## 8.1 Corto plazo (alto impacto)

1. Cuantizacion opcional de loop:
- 1/4, 1/8, 1/16.

2. Metronomo visual + audio:
- Cuenta previa (count-in) al grabar loop.

3. Presets musicales:
- Guardar/recuperar combinaciones de tonica, escala, instrumento, octava y sustain.

4. Export MIDI:
- Notas, duraciones, velocity, timestamps.

## 8.2 Mediano plazo

1. Arpegiador:
- patrones ascendentes, descendentes, random.

2. Instrumentos avanzados:
- Bass, lead, pad, pluck.

3. Modo acordes inteligentes:
- Triadas/7as diatonicas de la escala activa.

4. Control por gesto extra:
- mutear capa.
- borrar ultima capa.
- tap tempo.

## 8.3 Largo plazo

1. Motor polyfonico real por voz:
- voice allocation + release tails.

2. Escenas performables:
- cambios rapidos de set durante presentacion.

3. Colaboracion y sincronizacion:
- clock externo (MIDI clock / Ableton Link).

## 9. Recomendaciones de optimizacion musical

1. Portar DSP de `Piano` y `Drums` a C para estabilidad de latencia.
2. Introducir envelopes por instrumento (ADSR configurable por preset).
3. Agregar limitador suave final para evitar clipping en capas de loop.
4. Normalizar gain por instrumento para consistencia de volumen percibido.
