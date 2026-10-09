"""
Atmosphere Chemical Potential Monitor — GitHub Actions

Detección:
- Todos los sismos mundiales M >= 5.
- Ventana habitual de 15 minutos.
- Recuperación del intervalo desde la última consulta exitosa.
- Revisión adicional de reportes tardíos durante 24 horas.
- Cola persistente: los eventos detectados no caducan.
- La primera ejecución comienza 15 minutos antes de ejecutarse.
- No importa pendientes antiguos de queue.json.

Análisis y publicación:
- 30 días anteriores, con muestras horarias.
- NOAA GFS 0.25°: análisis y pronósticos de 1–5 horas.
- Mapa ±20° latitud y ±10° longitud.
- Colores JET, fronteras, estrella y barra ACP en eV.
- Gráfico con puntero sincronizado.
- Threshold = promedio + 3 desviaciones estándar muestrales.
- MP4 H.264, 5 escenas/segundo.
- Publicación en X con coordenadas.
- Registros persistentes para evitar publicaciones duplicadas.

El workflow debe:
- Preparar el worktree acp-state en ACP_STATE.
- Permitir contents: write.
- Impedir ejecuciones simultáneas con concurrency.
"""

import os
import json
import time
import math
import hashlib
import subprocess
from pathlib import Path

import requests
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio_ffmpeg

from requests_oauthlib import OAuth1
from matplotlib.animation import FFMpegWriter
from eccodes import (
    codes_new_from_message,
    codes_get,
    codes_get_values,
    codes_release,
)
from tqdm import tqdm


# ================================================================
# 01 · CREDENCIALES X
# ================================================================

# Introduce credenciales NUEVAS OAuth 1.0a de usuario.
# La aplicación debe tener permisos Read and Write.
# No guardes credenciales reales en un repositorio público.

X_API_KEY = "RAtaiIdrs9kqYxg2EEIXL628Y"
X_API_SECRET = "URrbi64bTCGWCFNKi14I0xKn6tAMSRjSS5R6KEZvkRF4D4nYBF"
X_ACCESS_TOKEN = "2107910331553001473-0W2lcmo6QcxHs2z1v4uLvlBlccyidL"
X_ACCESS_TOKEN_SECRET = "OLpSbtuQbBebc4JiJeNIHP8uG8TNSIgEGpcotTpSvtDxD"


# ================================================================
# 02 · CONFIGURACIÓN
# ================================================================

VERSION = "ACP-ALL-EVENTS-15MIN-PERSISTENT-v2"

PUBLICAR_EN_X = (
    os.environ.get("PUBLISH_TO_X", "true").strip().lower()
    == "true"
)

VENTANA_MINUTOS = 15
MAGNITUD_MINIMA = 5.0

# Revisión de eventos reportados o revisados con retraso.
REVISION_TARDIA_HORAS = 24

# 0 = procesar todos los pendientes.
# Si el workflow establece 1, los demás quedan guardados.
MAX_EVENTOS_POR_EJECUCION = int(
    os.environ.get("MAX_EVENTS_PER_RUN", "0")
)

DIAS_PREVIOS = 30
FPS_ESCENAS = 5
RADIO_LATITUD = 20
RADIO_LONGITUD = 10
PASO_GRILLA = 0.25

SIGMAS_THRESHOLD = 3
COBERTURA_MINIMA_PUBLICACION = 0.95

ROOT = Path(os.environ.get("ACP_OUTPUT", "output")).resolve()
STATE = Path(os.environ.get("ACP_STATE", ".state")).resolve()
CACHE = ROOT / "cache"

for directorio in (ROOT, STATE, CACHE):
    directorio.mkdir(parents=True, exist_ok=True)

# Archivo nuevo: evita importar la cola antigua de varios días.
ARCHIVO_MONITOR = STATE / "monitor_15min_v2.json"

HTTP = requests.Session()
HTTP.headers["User-Agent"] = "ACP-Monitor/GitHub-Actions"

HORA = pd.Timedelta(hours=1)

matplotlib.rcParams["animation.ffmpeg_path"] = (
    imageio_ffmpeg.get_ffmpeg_exe()
)


# ================================================================
# 03 · UTILIDADES Y ESTADO
# ================================================================

def guardar_json(ruta, contenido):
    ruta = Path(ruta)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(
        json.dumps(
            contenido,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporal.replace(ruta)


def leer_json(ruta, defecto=None):
    ruta = Path(ruta)
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {} if defecto is None else defecto


def ruta_estado(evento):
    return STATE / f"{evento['id']}.json"


def ocultar_secretos(texto):
    texto = str(texto)
    for secreto in (
        X_API_KEY,
        X_API_SECRET,
        X_ACCESS_TOKEN,
        X_ACCESS_TOKEN_SECRET,
    ):
        if secreto:
            texto = texto.replace(secreto, "[REDACTED]")
    return texto


def coordenadas_texto(latitud, longitud):
    return (
        f"{abs(latitud):.3f}°{'N' if latitud >= 0 else 'S'}, "
        f"{abs(longitud):.3f}°{'E' if longitud >= 0 else 'W'}"
    )


def validar_credenciales():
    if not PUBLICAR_EN_X:
        return

    valores = {
        "X_API_KEY": X_API_KEY,
        "X_API_SECRET": X_API_SECRET,
        "X_ACCESS_TOKEN": X_ACCESS_TOKEN,
        "X_ACCESS_TOKEN_SECRET": X_ACCESS_TOKEN_SECRET,
    }

    faltantes = [
        nombre for nombre, valor in valores.items()
        if not valor.strip() or valor.startswith("PEGA_AQUI")
    ]

    if faltantes:
        raise RuntimeError(
            "Completa las credenciales: " + ", ".join(faltantes)
        )


def validar_directorio_estado():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return

    resultado = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=STATE,
        capture_output=True,
        text=True,
    )

    if resultado.returncode != 0:
        raise RuntimeError(
            "ACP_STATE no es un repositorio Git. "
            "El workflow debe preparar el worktree acp-state."
        )

    if Path(resultado.stdout.strip()).resolve() != STATE:
        raise RuntimeError(
            "ACP_STATE debe ser la raíz del worktree acp-state."
        )


def guardar_estado_git():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return

    validar_directorio_estado()

    subprocess.run(
        ["git", "add", "--all"], cwd=STATE, check=True
    )

    cambios = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=STATE
    ).returncode

    if cambios == 1:
        subprocess.run(
            ["git", "commit", "-m", "Update ACP monitoring state"],
            cwd=STATE,
            check=True,
        )
    elif cambios != 0:
        raise RuntimeError("No se pudo comprobar el estado Git.")

    subprocess.run(
        ["git", "push", "origin", "HEAD:acp-state"],
        cwd=STATE,
        check=True,
    )


def persistir_monitor(monitor):
    if PUBLICAR_EN_X:
        guardar_json(ARCHIVO_MONITOR, monitor)
        guardar_estado_git()


# ================================================================
# 04 · HTTP
# ================================================================

def obtener(url, **kwargs):
    for intento in range(4):
        try:
            respuesta = HTTP.get(
                url, timeout=(20, 150), **kwargs
            )
        except (requests.Timeout, requests.ConnectionError):
            if intento == 3:
                raise
            time.sleep(3 * 2**intento)
            continue

        if respuesta.status_code == 404:
            respuesta.close()
            raise FileNotFoundError(url)

        if respuesta.status_code in (429, 500, 502, 503, 504):
            codigo = respuesta.status_code
            retry_after = respuesta.headers.get("Retry-After", "")
            espera = (
                float(retry_after)
                if retry_after.isdigit()
                else 3 * 2**intento
            )
            respuesta.close()

            if intento == 3 or espera > 120:
                raise RuntimeError(
                    f"Fuente limitada o no disponible: HTTP {codigo}."
                )

            print(f"HTTP {codigo}. Reintento en {espera:.0f}s.", flush=True)
            time.sleep(espera)
            continue

        try:
            respuesta.raise_for_status()
        except Exception:
            respuesta.close()
            raise

        return respuesta

    raise RuntimeError("Consulta HTTP incompleta.")


# ================================================================
# 05 · DETECCIÓN COMPLETA Y COLA PERSISTENTE
# ================================================================

def consultar_usgs(inicio, fin):
    """
    Descarga todos los eventos elegibles.
    Divide intervalos largos en bloques diarios y pagina.
    """
    encontrados = {}
    tramo_inicio = inicio

    while tramo_inicio < fin:
        tramo_fin = min(
            tramo_inicio + pd.Timedelta(days=1), fin
        )
        offset = 1

        while True:
            respuesta = obtener(
                "https://earthquake.usgs.gov/fdsnws/event/1/query",
                params={
                    "format": "geojson",
                    "starttime": tramo_inicio.isoformat(),
                    "endtime": tramo_fin.isoformat(),
                    "minmagnitude": MAGNITUD_MINIMA,
                    "eventtype": "earthquake",
                    "orderby": "time-asc",
                    "limit": 20000,
                    "offset": offset,
                },
                headers={"Cache-Control": "no-cache"},
            )

            try:
                if respuesta.status_code == 204:
                    eventos = []
                else:
                    eventos = respuesta.json().get("features", [])
            finally:
                respuesta.close()

            for evento in eventos:
                propiedades = evento.get("properties", {})
                magnitud = propiedades.get("mag")
                tiempo = propiedades.get("time")

                if magnitud is None or tiempo is None:
                    continue

                fecha = pd.to_datetime(tiempo, unit="ms", utc=True)

                if (
                    inicio <= fecha <= fin
                    and float(magnitud) >= MAGNITUD_MINIMA
                ):
                    encontrados[evento["id"]] = evento

            if len(eventos) < 20000:
                break

            offset += 20000

        tramo_inicio = tramo_fin

    return encontrados


def actualizar_monitor():
    ahora = pd.Timestamp.now(tz="UTC")
    inicio_reciente = ahora - pd.Timedelta(minutes=VENTANA_MINUTOS)

    monitor = leer_json(ARCHIVO_MONITOR)

    if not monitor:
        monitor = {
            "version": VERSION,
            "monitoring_start_utc": inicio_reciente.isoformat(),
            "last_successful_query_utc": None,
            "events": {},
        }

        # Fija el inicio incluso si la primera consulta falla.
        persistir_monitor(monitor)

    comienzo = pd.Timestamp(monitor["monitoring_start_utc"])

    if monitor.get("last_successful_query_utc"):
        ultima = pd.Timestamp(monitor["last_successful_query_utc"])

        if ultima > ahora:
            raise RuntimeError(
                "La última consulta guardada está en el futuro. "
                "Revisa el reloj y el estado del monitor."
            )

        # Recupera el intervalo no consultado y vuelve a revisar
        # 24 horas anteriores para reportes/revisiones tardíos.
        inicio = max(
            comienzo,
            min(
                inicio_reciente,
                ultima - pd.Timedelta(hours=REVISION_TARDIA_HORAS),
            ),
        )
    else:
        inicio = comienzo

    print("\nConsulta USGS", flush=True)
    print("Ventana reciente desde:", inicio_reciente.isoformat(), flush=True)
    print("Consulta con recuperación desde:", inicio.isoformat(), flush=True)
    print("Hasta:", ahora.isoformat(), flush=True)

    encontrados = consultar_usgs(inicio, ahora)

    nuevos = 0
    recientes = 0

    for event_id, evento in encontrados.items():
        fecha = pd.to_datetime(
            evento["properties"]["time"], unit="ms", utc=True
        )

        if fecha >= inicio_reciente:
            recientes += 1

        if event_id not in monitor["events"]:
            nuevos += 1
            monitor["events"][event_id] = {
                "first_seen_utc": ahora.isoformat(),
                "detection": (
                    "recent" if fecha >= inicio_reciente
                    else "recovered"
                ),
                "event": evento,
            }
        else:
            # Conserva los parámetros más recientes de USGS.
            monitor["events"][event_id]["event"] = evento

    # Avanza la fecha SOLO cuando terminó toda la consulta.
    monitor["last_successful_query_utc"] = ahora.isoformat()
    monitor["version"] = VERSION

    # Guarda todos los eventos ANTES del análisis costoso.
    persistir_monitor(monitor)

    guardar_json(ROOT / "last_query.json", {
        "version": VERSION,
        "query_start_utc": inicio.isoformat(),
        "query_end_utc": ahora.isoformat(),
        "recent_window_minutes": VENTANA_MINUTOS,
        "events_in_recent_window": recientes,
        "new_events": nuevos,
        "returned_events": len(encontrados),
    })

    print("Sismos en los últimos 15 minutos:", recientes, flush=True)
    print("Nuevos incorporados:", nuevos, flush=True)
    print("Eventos registrados:", len(monitor["events"]), flush=True)

    return monitor


def obtener_pendientes(monitor):
    pendientes = []

    for registro in monitor.get("events", {}).values():
        evento = registro["event"]
        estado = leer_json(ruta_estado(evento))
        situacion = estado.get("status")

        if situacion == "published":
            continue

        if situacion == "posting":
            print(
                "Publicación incierta; requiere revisión:",
                evento["id"],
                flush=True,
            )
            continue

        pendientes.append(evento)

    # Los intentos fallidos no bloquean los eventos nuevos.
    pendientes.sort(
        key=lambda evento: (
            leer_json(ruta_estado(evento)).get("attempts", 0),
            evento["properties"]["time"],
        )
    )

    return pendientes


# ================================================================
# 06 · NOAA GFS
# ================================================================

def url_noaa(fecha):
    ciclo = fecha.floor("6h")
    horizonte = int((fecha - ciclo) / HORA)

    return (
        "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
        f"gfs.{ciclo:%Y%m%d}/{ciclo:%H}/atmos/"
        f"gfs.t{ciclo:%H}z.pgrb2.0p25.f{horizonte:03d}"
    )


def leer_campo(url, lineas, nombre, latitudes, longitudes, fecha):
    coincidencias = [
        i for i, linea in enumerate(lineas)
        if f":{nombre}:2 m above ground:" in linea
    ]

    if len(coincidencias) != 1:
        raise RuntimeError(f"Campo NOAA {nombre} ausente o ambiguo.")

    indice = coincidencias[0]

    if indice + 1 >= len(lineas):
        raise RuntimeError("No se puede determinar el rango GRIB.")

    inicio = int(lineas[indice].split(":")[1])
    fin = int(lineas[indice + 1].split(":")[1]) - 1

    respuesta = obtener(
        url + f"?part={inicio}",
        headers={"Range": f"bytes={inicio}-{fin}"},
        stream=True,
    )

    try:
        if respuesta.status_code != 206:
            raise RuntimeError("El servidor ignoró HTTP Range.")
        contenido = respuesta.content
    finally:
        respuesta.close()

    if len(contenido) != fin - inicio + 1:
        raise RuntimeError("Mensaje GRIB incompleto.")

    identificador = codes_new_from_message(contenido)

    try:
        if (
            codes_get(identificador, "gridType") != "regular_ll"
            or codes_get(identificador, "scanningMode") != 0
        ):
            raise RuntimeError("Orden de grilla no compatible.")

        fecha_valida = pd.to_datetime(
            str(codes_get(identificador, "validityDate"))
            + f"{int(codes_get(identificador, 'validityTime')):04d}",
            format="%Y%m%d%H%M",
            utc=True,
        )

        if fecha_valida != fecha:
            raise RuntimeError("Fecha GRIB incorrecta.")

        nx = int(codes_get(identificador, "Ni"))
        ny = int(codes_get(identificador, "Nj"))

        dx = float(codes_get(
            identificador, "iDirectionIncrementInDegrees"
        ))
        dy = float(codes_get(
            identificador, "jDirectionIncrementInDegrees"
        ))
        lon0 = float(codes_get(
            identificador, "longitudeOfFirstGridPointInDegrees"
        ))
        lat0 = float(codes_get(
            identificador, "latitudeOfFirstGridPointInDegrees"
        ))

        columnas = (
            np.rint(((longitudes - lon0) % 360) / dx)
            .astype(int) % nx
        )
        filas = np.rint((lat0 - latitudes) / dy).astype(int)

        if np.any(filas < 0) or np.any(filas >= ny):
            raise RuntimeError("Latitudes fuera de la grilla.")

        valores = (
            codes_get_values(identificador)
            .reshape(ny, nx)[np.ix_(filas, columnas)]
            .astype(float)
        )

        ausente = codes_get(identificador, "missingValue")
        valores[
            (valores == ausente) | ~np.isfinite(valores)
        ] = np.nan

        return valores

    finally:
        codes_release(identificador)


def descargar_acp(fecha, latitudes, longitudes):
    url = url_noaa(fecha)

    identificacion = (
        "acp-v1|" + url
        + str((
            latitudes[0], latitudes[-1],
            longitudes[0], longitudes[-1],
            PASO_GRILLA,
        ))
    )
    clave = hashlib.sha256(identificacion.encode()).hexdigest()
    archivo = CACHE / f"{clave}.npy"

    if archivo.exists():
        try:
            valores = np.load(archivo, allow_pickle=False)
            if valores.shape == (len(latitudes), len(longitudes)):
                return valores
        except (ValueError, OSError):
            archivo.unlink(missing_ok=True)

    respuesta = obtener(url + ".idx")
    try:
        lineas = respuesta.text.strip().splitlines()
    finally:
        respuesta.close()

    temperatura = leer_campo(
        url, lineas, "TMP", latitudes, longitudes, fecha
    ) - 273.15

    humedad = leer_campo(
        url, lineas, "RH", latitudes, longitudes, fecha
    )

    validos = (
        np.isfinite(temperatura)
        & np.isfinite(humedad)
        & (humedad > 0)
        & (humedad <= 100)
    )

    acp = np.full(temperatura.shape, np.nan, dtype=np.float32)
    acp[validos] = (
        5.8e-10
        * (20 * temperatura[validos] + 5463) ** 2
        * np.log(100 / humedad[validos])
    )

    temporal = archivo.with_suffix(".tmp")
    with temporal.open("wb") as f:
        np.save(f, acp)
    temporal.replace(archivo)

    return acp


# ================================================================
# 07 · MAPA, GRÁFICO Y VIDEO
# ================================================================

def construir_video(evento, fechas, cubo, latitudes, longitudes, salida):
    fecha_sismo = pd.to_datetime(
        evento["properties"]["time"], unit="ms", utc=True
    )
    longitud, latitud, _ = evento["geometry"]["coordinates"]

    ix = int(np.argmin(np.abs(longitudes - longitud)))
    iy = int(np.argmin(np.abs(latitudes - latitud)))

    serie = cubo[:, iy, ix]
    muestras = serie[np.isfinite(serie)].astype(float)

    promedio = float(muestras.mean()) if muestras.size else None
    desviacion = (
        float(muestras.std(ddof=1)) if muestras.size >= 2 else None
    )
    umbral = (
        promedio + SIGMAS_THRESHOLD * desviacion
        if desviacion is not None else None
    )

    estadisticas = {
        "valid_samples": int(muestras.size),
        "mean_eV": promedio,
        "sample_std_eV": desviacion,
        "threshold_eV": umbral,
        "threshold_sigmas": SIGMAS_THRESHOLD,
        "samples_above_threshold": (
            int(np.sum(muestras > umbral))
            if umbral is not None else None
        ),
    }
    guardar_json(salida / "statistics.json", estadisticas)

    tabla = pd.DataFrame({
        "time_utc": fechas,
        "ACP_eV": serie,
        "threshold_eV": umbral if umbral is not None else np.nan,
        "source": [
            "analysis" if fecha.hour % 6 == 0
            else f"{fecha.hour % 6}h forecast"
            for fecha in fechas
        ],
    })
    tabla["above_threshold"] = [
        bool(valor > umbral)
        if umbral is not None and np.isfinite(valor)
        else pd.NA
        for valor in serie
    ]
    tabla.to_csv(salida / "hourly_ACP.csv", index=False)

    np.savez_compressed(
        salida / "ACP_grids.npz",
        ACP_eV=cubo,
        latitude=latitudes,
        longitude=longitudes,
        time_utc=fechas.astype(str).to_numpy(dtype=str),
    )

    maximo = max(float(np.nanmax(cubo)), 0.001)
    colores = plt.get_cmap("jet").copy()
    colores.set_bad("#c7cdd0")

    figura = plt.figure(
        figsize=(9.6, 9.92), dpi=100, facecolor="white"
    )
    distribucion = figura.add_gridspec(
        2, 2,
        height_ratios=[2.4, 1],
        width_ratios=[1, 0.045],
        hspace=0.32,
        wspace=0.12,
    )

    proyeccion = ccrs.PlateCarree(central_longitude=longitud)
    mapa = figura.add_subplot(
        distribucion[0, 0], projection=proyeccion
    )
    barra = figura.add_subplot(distribucion[0, 1])
    grafico = figura.add_subplot(distribucion[1, :])

    x = longitudes - longitud

    mapa.set_extent([
        x.min() - PASO_GRILLA / 2,
        x.max() + PASO_GRILLA / 2,
        max(-90, latitudes.min() - PASO_GRILLA / 2),
        min(90, latitudes.max() + PASO_GRILLA / 2),
    ], crs=proyeccion)

    imagen = mapa.pcolormesh(
        x, latitudes, np.ma.masked_invalid(cubo[0]),
        cmap=colores, vmin=0, vmax=maximo,
        shading="nearest", transform=proyeccion,
    )

    mapa.coastlines(resolution="110m", linewidth=0.7)
    mapa.add_feature(
        cfeature.BORDERS.with_scale("110m"), linewidth=0.5
    )
    mapa.plot(
        0, latitud,
        marker="*", color="#ffe04b",
        markeredgecolor="black", markersize=15,
        transform=proyeccion, zorder=6,
    )

    rejilla = mapa.gridlines(
        draw_labels=True, linewidth=0.3, alpha=0.5
    )
    rejilla.top_labels = False
    rejilla.right_labels = False

    figura.colorbar(imagen, cax=barra, label="ACP (eV)")

    grafico.plot(
        fechas, serie, color="#127660",
        linewidth=1, label="Hourly ACP",
    )

    if umbral is not None:
        grafico.axhline(
            umbral, color="#c93636", linestyle="--",
            linewidth=1.5,
            label=(
                f"Threshold: mean + {SIGMAS_THRESHOLD} SD"
                f" = {umbral:.5f} eV"
            ),
        )

    limite_y = max(
        float(muestras.max()) if muestras.size else 0,
        umbral if umbral is not None else 0,
        1e-6,
    )
    grafico.set_ylim(0, limite_y * 1.12)
    grafico.legend(loc="upper left", fontsize=7, framealpha=0.9)
    grafico.set(
        xlim=(
            fecha_sismo - pd.Timedelta(days=DIAS_PREVIOS),
            fecha_sismo,
        ),
        ylabel="ACP (eV)",
        xlabel="Day (UTC) — hourly samples",
    )
    grafico.grid(alpha=0.2)
    grafico.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    grafico.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b", tz=fecha_sismo.tz)
    )
    grafico.tick_params(axis="x", labelsize=8)

    puntero = grafico.axvline(
        fechas[0], color="#c44c28", linewidth=1.8
    )
    punto, = grafico.plot(
        [], [], "o", color="#c44c28", markersize=4
    )

    lon_punto = ((longitudes[ix] + 180) % 360) - 180
    grafico.set_title(
        "Grid point nearest epicenter: "
        f"{latitudes[iy]:.2f}°, {lon_punto:.2f}°",
        fontsize=10,
    )

    titulo = figura.suptitle("", fontsize=12, color="#16382f")
    figura.text(
        0.1, 0.025,
        "Exploratory ACP · Statistical threshold, "
        "not a validated seismic alarm",
        fontsize=8, color="#637970",
    )

    escritor = FFMpegWriter(
        fps=FPS_ESCENAS,
        codec="libx264",
        bitrate=3500,
        extra_args=[
            "-r", "30",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-profile:v", "high",
        ],
    )

    video = salida / f"ACP_{DIAS_PREVIOS}days_{FPS_ESCENAS}fps.mp4"
    lugar = evento["properties"].get("place") or evento["id"]

    try:
        with escritor.saving(figura, str(video), 100):
            for k, fecha in enumerate(
                tqdm(fechas, desc="Generando MP4", mininterval=10)
            ):
                imagen.set_array(
                    np.ma.masked_invalid(cubo[k]).ravel()
                )
                puntero.set_xdata([fecha, fecha])
                punto.set_data([fecha], [serie[k]])

                titulo.set_text(
                    "Atmosphere Chemical Potential Monitor\n"
                    f"M {evento['properties']['mag']:.1f} · "
                    f"{lugar[:65]}\n"
                    f"Earthquake {fecha_sismo:%Y-%m-%d %H:%M UTC}"
                    f" | Map {fecha:%Y-%m-%d %H:%M UTC}"
                )

                if k == len(fechas) - 1:
                    figura.savefig(
                        salida / "ACP_map_chart.png", dpi=150
                    )

                escritor.grab_frame()
    finally:
        plt.close(figura)

    return video, float(np.isfinite(serie).mean()), estadisticas


# ================================================================
# 08 · PUBLICACIÓN EN X
# ================================================================

def publicar(video, evento, salida):
    archivo_estado = ruta_estado(evento)
    anterior = leer_json(archivo_estado)

    if anterior.get("status") == "published":
        print("Ya publicado:", anterior.get("post_url"), flush=True)
        return

    if anterior.get("status") == "posting":
        raise RuntimeError(
            f"Publicación incierta. Revisa X y {archivo_estado}."
        )

    validar_credenciales()

    sesion = requests.Session()
    sesion.auth = OAuth1(
        X_API_KEY.strip(), X_API_SECRET.strip(),
        X_ACCESS_TOKEN.strip(), X_ACCESS_TOKEN_SECRET.strip(),
    )

    def llamar(metodo, endpoint, reintentar=False, **kwargs):
        intentos = 5 if reintentar else 1

        for intento in range(intentos):
            try:
                respuesta = sesion.request(
                    metodo,
                    "https://api.x.com/2" + endpoint,
                    timeout=(20, 180),
                    **kwargs,
                )
            except (requests.Timeout, requests.ConnectionError):
                if intento == intentos - 1:
                    raise RuntimeError(
                        f"Conexión con X interrumpida: {metodo} {endpoint}"
                    ) from None
                time.sleep(5 * 2**intento)
                continue

            if respuesta.ok:
                try:
                    return respuesta.json() if respuesta.content else {}
                finally:
                    respuesta.close()

            codigo = respuesta.status_code
            detalle = ocultar_secretos(respuesta.text[:800])
            retry_after = respuesta.headers.get("Retry-After", "")
            respuesta.close()

            if (
                reintentar
                and codigo in (429, 500, 502, 503, 504)
                and intento < intentos - 1
            ):
                espera = (
                    float(retry_after)
                    if retry_after.isdigit()
                    else 5 * 2**intento
                )
                if espera > 300:
                    raise RuntimeError(
                        f"X HTTP {codigo}: espera de {espera:.0f}s. {detalle}"
                    )
                print(
                    f"X HTTP {codigo}. Reintento en {espera:.0f}s.",
                    flush=True,
                )
                time.sleep(espera)
                continue

            raise RuntimeError(f"X HTTP {codigo}: {detalle}")

        raise RuntimeError("Solicitud X incompleta.")

    try:
        print("Iniciando carga del video en X…", flush=True)

        respuesta = llamar(
            "POST", "/media/upload/initialize", reintentar=True,
            json={
                "media_type": "video/mp4",
                "total_bytes": video.stat().st_size,
                "media_category": "tweet_video",
            },
        )
        media_id = str(respuesta["data"]["id"])

        with video.open("rb") as archivo:
            segmento = 0
            while True:
                contenido = archivo.read(4 * 1024 * 1024)
                if not contenido:
                    break

                llamar(
                    "POST",
                    f"/media/upload/{media_id}/append",
                    reintentar=True,
                    data={"segment_index": str(segmento)},
                    files={
                        "media": (
                            "chunk.mp4", contenido,
                            "application/octet-stream",
                        )
                    },
                )
                segmento += 1

        estado = llamar(
            "POST", f"/media/upload/{media_id}/finalize"
        ).get("data", {})

        limite = time.monotonic() + 900

        while True:
            informacion = estado.get("processing_info", {})
            situacion = informacion.get("state")

            if situacion in (None, "succeeded"):
                break

            if situacion == "failed":
                raise RuntimeError(
                    "X rechazó el video: "
                    + str(informacion.get("error", ""))
                )

            if time.monotonic() > limite:
                raise RuntimeError("Procesamiento de video en X agotado.")

            espera = min(
                60, max(1, float(informacion.get("check_after_secs", 5)))
            )
            time.sleep(espera)

            estado = llamar(
                "GET", "/media/upload", reintentar=True,
                params={"command": "STATUS", "media_id": media_id},
            ).get("data", {})

        fecha = pd.to_datetime(
            evento["properties"]["time"], unit="ms", utc=True
        )
        longitud, latitud, _ = evento["geometry"]["coordinates"]
        lugar = evento["properties"].get("place") or "Global earthquake"

        texto = (
            f"M{evento['properties']['mag']:.1f} · {lugar[:50]}\n"
            f"{fecha:%Y-%m-%d %H:%M UTC}\n"
            f"Epicenter: {coordenadas_texto(latitud, longitud)}\n"
            "30-day hourly ACP evolution.\n"
            "Exploratory, not a seismic prediction.\n"
            "https://earthquake.usgs.gov/earthquakes/eventpage/"
            f"{evento['id']}"
        )
        (salida / "post_text.txt").write_text(texto, encoding="utf-8")

        guardar_json(archivo_estado, {
            "status": "posting",
            "event_id": evento["id"],
            "media_id": media_id,
            "earthquake_time_utc": fecha.isoformat(),
            "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "attempts": anterior.get("attempts", 0),
        })

        # No se envía el tweet si no fue posible guardar el bloqueo.
        guardar_estado_git()

        # No reintentar automáticamente: podría duplicar el tweet
        # si X lo creó pero se perdió la respuesta.
        resultado = llamar(
            "POST", "/tweets",
            json={
                "text": texto,
                "media": {"media_ids": [media_id]},
            },
        )

        enlace = "https://x.com/i/web/status/" + str(resultado["data"]["id"])

        guardar_json(archivo_estado, {
            "status": "published",
            "event_id": evento["id"],
            "post_url": enlace,
            "media_id": media_id,
            "earthquake_time_utc": fecha.isoformat(),
            "published_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        })
        guardar_estado_git()

        print("Publicado correctamente:", enlace, flush=True)

    finally:
        sesion.close()


# ================================================================
# 09 · PROCESAMIENTO DE CADA SISMO
# ================================================================

def procesar_evento(evento):
    fecha_sismo = pd.to_datetime(
        evento["properties"]["time"], unit="ms", utc=True
    )
    longitud, latitud, profundidad = evento["geometry"]["coordinates"]

    salida = ROOT / evento["id"]
    salida.mkdir(parents=True, exist_ok=True)
    guardar_json(salida / "event.json", evento)

    print("\nProcesando:", evento["id"], flush=True)
    print("Magnitud:", evento["properties"]["mag"], flush=True)
    print("Lugar:", evento["properties"].get("place", ""), flush=True)
    print("Fecha UTC:", fecha_sismo, flush=True)
    print("Epicentro:", coordenadas_texto(latitud, longitud), flush=True)
    print("Profundidad:", profundidad, "km", flush=True)

    latitudes = np.arange(
        math.ceil(max(-90, latitud - RADIO_LATITUD) / PASO_GRILLA),
        math.floor(min(90, latitud + RADIO_LATITUD) / PASO_GRILLA) + 1,
    ) * PASO_GRILLA

    longitudes = np.arange(
        math.ceil((longitud - RADIO_LONGITUD) / PASO_GRILLA),
        math.floor((longitud + RADIO_LONGITUD) / PASO_GRILLA) + 1,
    ) * PASO_GRILLA

    ultima_hora = (
        fecha_sismo - pd.Timedelta(nanoseconds=1)
    ).floor("h")

    fechas = pd.date_range(
        end=ultima_hora, periods=DIAS_PREVIOS * 24, freq="h"
    )

    cubo = np.full(
        (len(fechas), len(latitudes), len(longitudes)),
        np.nan,
        dtype=np.float32,
    )
    faltantes = []

    for i, fecha in enumerate(
        tqdm(fechas, desc="Descargando ACP horario", mininterval=10)
    ):
        try:
            cubo[i] = descargar_acp(fecha, latitudes, longitudes)
        except FileNotFoundError:
            faltantes.append(fecha.isoformat())

    guardar_json(salida / "missing_times.json", faltantes)

    if not np.isfinite(cubo).any():
        raise RuntimeError("No se recuperaron datos NOAA válidos.")

    cobertura_mapa = float(np.isfinite(cubo).mean())

    video, cobertura_serie, estadisticas = construir_video(
        evento, fechas, cubo, latitudes, longitudes, salida
    )

    guardar_json(salida / "summary.json", {
        "version": VERSION,
        "event_id": evento["id"],
        "earthquake_time_utc": fecha_sismo.isoformat(),
        "hourly_slots": len(fechas),
        "duration_seconds": len(fechas) / FPS_ESCENAS,
        "scenes_per_second": FPS_ESCENAS,
        "map_coverage": cobertura_mapa,
        "chart_coverage": cobertura_serie,
        "missing_hours": len(faltantes),
        "source": "NOAA GFS analyses and 1–5 hour forecasts",
        "chart_statistics": estadisticas,
    })

    print("Video generado:", video, flush=True)
    print("Duración:", len(fechas) / FPS_ESCENAS, "segundos", flush=True)
    print(f"Cobertura del mapa: {cobertura_mapa:.1%}", flush=True)
    print(f"Cobertura del gráfico: {cobertura_serie:.1%}", flush=True)

    if PUBLICAR_EN_X:
        if min(cobertura_mapa, cobertura_serie) < COBERTURA_MINIMA_PUBLICACION:
            raise RuntimeError(
                "Cobertura insuficiente. El evento permanece pendiente."
            )
        publicar(video, evento, salida)
    else:
        print("Modo prueba: no se publica en X.", flush=True)


# ================================================================
# 10 · EJECUCIÓN
# ================================================================

def main():
    print("=" * 64, flush=True)
    print("Atmosphere Chemical Potential Monitor", flush=True)
    print("Versión:", VERSION, flush=True)
    print("Archivo:", Path(__file__).resolve(), flush=True)
    print("Commit:", os.environ.get("GITHUB_SHA", "local"), flush=True)
    print("Publicación:", PUBLICAR_EN_X, flush=True)
    print("=" * 64, flush=True)

    if PUBLICAR_EN_X:
        validar_directorio_estado()

    # Guarda detecciones antes de comprobar credenciales:
    # un error de configuración no elimina los eventos encontrados.
    monitor = actualizar_monitor()
    pendientes = obtener_pendientes(monitor)

    if not pendientes:
        print("No hay eventos pendientes de análisis.", flush=True)
        return

    validar_credenciales()

    seleccionados = (
        pendientes[:MAX_EVENTOS_POR_EJECUCION]
        if MAX_EVENTOS_POR_EJECUCION > 0
        else pendientes
    )

    print("Pendientes totales:", len(pendientes), flush=True)
    print("A procesar en esta ejecución:", len(seleccionados), flush=True)

    fallidos = []

    for evento in seleccionados:
        try:
            # Sin filtro de antigüedad aquí:
            # un evento detectado nunca caduca de la cola.
            procesar_evento(evento)

        except Exception as error:
            mensaje = ocultar_secretos(error)
            print(
                "No se pudo completar:", evento["id"], mensaje,
                flush=True,
            )
            fallidos.append(evento["id"])

            archivo = ruta_estado(evento)
            anterior = leer_json(archivo)

            if (
                PUBLICAR_EN_X
                and anterior.get("status") not in ("posting", "published")
            ):
                guardar_json(archivo, {
                    "status": "failed",
                    "event_id": evento["id"],
                    "attempts": anterior.get("attempts", 0) + 1,
                    "last_error": mensaje,
                    "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                })
                guardar_estado_git()

    if fallidos:
        raise RuntimeError(
            "Eventos incompletos conservados en la cola: "
            + ", ".join(fallidos)
        )

    print("Procesamiento terminado.", flush=True)


if __name__ == "__main__":
    main()
