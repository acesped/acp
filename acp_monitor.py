"""
Atmosphere Chemical Potential Monitor.

Detecta sismos mundiales M >= 5, genera un video con mapa y gráfico
ACP de los 30 días anteriores y publica en X.

El estado persistente evita repetir publicaciones.
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
# CONFIGURACIÓN
# ================================================================

PUBLICAR_EN_X = (
    os.environ.get("PUBLISH_TO_X", "true").lower() == "true"
)

FPS = 5
DIAS_PREVIOS = 30
RADIO_LATITUD = 20
RADIO_LONGITUD = 10
PASO = 0.25
SIGMAS_THRESHOLD = 3

COBERTURA_MINIMA = 0.95
MAX_EVENTOS = int(os.environ.get("MAX_EVENTS_PER_RUN", "1"))

ROOT = Path(os.environ.get("ACP_OUTPUT", "output"))
STATE = Path(os.environ.get("ACP_STATE", ".state"))
CACHE = ROOT / "cache"

ROOT.mkdir(parents=True, exist_ok=True)
STATE.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

HTTP = requests.Session()
HTTP.headers["User-Agent"] = "ACP-Monitor/GitHub-Actions"

HORA = pd.Timedelta(hours=1)


# ================================================================
# ESTADO PERSISTENTE
# ================================================================

def guardar_json(ruta, contenido):
    temporal = ruta.with_suffix(".tmp")
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
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {} if defecto is None else defecto


def guardar_estado_git():
    """Guarda únicamente la rama de estado acp-state."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return

    subprocess.run(
        ["git", "add", "--all"],
        cwd=STATE,
        check=True,
    )

    cambios = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=STATE,
    ).returncode

    if cambios == 1:
        subprocess.run(
            ["git", "commit", "-m", "Update ACP publication state"],
            cwd=STATE,
            check=True,
        )
        subprocess.run(
            ["git", "push", "origin", "HEAD:acp-state"],
            cwd=STATE,
            check=True,
        )
    elif cambios != 0:
        raise RuntimeError("No se pudo comprobar el estado de Git.")


def ruta_estado(evento):
    return STATE / f"{evento['id']}.json"


# ================================================================
# CONSULTAS HTTP
# ================================================================

def obtener(url, **kwargs):
    """Reintenta lecturas; no se usa para crear publicaciones."""
    for intento in range(4):
        try:
            respuesta = HTTP.get(
                url,
                timeout=(20, 150),
                **kwargs,
            )

            if respuesta.status_code == 404:
                respuesta.close()
                raise FileNotFoundError(url)

            if respuesta.status_code in (429, 500, 502, 503, 504):
                espera = respuesta.headers.get("Retry-After", "")
                espera = (
                    float(espera)
                    if espera.isdigit()
                    else 3 * 2**intento
                )
                respuesta.close()

                if intento == 3 or espera > 120:
                    raise RuntimeError(
                        "Fuente limitada o no disponible. "
                        "Se conserva la caché para reintentar."
                    )

                time.sleep(espera)
                continue

            respuesta.raise_for_status()
            return respuesta

        except (requests.Timeout, requests.ConnectionError):
            if intento == 3:
                raise
            time.sleep(3 * 2**intento)

    raise RuntimeError("No se pudo completar la consulta.")


# ================================================================
# CATÁLOGO USGS
# ================================================================

def actualizar_cola():
    archivo = STATE / "queue.json"

    estado = leer_json(
        archivo,
        {"initialized": False, "events": {}},
    )

    ahora = pd.Timestamp.now(tz="UTC")

    # Primera ejecución: últimas 24 horas.
    # Siguientes: revisar 7 días por reportes tardíos o revisiones.
    dias = 7 if estado["initialized"] else 1
    inicio = ahora - pd.Timedelta(days=dias)

    offset = 1

    while True:
        respuesta = obtener(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={
                "format": "geojson",
                "starttime": inicio.isoformat(),
                "endtime": ahora.isoformat(),
                "minmagnitude": 5,
                "eventtype": "earthquake",
                "orderby": "time-asc",
                "limit": 20000,
                "offset": offset,
            },
        )

        eventos = respuesta.json().get("features", [])

        for evento in eventos:
            estado["events"][evento["id"]] = evento

        if len(eventos) < 20000:
            break

        offset += 20000

    estado["initialized"] = True

    if PUBLICAR_EN_X:
        guardar_json(archivo, estado)
        guardar_estado_git()

    pendientes = []

    for evento in estado["events"].values():
        anterior = leer_json(ruta_estado(evento))
        situacion = anterior.get("status")

        if situacion == "published":
            continue

        if situacion == "posting":
            print(
                "Revisión manual: publicación anterior incierta:",
                evento["id"],
            )
            continue

        pendientes.append(evento)

    def prioridad(evento):
        anterior = leer_json(ruta_estado(evento))
        return (
            anterior.get("attempts", 0),
            evento["properties"]["time"],
        )

    # Los fallidos no bloquean todos los nuevos eventos.
    pendientes.sort(key=prioridad)

    return pendientes


# ================================================================
# NOAA GFS
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
        i
        for i, linea in enumerate(lineas)
        if f":{nombre}:2 m above ground:" in linea
    ]

    if len(coincidencias) != 1:
        raise RuntimeError("Campo NOAA ausente o ambiguo.")

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
            raise RuntimeError(
                "El servidor ignoró HTTP Range; "
                "se evita descargar el archivo global completo."
            )
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
            raise RuntimeError("Orden de grilla GFS no compatible.")

        fecha_valida = pd.to_datetime(
            str(codes_get(identificador, "validityDate"))
            + f"{int(codes_get(identificador, 'validityTime')):04d}",
            format="%Y%m%d%H%M",
            utc=True,
        )

        if fecha_valida != fecha:
            raise RuntimeError("Fecha GRIB diferente de la solicitada.")

        nx = int(codes_get(identificador, "Ni"))
        ny = int(codes_get(identificador, "Nj"))

        dx = codes_get(
            identificador, "iDirectionIncrementInDegrees"
        )
        dy = codes_get(
            identificador, "jDirectionIncrementInDegrees"
        )
        lon0 = codes_get(
            identificador, "longitudeOfFirstGridPointInDegrees"
        )
        lat0 = codes_get(
            identificador, "latitudeOfFirstGridPointInDegrees"
        )

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
            (valores == ausente) | (~np.isfinite(valores))
        ] = np.nan

        return valores

    finally:
        codes_release(identificador)


def descargar_acp(fecha, latitudes, longitudes):
    url = url_noaa(fecha)

    identificacion = (
        url
        + str((
            latitudes[0],
            latitudes[-1],
            longitudes[0],
            longitudes[-1],
            PASO,
        ))
    )

    clave = hashlib.sha256(
        identificacion.encode()
    ).hexdigest()

    archivo = CACHE / f"{clave}.npy"

    if archivo.exists():
        valores = np.load(archivo, allow_pickle=False)
        if valores.shape == (len(latitudes), len(longitudes)):
            return valores

    lineas = obtener(url + ".idx").text.strip().splitlines()

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

    acp = np.full(
        temperatura.shape, np.nan, dtype=np.float32
    )

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
# VIDEO CON MAPA Y GRÁFICO
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
        float(muestras.std(ddof=1))
        if muestras.size >= 2 else None
    )
    umbral = (
        promedio + SIGMAS_THRESHOLD * desviacion
        if desviacion is not None else None
    )

    guardar_json(salida / "statistics.json", {
        "valid_samples": int(muestras.size),
        "mean_eV": promedio,
        "sample_std_eV": desviacion,
        "threshold_eV": umbral,
        "definition": "mean + 3 sample standard deviations",
    })

    pd.DataFrame({
        "time_utc": fechas,
        "ACP_eV": serie,
        "threshold_eV": umbral,
        "source": [
            "analysis" if f.hour % 6 == 0
            else f"{f.hour % 6}h forecast"
            for f in fechas
        ],
    }).to_csv(salida / "hourly_ACP.csv", index=False)

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

    proyeccion = ccrs.PlateCarree(
        central_longitude=longitud
    )
    mapa = figura.add_subplot(
        distribucion[0, 0], projection=proyeccion
    )
    barra = figura.add_subplot(distribucion[0, 1])
    grafico = figura.add_subplot(distribucion[1, :])

    x = longitudes - longitud

    mapa.set_extent([
        x.min() - PASO / 2,
        x.max() + PASO / 2,
        max(-90, latitudes.min() - PASO / 2),
        min(90, latitudes.max() + PASO / 2),
    ], crs=proyeccion)

    imagen = mapa.pcolormesh(
        x, latitudes,
        np.ma.masked_invalid(cubo[0]),
        cmap=colores,
        vmin=0,
        vmax=maximo,
        shading="nearest",
        transform=proyeccion,
    )

    mapa.coastlines(resolution="110m", linewidth=0.7)
    mapa.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        linewidth=0.5,
    )
    mapa.plot(
        0, latitud,
        marker="*",
        color="#ffe04b",
        markeredgecolor="black",
        markersize=15,
        transform=proyeccion,
        zorder=6,
    )

    rejilla = mapa.gridlines(
        draw_labels=True, linewidth=0.3, alpha=0.5
    )
    rejilla.top_labels = False
    rejilla.right_labels = False

    figura.colorbar(imagen, cax=barra, label="ACP (eV)")

    grafico.plot(
        fechas, serie,
        color="#127660", linewidth=1, label="Hourly ACP"
    )

    if umbral is not None:
        grafico.axhline(
            umbral,
            color="#c93636",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold: mean + 3 SD = {umbral:.5f} eV",
        )
        grafico.set_ylim(
            0,
            max(float(muestras.max()), umbral, 1e-6) * 1.12,
        )

    grafico.legend(loc="upper left", fontsize=7)
    grafico.set(
        xlim=(
            fecha_sismo - pd.Timedelta(days=DIAS_PREVIOS),
            fecha_sismo,
        ),
        ylabel="ACP (eV)",
        xlabel="Day (UTC) — hourly samples",
    )
    grafico.grid(alpha=0.2)
    grafico.xaxis.set_major_locator(
        mdates.DayLocator(interval=3)
    )
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

    titulo = figura.suptitle(
        "", fontsize=12, color="#16382f"
    )
    figura.text(
        0.1, 0.025,
        "Exploratory ACP · Statistical threshold, "
        "not a validated seismic alarm",
        fontsize=8,
    )

    matplotlib.rcParams["animation.ffmpeg_path"] = (
        imageio_ffmpeg.get_ffmpeg_exe()
    )

    # Cinco escenas por segundo; salida H.264 a 30 fps codificados.
    escritor = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=3500,
        extra_args=[
            "-r", "30",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-profile:v", "high",
        ],
    )

    archivo = salida / f"ACP_{DIAS_PREVIOS}days_{FPS}fps.mp4"

    try:
        with escritor.saving(figura, str(archivo), 100):
            for k, fecha in enumerate(
                tqdm(fechas, desc="Generando MP4")
            ):
                imagen.set_array(
                    np.ma.masked_invalid(cubo[k]).ravel()
                )
                puntero.set_xdata([fecha, fecha])
                punto.set_data([fecha], [serie[k]])

                lugar = (
                    evento["properties"].get("place")
                    or evento["id"]
                )
                titulo.set_text(
                    "Atmosphere Chemical Potential Monitor\n"
                    f"M {evento['properties']['mag']:.1f} · "
                    f"{lugar[:65]}\n"
                    f"Earthquake {fecha_sismo:%Y-%m-%d %H:%M UTC}"
                    f" | Map {fecha:%Y-%m-%d %H:%M UTC}"
                )

                if k == len(fechas) - 1:
                    figura.savefig(
                        salida / "ACP_map_chart.png",
                        dpi=150,
                    )

                escritor.grab_frame()
    finally:
        plt.close(figura)

    return archivo, float(np.isfinite(serie).mean())


# ================================================================
# PUBLICACIÓN EN X
# ================================================================

def publicar(video, evento, salida):
    archivo_estado = ruta_estado(evento)
    anterior = leer_json(archivo_estado)

    if anterior.get("status") == "published":
        print("Ya publicado:", anterior.get("post_url"))
        return

    if anterior.get("status") == "posting":
        raise RuntimeError(
            f"Resultado anterior incierto. Revisa X y "
            f"acp-state/{evento['id']}.json antes de reintentar."
        )

    nombres = [
        "X_API_KEY",
        "X_API_SECRET",
        "X_ACCESS_TOKEN",
        "X_ACCESS_TOKEN_SECRET",
    ]
    credenciales = [os.environ.get(n, "") for n in nombres]

    if not all(credenciales):
        raise RuntimeError(
            "Faltan GitHub Secrets: "
            + ", ".join(
                n for n, v in zip(nombres, credenciales) if not v
            )
        )

    sesion = requests.Session()
    sesion.auth = OAuth1(*credenciales)

    def llamar(metodo, endpoint, **kwargs):
        respuesta = sesion.request(
            metodo,
            "https://api.x.com/2" + endpoint,
            timeout=(20, 180),
            **kwargs,
        )
        if not respuesta.ok:
            raise RuntimeError(
                f"X HTTP {respuesta.status_code}: "
                f"{respuesta.reason}"
            )
        return respuesta.json() if respuesta.content else {}

    inicio = llamar(
        "POST",
        "/media/upload/initialize",
        json={
            "media_type": "video/mp4",
            "total_bytes": video.stat().st_size,
            "media_category": "tweet_video",
        },
    )
    media_id = inicio["data"]["id"]

    with video.open("rb") as archivo:
        segmento = 0
        while True:
            contenido = archivo.read(4 * 1024 * 1024)
            if not contenido:
                break

            llamar(
                "POST",
                f"/media/upload/{media_id}/append",
                data={"segment_index": str(segmento)},
                files={
                    "media": (
                        "chunk.mp4",
                        contenido,
                        "application/octet-stream",
                    )
                },
            )
            segmento += 1

    estado = llamar(
        "POST",
        f"/media/upload/{media_id}/finalize",
    ).get("data", {})

    limite = time.monotonic() + 900

    while True:
        informacion = estado.get("processing_info", {})
        situacion = informacion.get("state")

        if situacion in (None, "succeeded"):
            break

        if situacion == "failed":
            raise RuntimeError(
                "X rechazó el procesamiento: "
                + str(informacion.get("error", ""))
            )

        if time.monotonic() > limite:
            raise RuntimeError("Tiempo de procesamiento X agotado.")

        time.sleep(
            min(60, max(1, informacion.get("check_after_secs", 5)))
        )

        estado = llamar(
            "GET",
            "/media/upload",
            params={
                "command": "STATUS",
                "media_id": media_id,
            },
        ).get("data", {})

    fecha = pd.to_datetime(
        evento["properties"]["time"], unit="ms", utc=True
    )
    lon, lat, _ = evento["geometry"]["coordinates"]
    coordenadas = (
        f"{abs(lat):.3f}°{'N' if lat >= 0 else 'S'}, "
        f"{abs(lon):.3f}°{'E' if lon >= 0 else 'W'}"
    )
    lugar = (
        evento["properties"].get("place") or "Global earthquake"
    )

    texto = (
        f"M{evento['properties']['mag']:.1f} · {lugar[:50]}\n"
        f"{fecha:%Y-%m-%d %H:%M UTC}\n"
        f"Epicenter: {coordenadas}\n"
        "30-day hourly ACP evolution.\n"
        "Exploratory, not a seismic prediction.\n"
        f"https://earthquake.usgs.gov/earthquakes/eventpage/{evento['id']}"
    )

    (salida / "post_text.txt").write_text(
        texto, encoding="utf-8"
    )

    # Registrar antes del POST evita duplicados si se pierde
    # la respuesta de X o se interrumpe el runner.
    guardar_json(archivo_estado, {
        "status": "posting",
        "event_id": evento["id"],
        "media_id": media_id,
    })
    guardar_estado_git()

    resultado = llamar(
        "POST",
        "/tweets",
        json={
            "text": texto,
            "media": {"media_ids": [media_id]},
        },
    )

    enlace = (
        "https://x.com/i/web/status/" + resultado["data"]["id"]
    )

    guardar_json(archivo_estado, {
        "status": "published",
        "event_id": evento["id"],
        "post_url": enlace,
        "media_id": media_id,
    })
    guardar_estado_git()

    print("Publicado:", enlace)


# ================================================================
# PROCESAMIENTO DE CADA SISMO
# ================================================================

def procesar_evento(evento):
    fecha = pd.to_datetime(
        evento["properties"]["time"], unit="ms", utc=True
    )
    lon, lat, _ = evento["geometry"]["coordinates"]

    salida = ROOT / evento["id"]
    salida.mkdir(parents=True, exist_ok=True)
    guardar_json(salida / "event.json", evento)

    print(
        "Procesando:",
        evento["id"],
        "M", evento["properties"]["mag"],
        fecha,
        evento["properties"].get("place", ""),
    )

    latitudes = np.arange(
        math.ceil(max(-90, lat - RADIO_LATITUD) / PASO),
        math.floor(min(90, lat + RADIO_LATITUD) / PASO) + 1,
    ) * PASO

    longitudes = np.arange(
        math.ceil((lon - RADIO_LONGITUD) / PASO),
        math.floor((lon + RADIO_LONGITUD) / PASO) + 1,
    ) * PASO

    fechas = pd.date_range(
        (fecha - pd.Timedelta(days=DIAS_PREVIOS)).ceil("h"),
        fecha,
        freq="h",
        inclusive="left",
    )
    assert len(fechas) == DIAS_PREVIOS * 24

    cubo = np.full(
        (len(fechas), len(latitudes), len(longitudes)),
        np.nan,
        dtype=np.float32,
    )

    faltantes = []

    for i, hora in enumerate(
        tqdm(fechas, desc="Descargando ACP")
    ):
        try:
            cubo[i] = descargar_acp(
                hora, latitudes, longitudes
            )
        except FileNotFoundError:
            faltantes.append(hora.isoformat())

    guardar_json(salida / "missing_times.json", faltantes)

    if not np.isfinite(cubo).any():
        raise RuntimeError("No se recuperaron datos NOAA válidos.")

    cobertura = float(np.isfinite(cubo).mean())

    video, cobertura_serie = construir_video(
        evento, fechas, cubo, latitudes, longitudes, salida
    )

    guardar_json(salida / "summary.json", {
        "event_id": evento["id"],
        "duration_seconds": len(fechas) / FPS,
        "scenes_per_second": FPS,
        "map_coverage": cobertura,
        "chart_coverage": cobertura_serie,
        "missing_hours": len(faltantes),
        "source": "NOAA GFS analyses and 1–5 hour forecasts",
    })

    print(
        "Video:", video,
        "| duración:", len(fechas) / FPS,
        "| cobertura mapa:", f"{cobertura:.1%}",
        "| cobertura gráfico:", f"{cobertura_serie:.1%}",
    )

    if PUBLICAR_EN_X:
        if min(cobertura, cobertura_serie) < COBERTURA_MINIMA:
            raise RuntimeError(
                "Cobertura insuficiente para publicar. "
                "Se reintentará en otra ejecución."
            )
        publicar(video, evento, salida)
    else:
        print("Modo prueba: no se publica en X.")


# ================================================================
# EJECUCIÓN
# ================================================================

def main():
    pendientes = actualizar_cola()

    print(
        f"{len(pendientes)} eventos pendientes; "
        f"máximo {MAX_EVENTOS} por ejecución."
    )

    fallidos = []

    for evento in pendientes[:MAX_EVENTOS]:
        try:
            procesar_evento(evento)

        except Exception as error:
            print("Evento fallido:", evento["id"], str(error))
            fallidos.append(evento["id"])

            archivo = ruta_estado(evento)
            anterior = leer_json(archivo)

            # No sobrescribir estados de publicación incierta.
            if (
                PUBLICAR_EN_X
                and anterior.get("status")
                not in ("posting", "published")
            ):
                guardar_json(archivo, {
                    "status": "failed",
                    "attempts": anterior.get("attempts", 0) + 1,
                })
                guardar_estado_git()

    if fallidos:
        raise RuntimeError(
            "Eventos incompletos: " + ", ".join(fallidos)
        )


if __name__ == "__main__":
    main()
