"""Módulo de almacenamiento y consulta en JSON para Tribuna Global.

Gestiona la lectura y validación del catálogo de jugadores, la persistencia
atómica de opiniones en disco y la consulta filtrada en memoria sin invocar al modelo.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
RUTA_JUGADORES = DATOS_DIR / "jugadores.json"
RUTA_OPINIONES = DATOS_DIR / "opiniones.json"

SENTIMIENTOS_VALIDOS = {"positivo", "negativo", "neutro", "mixto"}
CAMPOS_REGISTRO_REQUERIDOS = {
    "id",
    "fecha_utc",
    "jugador_id",
    "jugador_nombre",
    "opinion_original",
    "sentimiento",
    "razon",
    "modelo",
    "version_prompt",
}


class ErrorCatalogoJugadores(ValueError):
    """Error al cargar o validar el catálogo de futbolistas."""


class ArchivoOpinionesCorruptoError(ValueError):
    """Error cuando el archivo de opiniones existe pero su JSON o esquema es inválido."""


def cargar_jugadores(ruta: str | Path = RUTA_JUGADORES) -> list[dict[str, str]]:
    """Lee y valida el catálogo de jugadores desde un archivo JSON.

    Comprueba que el esquema sea válido y que no existan identificadores repetidos.
    """
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        raise ErrorCatalogoJugadores(
            f"No se encontró el catálogo de jugadores en: {ruta_path}"
        )

    try:
        contenido = ruta_path.read_text(encoding="utf-8")
        datos = json.loads(contenido)
    except json.JSONDecodeError as exc:
        raise ErrorCatalogoJugadores(
            f"El archivo de jugadores contiene JSON inválido ({ruta_path.name}): {exc}"
        ) from exc
    except OSError as exc:
        raise ErrorCatalogoJugadores(
            f"No se pudo leer el catálogo de jugadores ({ruta_path.name}): {exc}"
        ) from exc

    if not isinstance(datos, dict) or datos.get("version") != 1:
        raise ErrorCatalogoJugadores(
            "El catálogo de jugadores debe ser un objeto JSON con 'version': 1."
        )

    jugadores = datos.get("jugadores")
    if not isinstance(jugadores, list) or not jugadores:
        raise ErrorCatalogoJugadores(
            "El catálogo debe incluir una lista no vacía en la clave 'jugadores'."
        )

    ids_vistos: set[str] = set()
    catalogo_validado: list[dict[str, str]] = []

    for indice, item in enumerate(jugadores):
        if not isinstance(item, dict):
            raise ErrorCatalogoJugadores(
                f"El elemento en la posición {indice} del catálogo no es un objeto válido."
            )

        jugador_id = str(item.get("id", "")).strip()
        nombre = str(item.get("nombre", "")).strip()
        pais = str(item.get("pais", "")).strip()

        if not jugador_id or not nombre or not pais:
            raise ErrorCatalogoJugadores(
                f"El jugador en la posición {indice} tiene campos obligatorios vacíos (id, nombre, pais)."
            )

        if jugador_id in ids_vistos:
            raise ErrorCatalogoJugadores(
                f"Identificador de jugador duplicado en el catálogo: '{jugador_id}'."
            )

        ids_vistos.add(jugador_id)
        catalogo_validado.append(
            {
                "id": jugador_id,
                "nombre": nombre,
                "pais": pais,
                "apodo": str(item.get("apodo", "")),
                "posicion": str(item.get("posicion", "Futbolista")),
                "foto_url": str(item.get("foto_url", "")),
                "partidos": int(item.get("partidos", 0)),
                "goles": int(item.get("goles", 0)),
                "asistencias": int(item.get("asistencias", 0)),
                "titulos": int(item.get("titulos", 0)),
                "balones_de_oro": int(item.get("balones_de_oro", 0)),
                "botas_de_oro": int(item.get("botas_de_oro", 0)),
                "trayectoria": list(item.get("trayectoria", [])),
                "logros": list(item.get("logros", [])),
            }
        )

    return catalogo_validado


def _validar_esquema_opiniones(datos: Any, ruta_path: Path) -> dict[str, Any]:
    """Verifica que el diccionario de opiniones cumpla con el esquema esperado."""
    if not isinstance(datos, dict) or datos.get("version") != 1:
        raise ArchivoOpinionesCorruptoError(
            f"Esquema inesperado en '{ruta_path.name}': se esperaba un objeto con 'version': 1."
        )

    opiniones = datos.get("opiniones")
    if not isinstance(opiniones, list):
        raise ArchivoOpinionesCorruptoError(
            f"Esquema inesperado en '{ruta_path.name}': la clave 'opiniones' debe ser una lista."
        )

    for indice, reg in enumerate(opiniones):
        if not isinstance(reg, dict):
            raise ArchivoOpinionesCorruptoError(
                f"El registro #{indice} en '{ruta_path.name}' no es un objeto JSON válido."
            )
        faltantes = CAMPOS_REGISTRO_REQUERIDOS - set(reg.keys())
        if faltantes:
            raise ArchivoOpinionesCorruptoError(
                f"El registro #{indice} en '{ruta_path.name}' no contiene los campos requeridos: {sorted(faltantes)}."
            )
        if reg.get("sentimiento") not in SENTIMIENTOS_VALIDOS:
            raise ArchivoOpinionesCorruptoError(
                f"El registro #{indice} en '{ruta_path.name}' tiene un sentimiento inválido: {reg.get('sentimiento')!r}."
            )

    return datos


def cargar_opiniones(ruta: str | Path = RUTA_OPINIONES) -> dict[str, Any]:
    """Lee el diccionario persistente de opiniones.

    Si el archivo no existe, inicializa la estructura vacía.
    Si el archivo existe pero contiene JSON inválido o un esquema inesperado,
    lanza ArchivoOpinionesCorruptoError y conserva el archivo intacto en disco.
    """
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        estructura_inicial: dict[str, Any] = {"version": 1, "opiniones": []}
        try:
            ruta_path.parent.mkdir(parents=True, exist_ok=True)
            _escribir_json_atomico(ruta_path, estructura_inicial)
        except OSError:
            # Si no se puede escribir en el momento de lectura inicial, devolvemos la estructura en memoria
            pass
        return estructura_inicial

    try:
        contenido = ruta_path.read_text(encoding="utf-8")
        datos = json.loads(contenido)
    except json.JSONDecodeError as exc:
        raise ArchivoOpinionesCorruptoError(
            f"El archivo '{ruta_path.name}' contiene JSON inválido y fue conservado sin cambios para su revisión: {exc}"
        ) from exc

    return _validar_esquema_opiniones(datos, ruta_path)


def crear_registro(
    jugador: dict[str, str],
    texto: str,
    analisis: Any,
    modelo: str,
    version_prompt: str = "1.0",
    registro_id: str | None = None,
) -> dict[str, str]:
    """Incorpora UUID, fecha UTC e información del jugador al resultado del análisis.

    Valida que el análisis sea pertinente y posea un sentimiento válido antes de crear el registro.
    """
    if not isinstance(jugador, dict) or not jugador.get("id") or not jugador.get("nombre"):
        raise ValueError("Datos del jugador incompletos para crear el registro.")

    texto_limpio = texto.strip() if isinstance(texto, str) else ""
    if len(texto_limpio) < 5 or len(texto_limpio) > 1000:
        raise ValueError(
            "La opinión debe tener entre 5 y 1000 caracteres después de eliminar espacios exteriores."
        )

    es_pertinente = getattr(analisis, "es_pertinente", None)
    sentimiento = getattr(analisis, "sentimiento", None)
    razon = getattr(analisis, "razon", None)

    if es_pertinente is not True:
        raise ValueError("No se puede crear un registro persistente para una opinión no pertinente.")

    if sentimiento not in SENTIMIENTOS_VALIDOS:
        raise ValueError(
            f"Sentimiento inválido para registro persistente: {sentimiento!r}."
        )

    if not isinstance(razon, str) or not razon.strip():
        raise ValueError("El análisis debe incluir una justificación ('razon') no vacía.")

    return {
        "id": str(registro_id or uuid.uuid4()),
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jugador_id": str(jugador["id"]).strip(),
        "jugador_nombre": str(jugador["nombre"]).strip(),
        "opinion_original": texto_limpio,
        "sentimiento": sentimiento,
        "razon": razon.strip(),
        "modelo": str(modelo).strip() or "modelo_desconocido",
        "version_prompt": str(version_prompt).strip() or "1.0",
    }


def _escribir_json_atomico(ruta_path: Path, datos: dict[str, Any]) -> None:
    """Escribe un diccionario como JSON usando un archivo temporal en el mismo directorio y reemplazo atómico."""
    ruta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=ruta_path.parent,
            prefix=f".{ruta_path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as archivo_tmp:
            tmp_path = Path(archivo_tmp.name)
            json.dump(datos, archivo_tmp, ensure_ascii=False, indent=2)
            archivo_tmp.write("\n")
            archivo_tmp.flush()
            os.fsync(archivo_tmp.fileno())

        os.replace(tmp_path, ruta_path)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def guardar_opinion(ruta: str | Path, registro: dict[str, Any]) -> bool:
    """Añade un registro al archivo de opiniones sin perder los anteriores.

    Utiliza escritura atómica mediante archivo temporal y reemplazo.
    Evita duplicar un registro si su 'id' ya se encuentra almacenado (reintentos).
    Retorna True si se agregó el registro o False si el UUID ya existía.
    """
    ruta_path = Path(ruta)

    if not isinstance(registro, dict):
        raise ValueError("El registro a guardar debe ser un diccionario.")

    faltantes = CAMPOS_REGISTRO_REQUERIDOS - set(registro.keys())
    if faltantes:
        raise ValueError(
            f"El registro no contiene todos los campos requeridos. Faltan: {sorted(faltantes)}"
        )

    if registro.get("sentimiento") not in SENTIMIENTOS_VALIDOS:
        raise ValueError(
            f"No se puede guardar un registro con sentimiento inválido: {registro.get('sentimiento')!r}"
        )

    datos = cargar_opiniones(ruta_path)
    opiniones: list[dict[str, Any]] = datos["opiniones"]

    # Evitar duplicación del mismo envío durante un reintento por su UUID
    for existente in opiniones:
        if existente.get("id") == registro["id"]:
            return False

    opiniones.append(registro)
    _escribir_json_atomico(ruta_path, datos)
    return True


def consultar_opiniones(
    registros: list[dict[str, Any]] | dict[str, Any],
    jugador_id: str | None = None,
    sentimiento: str | None = None,
) -> list[dict[str, Any]]:
    """Filtra registros por jugador_id y/o sentimiento sin consultar al modelo.

    Acepta tanto la lista de opiniones como el diccionario completo {'version': 1, 'opiniones': [...]}.
    """
    if isinstance(registros, dict):
        lista_registros = registros.get("opiniones", [])
    elif isinstance(registros, list):
        lista_registros = registros
    else:
        raise ValueError("Se esperaba una lista de registros o el diccionario de opiniones.")

    filtro_jugador = (jugador_id or "").strip()
    if filtro_jugador.lower() in {"", "todos", "all"}:
        filtro_jugador = ""

    filtro_sentimiento = (sentimiento or "").strip().lower()
    if filtro_sentimiento in {"", "todos", "all"}:
        filtro_sentimiento = ""

    resultado: list[dict[str, Any]] = []
    for reg in lista_registros:
        if filtro_jugador and reg.get("jugador_id") != filtro_jugador:
            continue
        if filtro_sentimiento and str(reg.get("sentimiento", "")).lower() != filtro_sentimiento:
            continue
        resultado.append(reg)

    return resultado

