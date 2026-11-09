"""Interfaz de usuario en Streamlit para Tribuna Global.

Permite seleccionar un futbolista del catálogo, emitir opiniones analizadas con
LangChain (salida estructurada Pydantic) y consultar/descargar los registros
persistidos en disco sin invocar nuevamente al modelo.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from almacenamiento import (
    ArchivoOpinionesCorruptoError,
    ErrorCatalogoJugadores,
    RUTA_JUGADORES,
    RUTA_OPINIONES,
    cargar_jugadores,
    cargar_opiniones,
    consultar_opiniones,
    crear_registro,
    guardar_opinion,
)
from analisis import (
    ErrorClasificacionIA,
    ErrorValidacionEntrada,
    VERSION_PROMPT,
    analizar_opinion,
    obtener_nombre_modelo,
    validar_texto_opinion,
)

ETIQUETAS_SENTIMIENTO = {
    "positivo": "🟢 Positivo",
    "negativo": "🔴 Negativo",
    "neutro": "⚪ Neutro",
    "mixto": "🟡 Mixto",
}


def inicializar_estado_sesion() -> None:
    """Inicializa las claves necesarias en st.session_state."""
    if "historial_chat" not in st.session_state:
        st.session_state["historial_chat"] = []
    if "ultimo_error" not in st.session_state:
        st.session_state["ultimo_error"] = None
    if "opinion_recuperable" not in st.session_state:
        st.session_state["opinion_recuperable"] = None


def limpiar_conversacion() -> None:
    """Limpia únicamente el historial visual de la sesión y conserva el archivo en disco."""
    st.session_state["historial_chat"] = []
    st.session_state["ultimo_error"] = None
    st.session_state["opinion_recuperable"] = None


def procesar_envio_opinion(jugador_capturado: dict[str, str], texto_entrada: str) -> None:
    """Valida, clasifica y persiste una opinión asociándola al jugador capturado al enviar."""
    st.session_state["ultimo_error"] = None
    st.session_state["opinion_recuperable"] = None

    # 1. Validación de entrada (sin llamar al modelo si falla)
    try:
        texto_limpio = validar_texto_opinion(texto_entrada)
    except ErrorValidacionEntrada as exc:
        st.session_state["ultimo_error"] = {
            "tipo": "entrada",
            "mensaje": str(exc),
        }
        st.session_state["opinion_recuperable"] = {
            "jugador": jugador_capturado,
            "texto": texto_entrada,
            "analisis": None,
            "registro": None,
        }
        return

    # 2. Clasificación con el modelo estructurado (solo considera la opinión actual)
    try:
        analisis = analizar_opinion(jugador_capturado, texto_limpio)
    except (ErrorClasificacionIA, ErrorValidacionEntrada) as exc:
        st.session_state["ultimo_error"] = {
            "tipo": "clasificacion",
            "mensaje": str(exc),
        }
        st.session_state["opinion_recuperable"] = {
            "jugador": jugador_capturado,
            "texto": texto_limpio,
            "analisis": None,
            "registro": None,
        }
        return
    except Exception as exc:
        st.session_state["ultimo_error"] = {
            "tipo": "clasificacion",
            "mensaje": f"Fallo inesperado al clasificar la opinión: {exc}",
        }
        st.session_state["opinion_recuperable"] = {
            "jugador": jugador_capturado,
            "texto": texto_limpio,
            "analisis": None,
            "registro": None,
        }
        return

    # 3. Verificar pertinencia (si no es pertinente, se informa pero NO se guarda)
    if not analisis.es_pertinente:
        st.session_state["historial_chat"].append(
            {
                "jugador_id": jugador_capturado["id"],
                "jugador_nombre": jugador_capturado["nombre"],
                "jugador_pais": jugador_capturado["pais"],
                "opinion": texto_limpio,
                "es_pertinente": False,
                "sentimiento": None,
                "razon": analisis.razon,
                "guardado": False,
                "registro_id": None,
            }
        )
        return

    # 4. Creación del registro y escritura atómica en disco
    modelo_usado = obtener_nombre_modelo()
    try:
        registro = crear_registro(
            jugador=jugador_capturado,
            texto=texto_limpio,
            analisis=analisis,
            modelo=modelo_usado,
            version_prompt=VERSION_PROMPT,
        )
        guardar_opinion(RUTA_OPINIONES, registro)
    except ArchivoOpinionesCorruptoError as exc:
        st.session_state["ultimo_error"] = {
            "tipo": "almacenamiento",
            "mensaje": (
                "La opinión fue clasificada, pero NO pudo guardarse porque el archivo de opiniones "
                f"presenta un daño de formato: {exc}"
            ),
        }
        st.session_state["opinion_recuperable"] = {
            "jugador": jugador_capturado,
            "texto": texto_limpio,
            "analisis": analisis,
            "registro": None,
        }
        return
    except Exception as exc:
        st.session_state["ultimo_error"] = {
            "tipo": "almacenamiento",
            "mensaje": (
                "La opinión fue analizada, pero ocurrió un error al guardar en disco. "
                f"No se registró el envío: {exc}"
            ),
        }
        st.session_state["opinion_recuperable"] = {
            "jugador": jugador_capturado,
            "texto": texto_limpio,
            "analisis": analisis,
            "registro": None,
        }
        return

    # 5. Solo tras una escritura exitosa se agrega confirmación de guardado al historial
    st.session_state["historial_chat"].append(
        {
            "jugador_id": jugador_capturado["id"],
            "jugador_nombre": jugador_capturado["nombre"],
            "jugador_pais": jugador_capturado["pais"],
            "opinion": texto_limpio,
            "es_pertinente": True,
            "sentimiento": analisis.sentimiento,
            "razon": analisis.razon,
            "guardado": True,
            "registro_id": registro["id"],
            "fecha_utc": registro["fecha_utc"],
        }
    )


def renderizar_seccion_emitir(jugador_seleccionado: dict[str, str]) -> None:
    """Renderiza el chat de emisión de opiniones y su historial de sesión."""
    st.subheader("💬 Emitir opinión")
    st.caption(
        f"Futbolista seleccionado actualmente: **{jugador_seleccionado['nombre']} ({jugador_seleccionado['pais']})** "
        f"— Código interno: `{jugador_seleccionado['id']}`. "
        "Cada envío se evalúa de manera independiente sin mezclar mensajes anteriores."
    )

    # Mostrar historial de la sesión actual
    if not st.session_state["historial_chat"]:
        st.info(
            "Aún no has enviado opiniones en esta sesión. Selecciona un futbolista en la barra lateral "
            "y escribe tu comentario abajo (entre 5 y 1000 caracteres)."
        )

    for item in st.session_state["historial_chat"]:
        with st.chat_message("user"):
            st.markdown(
                f"**[{item['jugador_nombre']} ({item['jugador_pais']})]**\n\n{item['opinion']}"
            )
        with st.chat_message("assistant"):
            if not item["es_pertinente"]:
                st.warning(
                    f"**Mensaje no pertinente para {item['jugador_nombre']}**\n\n"
                    f"**Justificación:** {item['razon']}\n\n"
                    "ℹ️ *Este comentario no fue guardado en el registro de opiniones.*"
                )
            else:
                etiqueta = ETIQUETAS_SENTIMIENTO.get(
                    item["sentimiento"], str(item["sentimiento"]).capitalize()
                )
                st.markdown(
                    f"**Jugador evaluado:** {item['jugador_nombre']} (`{item['jugador_id']}`)\n\n"
                    f"**Sentimiento estimado:** {etiqueta}\n\n"
                    f"**Justificación:** {item['razon']}"
                )
                if item.get("guardado"):
                    st.success(
                        f"✅ Opinión guardada exitosamente en `opiniones.json` (ID: `{item['registro_id']}`)."
                    )

    # Mostrar bloque de error y conservación de la opinión en pantalla si el último intento falló
    error_actual: dict[str, Any] | None = st.session_state.get("ultimo_error")
    recuperable: dict[str, Any] | None = st.session_state.get("opinion_recuperable")

    if error_actual:
        tipo_err = error_actual.get("tipo", "error")
        titulo_err = {
            "entrada": "⚠️ Error de validación de entrada",
            "clasificacion": "❌ Error de clasificación (API / Modelo)",
            "almacenamiento": "❌ Error de escritura en almacenamiento",
        }.get(tipo_err, "❌ Error")

        st.error(f"**{titulo_err}:** {error_actual.get('mensaje')}\n\n*El registro no fue guardado.*")

        if recuperable and recuperable.get("texto"):
            st.markdown("**Opinión conservada en pantalla para su recuperación:**")
            st.code(recuperable["texto"], language="text")
            jugador_rec = recuperable.get("jugador", jugador_seleccionado)
            if tipo_err in {"clasificacion", "almacenamiento"}:
                if st.button("🔄 Reintentar envío de esta opinión"):
                    procesar_envio_opinion(jugador_rec, recuperable["texto"])
                    st.rerun()

    # Entrada de chat
    texto_usuario = st.chat_input(
        f"Escribe tu opinión sobre {jugador_seleccionado['nombre']} (5 a 1000 caracteres)..."
    )
    if texto_usuario is not None:
        # Capturamos una copia inmutable del jugador seleccionado en el instante del envío
        jugador_capturado = dict(jugador_seleccionado)
        procesar_envio_opinion(jugador_capturado, texto_usuario)
        st.rerun()


def renderizar_seccion_consultar(
    catalogo_jugadores: list[dict[str, str]],
    mapa_jugadores: dict[str, dict[str, str]],
) -> None:
    """Renderiza la consulta filtrada, recuentos y descarga JSON sin llamar al modelo."""
    st.subheader("🔎 Consultar opiniones registradas")
    st.caption(
        "Los datos mostrados provienen exclusivamente del archivo persistente `datos/opiniones.json`. "
        "Esta sección no realiza llamadas al modelo de lenguaje."
    )

    try:
        datos_persistidos = cargar_opiniones(RUTA_OPINIONES)
    except ArchivoOpinionesCorruptoError as exc:
        st.error(
            f"**Archivo de opiniones dañado:** {exc}\n\n"
            "El archivo original se ha conservado en disco sin sobrescribirse para evitar pérdida de datos."
        )
        return
    except Exception as exc:
        st.error(f"No fue posible leer el archivo de opiniones: {exc}")
        return

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        opciones_jugador = ["todos"] + [j["id"] for j in catalogo_jugadores]

        def formatear_filtro_jugador(jid: str) -> str:
            if jid == "todos":
                return "Todos los jugadores"
            info = mapa_jugadores.get(jid)
            return f"{info['nombre']} ({info['pais']})" if info else jid

        filtro_jugador = st.selectbox(
            "Filtrar por jugador",
            options=opciones_jugador,
            format_func=formatear_filtro_jugador,
            key="filtro_consulta_jugador",
        )

    with col_f2:
        opciones_sentimiento = ["todos", "positivo", "negativo", "neutro", "mixto"]
        etiquetas_filtro_sent = {
            "todos": "Todos los sentimientos",
            "positivo": "Positivo",
            "negativo": "Negativo",
            "neutro": "Neutro",
            "mixto": "Mixto",
        }
        filtro_sentimiento = st.selectbox(
            "Filtrar por sentimiento",
            options=opciones_sentimiento,
            format_func=lambda s: etiquetas_filtro_sent.get(s, s),
            key="filtro_consulta_sentimiento",
        )

    registros_filtrados = consultar_opiniones(
        datos_persistidos,
        jugador_id=filtro_jugador,
        sentimiento=filtro_sentimiento,
    )

    # Recuentos usando exactamente los mismos filtros que la tabla
    conteo_por_sentimiento = {
        "positivo": 0,
        "negativo": 0,
        "neutro": 0,
        "mixto": 0,
    }
    for reg in registros_filtrados:
        sent = reg.get("sentimiento")
        if sent in conteo_por_sentimiento:
            conteo_por_sentimiento[sent] += 1

    m_total, m_pos, m_neg, m_neu, m_mix = st.columns(5)
    m_total.metric("Total filtrados", len(registros_filtrados))
    m_pos.metric("🟢 Positivo", conteo_por_sentimiento["positivo"])
    m_neg.metric("🔴 Negativo", conteo_por_sentimiento["negativo"])
    m_neu.metric("⚪ Neutro", conteo_por_sentimiento["neutro"])
    m_mix.metric("🟡 Mixto", conteo_por_sentimiento["mixto"])

    if not registros_filtrados:
        st.info("No se encontraron opiniones registradas para la combinación de filtros seleccionada.")
    else:
        filas_tabla = [
            {
                "Fecha (UTC)": reg.get("fecha_utc", ""),
                "Jugador": reg.get("jugador_nombre", ""),
                "Opinión original": reg.get("opinion_original", ""),
                "Sentimiento": reg.get("sentimiento", ""),
                "Razón": reg.get("razon", ""),
            }
            for reg in registros_filtrados
        ]
        st.dataframe(filas_tabla, width="stretch", hide_index=True)

    # Descarga de los resultados filtrados como JSON
    json_descarga = json.dumps(
        {
            "version": 1,
            "filtros_aplicados": {
                "jugador_id": filtro_jugador,
                "sentimiento": filtro_sentimiento,
            },
            "total": len(registros_filtrados),
            "opiniones": registros_filtrados,
        },
        ensure_ascii=False,
        indent=2,
    )
    st.download_button(
        label="⬇️ Descargar resultados filtrados (JSON)",
        data=json_descarga.encode("utf-8"),
        file_name="opiniones_filtradas.json",
        mime="application/json",
    )


def main() -> None:
    """Punto de entrada principal de la aplicación Streamlit."""
    st.set_page_config(
        page_title="Tribuna Global | Percepción de Futbolistas",
        page_icon="⚽",
        layout="wide",
    )
    inicializar_estado_sesion()

    st.title("⚽ Tribuna Global")
    st.markdown(
        "Sistema de registro y análisis de percepción de aficionados sobre futbolistas "
        "mediante **LangChain** y salida estructurada."
    )

    try:
        catalogo_jugadores = cargar_jugadores(RUTA_JUGADORES)
    except ErrorCatalogoJugadores as exc:
        st.error(f"**Error al cargar el catálogo de jugadores:** {exc}")
        st.stop()

    mapa_jugadores = {j["id"]: j for j in catalogo_jugadores}
    ids_jugadores = [j["id"] for j in catalogo_jugadores]

    with st.sidebar:
        st.header("⚙️ Panel de Control")

        seccion_activa = st.radio(
            "Sección",
            options=["Emitir opinión", "Consultar opiniones"],
            index=0,
        )

        st.divider()
        st.subheader("🏃 Selección de Futbolista")
        jugador_id_activo = st.selectbox(
            "Futbolista a evaluar",
            options=ids_jugadores,
            format_func=lambda jid: f"{mapa_jugadores[jid]['nombre']} ({mapa_jugadores[jid]['pais']})",
            key="selector_jugador_sidebar",
        )
        jugador_seleccionado = mapa_jugadores[jugador_id_activo]

        st.divider()
        if st.button("🧹 Nueva conversación", width="stretch"):
            limpiar_conversacion()
            st.rerun()
        st.caption(
            "«Nueva conversación» limpia el chat de la sesión actual y conserva todas las "
            "opiniones guardadas en `datos/opiniones.json`."
        )

    if seccion_activa == "Emitir opinión":
        renderizar_seccion_emitir(jugador_seleccionado)
    else:
        renderizar_seccion_consultar(catalogo_jugadores, mapa_jugadores)


if __name__ == "__main__":
    main()
