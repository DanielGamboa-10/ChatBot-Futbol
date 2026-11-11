"""Interfaz de usuario en Streamlit para Tribuna Global.

Permite seleccionar un futbolista del catalogo, emitir opiniones analizadas con
LangChain (salida estructurada Pydantic), consultar registros persistidos,
ver fichas tecnicas con fotos y trayectoria, comparar carreras y explorar
las 5 grandes ligas europeas y competiciones UEFA.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

ASSETS_DIR = BASE_DIR / "assets"

from almacenamiento import (
    ArchivoOpinionesCorruptoError,
    ErrorCatalogoJugadores,
    RUTA_COMPETICIONES,
    RUTA_JUGADORES,
    RUTA_OPINIONES,
    cargar_competiciones,
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
    "positivo": "Positivo",
    "negativo": "Negativo",
    "neutro": "Neutro",
    "mixto": "Mixto",
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
    """Limpia unicamente el historial visual de la sesion y conserva el archivo en disco."""
    st.session_state["historial_chat"] = []
    st.session_state["ultimo_error"] = None
    st.session_state["opinion_recuperable"] = None


def procesar_envio_opinion(jugador_capturado: dict[str, Any], texto_entrada: str) -> None:
    """Valida, clasifica y persiste una opinion asociandola al jugador capturado al enviar."""
    st.session_state["ultimo_error"] = None
    st.session_state["opinion_recuperable"] = None

    # 1. Validacion de entrada
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

    # 2. Clasificacion con el modelo estructurado
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
            "mensaje": f"Fallo inesperado al clasificar la opinion: {exc}",
        }
        st.session_state["opinion_recuperable"] = {
            "jugador": jugador_capturado,
            "texto": texto_limpio,
            "analisis": None,
            "registro": None,
        }
        return

    # 3. Verificar pertinencia
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

    # 4. Creacion del registro y escritura atomica en disco
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
                "La opinion fue clasificada, pero NO pudo guardarse porque el archivo de opiniones "
                f"presenta un dano de formato: {exc}"
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
                "La opinion fue analizada, pero ocurrio un error al guardar en disco. "
                f"No se registro el envio: {exc}"
            ),
        }
        st.session_state["opinion_recuperable"] = {
            "jugador": jugador_capturado,
            "texto": texto_limpio,
            "analisis": analisis,
            "registro": None,
        }
        return

    # 5. Confirmacion de guardado en el historial tras escritura exitosa
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


def renderizar_tarjeta_jugador(jugador: dict[str, Any]) -> None:
    """Muestra tarjeta con foto oficial, apodo, posicion y metricas reales."""
    col_img, col_datos = st.columns([1, 3])
    with col_img:
        foto = jugador.get("foto_url")
        if foto:
            st.image(foto, width=170, caption=jugador.get("nombre"))
        else:
            st.info("Foto no disponible")

    with col_datos:
        apodo = f" *«{jugador['apodo']}»*" if jugador.get("apodo") else ""
        st.markdown(f"### {jugador['nombre']}{apodo}")
        st.markdown(f"Pais: {jugador.get('pais')} | Posicion: {jugador.get('posicion', 'Futbolista')}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Partidos", jugador.get("partidos", 0))
        m2.metric("Goles", jugador.get("goles", 0))
        m3.metric("Asistencias", jugador.get("asistencias", 0))
        m4.metric("Titulos", jugador.get("titulos", 0))
        b_oro = jugador.get("balones_de_oro", 0)
        m5.metric("Balones de Oro", b_oro if b_oro > 0 else "-")


def renderizar_seccion_emitir(jugador_seleccionado: dict[str, Any]) -> None:
    """Renderiza el chat de emision de opiniones, tarjeta del jugador y su historial de sesion."""
    renderizar_tarjeta_jugador(jugador_seleccionado)
    st.divider()

    st.subheader("Emitir opinion")
    st.caption(
        f"Evaluando actualmente a: **{jugador_seleccionado['nombre']} ({jugador_seleccionado['pais']})** "
        f"- Codigo: `{jugador_seleccionado['id']}`. Cada opinion se analiza de forma independiente."
    )

    if not st.session_state["historial_chat"]:
        st.info(
            "Escribe abajo tu opinion o valoracion sobre este futbolista (entre 5 y 1000 caracteres)."
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
                    f"**Justificacion:** {item['razon']}\n\n"
                    "*Este comentario no fue guardado en el registro de opiniones.*"
                )
            else:
                etiqueta = ETIQUETAS_SENTIMIENTO.get(
                    item["sentimiento"], str(item["sentimiento"]).capitalize()
                )
                st.markdown(
                    f"**Jugador evaluado:** {item['jugador_nombre']} (`{item['jugador_id']}`)\n\n"
                    f"**Sentimiento estimado:** {etiqueta}\n\n"
                    f"**Justificacion:** {item['razon']}"
                )
                if item.get("guardado"):
                    st.success(
                        f"Opinion guardada exitosamente en opiniones.json (ID: `{item['registro_id']}`)."
                    )

    error_actual: dict[str, Any] | None = st.session_state.get("ultimo_error")
    recuperable: dict[str, Any] | None = st.session_state.get("opinion_recuperable")

    if error_actual:
        tipo_err = error_actual.get("tipo", "error")
        titulo_err = {
            "entrada": "Error de validacion de entrada",
            "clasificacion": "Error de clasificacion (API / Modelo)",
            "almacenamiento": "Error de escritura en almacenamiento",
        }.get(tipo_err, "Error")

        st.error(f"**{titulo_err}:** {error_actual.get('mensaje')}\n\n*El registro no fue guardado.*")

        if recuperable and recuperable.get("texto"):
            st.markdown("**Opinion conservada en pantalla para su recuperacion:**")
            st.code(recuperable["texto"], language="text")
            jugador_rec = recuperable.get("jugador", jugador_seleccionado)
            if tipo_err in {"clasificacion", "almacenamiento"}:
                if st.button("Reintentar envio de esta opinion"):
                    procesar_envio_opinion(jugador_rec, recuperable["texto"])
                    st.rerun()

    texto_usuario = st.chat_input(
        f"Escribe tu opinion sobre {jugador_seleccionado['nombre']} (5 a 1000 caracteres)..."
    )
    if texto_usuario is not None:
        jugador_capturado = dict(jugador_seleccionado)
        procesar_envio_opinion(jugador_capturado, texto_usuario)
        st.rerun()


def renderizar_seccion_ficha_estadisticas(jugador: dict[str, Any]) -> None:
    """Muestra la ficha tecnica detallada: clubes, goles, asistencias y grafica de carrera."""
    st.subheader(f"Ficha Tecnica y Trayectoria: {jugador['nombre']}")
    renderizar_tarjeta_jugador(jugador)

    col_tray, col_logros = st.columns([3, 2])

    with col_tray:
        st.markdown("#### Trayectoria por Clubes y Seleccion")
        trayectoria = jugador.get("trayectoria", [])
        if trayectoria:
            df_tray = pd.DataFrame(trayectoria)
            df_tray.columns = ["Club / Seleccion", "Periodo", "Partidos", "Goles"]
            df_tray["Promedio (Gol/PJ)"] = (df_tray["Goles"] / df_tray["Partidos"]).round(2)
            st.dataframe(df_tray, width="stretch", hide_index=True)

            fig_goles = px.bar(
                df_tray,
                x="Club / Seleccion",
                y="Goles",
                color="Goles",
                text="Goles",
                title=f"Distribucion de Goles de {jugador['nombre']} por Club / Seleccion",
                color_continuous_scale="Blues",
            )
            fig_goles.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_goles, use_container_width=True)
        else:
            st.info("Informacion de clubes no detallada.")

    with col_logros:
        st.markdown("#### Palmares y Logros Destacados")
        logros = jugador.get("logros", [])
        if logros:
            for l in logros:
                st.markdown(f"- **{l}**")
        else:
            st.write("Sin logros registrados.")

        st.markdown("#### Ratios de Efectividad Global")
        partidos = jugador.get("partidos", 1) or 1
        goles = jugador.get("goles", 0)
        asist = jugador.get("asistencias", 0)

        prom_gol = round(goles / partidos, 2)
        participaciones = goles + asist
        prom_part = round(participaciones / partidos, 2)

        st.metric("Promedio Goleador Global", f"{prom_gol} goles/partido")
        st.metric("Goles + Asistencias (G/A)", f"{participaciones} totales ({prom_part} por partido)")


def renderizar_seccion_comparacion(catalogo: list[dict[str, Any]]) -> None:
    """Permite seleccionar dos futbolistas y comparar sus estadisticas y graficas de carrera."""
    st.subheader("Comparador Historico de Carreras")
    st.caption("Selecciona dos futbolistas del catalogo para comparar numeros reales, palmares y promedios.")

    nombres = [j["nombre"] for j in catalogo]
    mapa = {j["nombre"]: j for j in catalogo}

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        j1_nombre = st.selectbox("Primer Futbolista", options=nombres, index=0, key="comp_j1")
    with col_s2:
        j2_nombre = st.selectbox("Segundo Futbolista", options=nombres, index=1 if len(nombres) > 1 else 0, key="comp_j2")

    j1 = mapa[j1_nombre]
    j2 = mapa[j2_nombre]

    c_img1, c_vs, c_img2 = st.columns([2, 1, 2])
    with c_img1:
        if j1.get("foto_url"):
            st.image(j1["foto_url"], width=160, caption=f"{j1['nombre']} ({j1['pais']})")
        st.markdown(f"**{j1['nombre']}** - *{j1.get('posicion', '')}*")
    with c_vs:
        st.markdown("<h1 style='text-align: center; padding-top: 30px;'>VS</h1>", unsafe_allow_html=True)
    with c_img2:
        if j2.get("foto_url"):
            st.image(j2["foto_url"], width=160, caption=f"{j2['nombre']} ({j2['pais']})")
        st.markdown(f"**{j2['nombre']}** - *{j2.get('posicion', '')}*")

    st.divider()

    p1, p2 = j1.get("partidos", 1) or 1, j2.get("partidos", 1) or 1
    g1, g2 = j1.get("goles", 0), j2.get("goles", 0)
    a1, a2 = j1.get("asistencias", 0), j2.get("asistencias", 0)
    t1, t2 = j1.get("titulos", 0), j2.get("titulos", 0)
    b1, b2 = j1.get("balones_de_oro", 0), j2.get("balones_de_oro", 0)

    prom1 = round(g1 / p1, 2)
    prom2 = round(g2 / p2, 2)

    df_comp = pd.DataFrame({
        "Estadistica": [
            "Partidos Oficiales",
            "Goles Totales",
            "Asistencias",
            "Goles + Asistencias (G/A)",
            "Promedio Goleador (Gol/PJ)",
            "Titulos Oficiales",
            "Balones de Oro"
        ],
        j1["nombre"]: [p1, g1, a1, g1 + a1, prom1, t1, b1],
        j2["nombre"]: [p2, g2, a2, g2 + a2, prom2, t2, b2],
    })
    st.markdown("#### Cara a Cara Numerico")
    st.dataframe(df_comp, width="stretch", hide_index=True)

    st.markdown("#### Grafica Comparativa de Produccion Ofensiva")
    metricas = ["Goles", "Asistencias", "Titulos", "Balones de Oro"]
    fig_comp = go.Figure(data=[
        go.Bar(name=j1["nombre"], x=metricas, y=[g1, a1, t1, b1], marker_color="#1E88E5"),
        go.Bar(name=j2["nombre"], x=metricas, y=[g2, a2, t2, b2], marker_color="#D81B60")
    ])
    fig_comp.update_layout(barmode="group", height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("#### Radar de Rendimiento")
    categorias_radar = ["Goles", "Asistencias", "Titulos", "Balones de Oro", "Promedio x100"]
    r1 = [g1, a1, t1 * 10, b1 * 50, prom1 * 500]
    r2 = [g2, a2, t2 * 10, b2 * 50, prom2 * 500]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=r1, theta=categorias_radar, fill="toself", name=j1["nombre"], line_color="#1E88E5"))
    fig_radar.add_trace(go.Scatterpolar(r=r2, theta=categorias_radar, fill="toself", name=j2["nombre"], line_color="#D81B60"))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=450,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


def _mostrar_logo_competicion(archivo: str | None) -> None:
    """Muestra el logo local o placeholder si no esta disponible."""
    if not archivo:
        return
    ruta_local = ASSETS_DIR / archivo
    if ruta_local.exists():
        st.image(str(ruta_local), width=140)
    else:
        st.write(f"[{archivo}]")


def renderizar_seccion_competiciones() -> None:
    """Muestra el analisis y tablas de las 5 Grandes Ligas y Competiciones Europeas (UCL, UEL, UECL)."""
    st.subheader("Grandes Ligas Europeas & Torneos UEFA")
    st.caption("Tablas de posiciones oficiales, campeones vigentes y estatus competitivo.")

    datos_comp = cargar_competiciones(RUTA_COMPETICIONES)
    ligas = datos_comp.get("ligas", [])
    torneos_uefa = datos_comp.get("competiciones_europeas", [])

    pestana_ligas, pestana_uefa = st.tabs(["5 Grandes Ligas", "Competiciones Europeas (UCL / UEL / UECL)"])

    with pestana_ligas:
        nombres_ligas = [l["nombre"] for l in ligas]
        mapa_ligas = {l["nombre"]: l for l in ligas}

        liga_elegida = st.selectbox("Selecciona una Liga", options=nombres_ligas, index=0)
        info_liga = mapa_ligas[liga_elegida]

        col_logo, col_desc = st.columns([1, 4])
        with col_logo:
            _mostrar_logo_competicion(info_liga.get("logo_archivo"))
        with col_desc:
            st.markdown(f"### {info_liga['nombre']} ({info_liga['pais']})")
            st.markdown(f"**Campeon Vigente:** {info_liga.get('campeon_actual', 'Por definir')}")
            st.write(info_liga.get("descripcion", ""))

        st.markdown("#### Tabla de Posiciones y Zona de Clasificacion")
        equipos = info_liga.get("equipos", [])
        if equipos:
            df_equipos = pd.DataFrame(equipos)
            df_equipos.columns = ["Equipo", "Posicion", "Partidos Jugados (PJ)", "Puntos", "Diferencia de Gol (DG)", "Estado"]
            st.dataframe(df_equipos, width="stretch", hide_index=True)

            fig_pts = px.bar(
                df_equipos,
                x="Equipo",
                y="Puntos",
                color="Puntos",
                text="Puntos",
                title=f"Puntaje Actual - {info_liga['nombre']}",
                color_continuous_scale="Viridis",
            )
            fig_pts.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pts, use_container_width=True)

    with pestana_uefa:
        nombres_uefa = [u["nombre"] for u in torneos_uefa]
        mapa_uefa = {u["nombre"]: u for u in torneos_uefa}

        torneo_elegido = st.selectbox("Selecciona Competicion Continental", options=nombres_uefa, index=0)
        info_torneo = mapa_uefa[torneo_elegido]

        col_logo_u, col_desc_u = st.columns([1, 4])
        with col_logo_u:
            _mostrar_logo_competicion(info_torneo.get("logo_archivo"))
        with col_desc_u:
            st.markdown(f"### {info_torneo['nombre']}")
            st.markdown(f"**Campeon Actual:** {info_torneo.get('campeon_actual', '')} | **Formato:** {info_torneo.get('tipo', '')}")
            st.write(info_torneo.get("descripcion", ""))

        st.markdown("#### Tabla de la Fase de Liga Unica")
        tabla_fase = info_torneo.get("tabla_fase_liga", [])
        if tabla_fase:
            df_uefa = pd.DataFrame(tabla_fase)
            df_uefa.columns = ["Club", "Posicion", "Partidos Jugados (PJ)", "Puntos", "Diferencia de Gol (DG)", "Zona"]
            st.dataframe(df_uefa, width="stretch", hide_index=True)

            fig_uefa = px.bar(
                df_uefa,
                x="Club",
                y="Puntos",
                color="Puntos",
                text="Puntos",
                title=f"Clasificacion Fase de Liga - {info_torneo['nombre']}",
                color_continuous_scale="Plasma",
            )
            fig_uefa.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_uefa, use_container_width=True)


def renderizar_seccion_consultar(
    catalogo_jugadores: list[dict[str, Any]],
    mapa_jugadores: dict[str, dict[str, Any]],
) -> None:
    """Renderiza la consulta filtrada, recuentos y descarga JSON sin llamar al modelo."""
    st.subheader("Consultar opiniones registradas")
    st.caption(
        "Los datos mostrados provienen exclusivamente del archivo persistente datos/opiniones.json. "
        "Esta seccion no realiza llamadas al modelo de lenguaje."
    )

    try:
        datos_persistidos = cargar_opiniones(RUTA_OPINIONES)
    except ArchivoOpinionesCorruptoError as exc:
        st.error(
            f"**Archivo de opiniones danado:** {exc}\n\n"
            "El archivo original se ha conservado en disco sin sobrescribirse para evitar perdida de datos."
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
    m_pos.metric("Positivo", conteo_por_sentimiento["positivo"])
    m_neg.metric("Negativo", conteo_por_sentimiento["negativo"])
    m_neu.metric("Neutro", conteo_por_sentimiento["neutro"])
    m_mix.metric("Mixto", conteo_por_sentimiento["mixto"])

    if not registros_filtrados:
        st.info("No se encontraron opiniones registradas para la combinacion de filtros seleccionada.")
    else:
        filas_tabla = [
            {
                "Fecha (UTC)": reg.get("fecha_utc", ""),
                "Jugador": reg.get("jugador_nombre", ""),
                "Opinion original": reg.get("opinion_original", ""),
                "Sentimiento": reg.get("sentimiento", ""),
                "Razon": reg.get("razon", ""),
            }
            for reg in registros_filtrados
        ]
        st.dataframe(filas_tabla, width="stretch", hide_index=True)

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
        label="Descargar resultados filtrados (JSON)",
        data=json_descarga.encode("utf-8"),
        file_name="opiniones_filtradas.json",
        mime="application/json",
    )


def main() -> None:
    """Punto de entrada principal de la aplicacion Streamlit."""
    st.set_page_config(
        page_title="Tribuna Global | Percepcion y Estadisticas de Futbolistas",
        layout="wide",
    )
    inicializar_estado_sesion()

    st.title("Tribuna Global: Hub de Futbol & IA")
    st.markdown(
        "Sistema integral de analisis de percepcion mediante **LangChain** y **Pydantic**, "
        "con estadisticas historicas reales, fichas de carrera y comparativas graficas cara a cara."
    )

    try:
        catalogo_jugadores = cargar_jugadores(RUTA_JUGADORES)
    except ErrorCatalogoJugadores as exc:
        st.error(f"**Error al cargar el catalogo de jugadores:** {exc}")
        st.stop()

    mapa_jugadores = {j["id"]: j for j in catalogo_jugadores}
    ids_jugadores = [j["id"] for j in catalogo_jugadores]

    with st.sidebar:
        st.header("Panel de Navegacion")

        seccion_activa = st.radio(
            "Seccion",
            options=[
                "Emitir opinion",
                "Consultar opiniones",
                "Ficha y Trayectoria",
                "Comparador de Carreras",
                "Grandes Ligas & UEFA"
            ],
            index=0,
        )

        st.divider()
        st.subheader("Futbolista Activo")
        jugador_id_activo = st.selectbox(
            "Selecciona un jugador",
            options=ids_jugadores,
            format_func=lambda jid: f"{mapa_jugadores[jid]['nombre']} ({mapa_jugadores[jid]['pais']})",
            key="selector_jugador_sidebar",
        )
        jugador_seleccionado = mapa_jugadores[jugador_id_activo]

        st.divider()
        if st.button("Nueva conversacion", width="stretch"):
            limpiar_conversacion()
            st.rerun()
        st.caption(
            "«Nueva conversacion» limpia el chat de la sesion actual y conserva todas las "
            "opiniones guardadas en datos/opiniones.json."
        )

    if seccion_activa == "Emitir opinion":
        renderizar_seccion_emitir(jugador_seleccionado)
    elif seccion_activa == "Consultar opiniones":
        renderizar_seccion_consultar(catalogo_jugadores, mapa_jugadores)
    elif seccion_activa == "Ficha y Trayectoria":
        renderizar_seccion_ficha_estadisticas(jugador_seleccionado)
    elif seccion_activa == "Comparador de Carreras":
        renderizar_seccion_comparacion(catalogo_jugadores)
    elif seccion_activa == "Grandes Ligas & UEFA":
        renderizar_seccion_competiciones()


if __name__ == "__main__":
    main()
