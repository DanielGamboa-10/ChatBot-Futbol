# Tribuna Global: Hub de Futbol & Analisis con Inteligencia Artificial

[![Python 3.12 / 3.14](https://img.shields.io/badge/Python-3.12%20%7C%203.14-purple.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-a855f7.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain--Core-1.7.6-c084fc.svg)](https://python.langchain.com/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-2.14.3-9333ea.svg)](https://docs.pydantic.dev/)
[![Tests](https://img.shields.io/badge/pytest-11%20passed%20(100%25)-success.svg)](tribuna_global/tests/test_aceptacion.py)
[![UI Theme](https://img.shields.io/badge/Theme-Neon%20Purple%20Cyberpunk-7e22ce.svg)](.streamlit/config.toml)

**Tribuna Global** es una plataforma web interactiva y modular construida sobre **Streamlit**, **LangChain (LCEL)** y modelos de lenguaje de última generación (`ChatOpenAI` / `ChatGoogleGenerativeAI`). Permite capturar valoraciones futbolísticas de aficionados, clasificar estructuradamente su sentimiento mediante un contrato estricto de **Pydantic**, almacenar los registros atómicamente en disco, y explorar un hub completo con estadísticas de carrera reales, comparación cara a cara de leyendas e información de las 5 grandes ligas y torneos UEFA para la temporada 2026-27.

---

## Tabla de Contenidos
1. [Requisitos Previos](#requisitos-previos)
2. [Guia de Instalacion Paso a Paso](#guia-de-instalacion-paso-a-paso)
3. [Configuracion de Variables de Entorno](#configuracion-de-variables-de-entorno)
4. [Levantamiento del Proyecto](#levantamiento-del-proyecto)
5. [Optima Utilizacion y Flujo de Trabajo](#optima-utilizacion-y-flujo-de-trabajo)
6. [Estructura del Proyecto y Modulos](#estructura-del-proyecto-y-modulos)
7. [Arquitectura y Contratos de Inteligencia Artificial](#arquitectura-y-contratos-de-inteligencia-artificial)
8. [Verificacion y Pruebas Automatizadas](#verificacion-y-pruebas-automatizadas)
9. [Solucion de Problemas Frecuentes](#solucion-de-problemas-frecuentes)

---

## Requisitos Previos

Antes de desplegar el proyecto, asegúrate de contar con:
- **Python**: Versión 3.11, 3.12 o 3.14 instalada y accesible en el sistema (`py` o `python`).
- **Gestor de Paquetes**: `pip` actualizado.
- **Git**: Para control de versiones y clonación del repositorio.
- **Clave de API (Opcional para pruebas)**:
  - `OPENAI_API_KEY` (OpenAI GPT-4o / GPT-4o-mini) o
  - `GOOGLE_API_KEY` (Google Gemini 2.5 Flash / Pro).
  *Nota*: La aplicación cuenta con un motor de respaldo determinista y estructurado que garantiza operatividad local continua incluso si no se dispone de cuota o clave activa en el momento.

---

## Guia de Instalacion Paso a Paso

### 1. Clonar el Repositorio
```bash
git clone https://github.com/DanielGamboa-10/ChatBot-Futbol.git
cd ChatBot-Futbol
```

### 2. Crear y Activar un Entorno Virtual (Recomendado)
Para aislar las dependencias del proyecto:
- **En Windows (PowerShell):**
  ```powershell
  py -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **En Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Instalar Dependencias
Instala los paquetes fijados en `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Configuracion de Variables de Entorno

Copia el archivo de plantilla `.env.example` para generar tu archivo `.env`:

```powershell
# En Windows PowerShell:
Copy-Item .env.example .env

# En Linux / macOS:
cp .env.example .env
```

Edita `.env` con un editor de texto e ingresa tu proveedor preferido:
```ini
# Proveedor principal (openai o google)
LLM_PROVIDER=openai
OPENAI_API_KEY=tu-clave-de-openai-aqui
OPENAI_MODEL_NAME=gpt-4o-mini

# Proveedor alternativo (opcional)
GOOGLE_API_KEY=tu-clave-de-gemini-aqui
GOOGLE_MODEL_NAME=gemini-2.5-flash
```

---

## Levantamiento del Proyecto

Para iniciar la aplicación en tu entorno local:

```bash
# Ejecucion directa con Streamlit
py -m streamlit run tribuna_global/app.py
```

Streamlit desplegará automáticamente la aplicación en tu navegador predeterminado en:
```text
Local URL: http://localhost:8501
Network URL: http://<tu-ip-local>:8501
```

> **Consejo de rendimiento:** Si necesitas levantar la aplicación sin abrir el navegador automáticamente:
> ```bash
> py -m streamlit run tribuna_global/app.py --server.headless=true --server.port=8501
> ```

---

## Optima Utilizacion y Flujo de Trabajo

La interfaz se divide en 5 módulos accesibles desde la barra lateral izquierda:

```text
Tribuna Global
├── 1. Emitir opinion          -> Chat interactivo con analisis de IA y tarjeta del jugador
├── 2. Consultar opiniones     -> Explorador de registros filtrados por jugador y sentimiento
├── 3. Ficha y Trayectoria     -> Historial por clubes, goles y logros deportivos
├── 4. Comparador de Carreras  -> Analisis cara a cara con graficas Radar y Barras comparativas
└── 5. Grandes Ligas & UEFA    -> Tablas de clasificacion temporada 2026-27 al 10/11/2026
```

### 1. Evaluacion de Percepcion (Emitir Opinion)
1. En la barra lateral, selecciona el **Futbolista Activo** que deseas evaluar (ej. *Lionel Messi*, *Cristiano Ronaldo*, *Kylian Mbappé*).
2. En la caja de chat inferior, ingresa una opinión sobre su rendimiento, legado o estilo de juego (mínimo 5 caracteres, máximo 1000).
3. La IA evalúa dos aspectos fundamentales:
   - **Pertinencia**: Si el comentario habla efectivamente de dicho futbolista o del ámbito deportivo relacionado. Si el comentario es ajeno (ej. una receta de cocina o sobre otro deportista no relacionado), se descarta de forma pedagógica y no se persiste.
   - **Sentimiento**: Clasificación estricta en `positivo`, `negativo`, `neutro` o `mixto`, acompañada de una justificación razonada.
4. Si la opinión es pertinente, se genera un identificador único (UUID v4) y se persiste atómicamente en disco.

### 2. Consulta y Auditoria de Registros (Consultar Opiniones)
- Filtra por jugador específico o visualiza el universo completo de opiniones guardadas.
- Aplica filtros cruzados por categoría de sentimiento.
- Genera resúmenes visuales de métricas agregadas y permite descargar la selección en formato JSON para análisis de datos externo.
- **Importante**: Esta sección lee exclusivamente de `datos/opiniones.json` y no consume tokens ni llamadas a la API de lenguaje.

### 3. Fichas de Trayectoria y Comparador de Carreras
- Consulta la trayectoria desglosada por cada club en el que militó el futbolista, indicando períodos, partidos jugados y goles convertidos.
- Utiliza el **Comparador de Carreras** para enfrentar a dos futbolistas del catálogo y visualizar en paralelo sus métricas de eficacia (promedio de gol por partido, ratio de asistencias, títulos acumulados y Balones de Oro) mediante gráficos interactivos de Plotly (gráfico de radar y gráfico de barras).

### 4. Seguimiento de Competiciones (Temporada 2026-27 al 10/11/2026)
- **Competiciones Europeas**: Consulta el estado de la Fase de Liga Única de la UEFA Champions League (36 equipos), UEFA Europa League y UEFA Conference League.
- **5 Grandes Ligas**: Clasificaciones de Premier League, LaLiga EA Sports, Serie A Enilive, Bundesliga y Ligue 1.

---

## Estructura del Proyecto y Modulos

```text
docker-clase/
├── .streamlit/
│   └── config.toml             # Configuracion visual del tema morado y acentos neon
├── tribuna_global/
│   ├── assets/                 # Logos locales de alta resolucion (UCL, UEL, UECL y ligas)
│   ├── datos/
│   │   ├── jugadores.json      # Catalogo oficial con 20 futbolistas, fotos y estadisticas reales
│   │   ├── opiniones.json      # Base de datos persistente JSON con registros atomicos
│   │   └── competiciones.json  # Tablas de clasificacion temporada 2026-27
│   ├── tests/
│   │   └── test_aceptacion.py  # Suite de pruebas automatizadas con pytest (11/11 pasadas)
│   ├── almacenamiento.py       # Capa de datos, escrituras atomicas y validaciones
│   ├── analisis.py             # Cadena LCEL, prompts tipados y contrato Pydantic
│   └── app.py                  # Interfaz de usuario Streamlit con tema neon y animacion CSS
├── requirements.txt            # Dependencias fijadas con versiones exactas
├── .env.example                # Plantilla de variables de entorno
└── README.md                   # Documentacion tecnica de referencia
```

---

## Arquitectura y Contratos de Inteligencia Artificial

### Contrato Estricto con Pydantic (`AnalisisOpinion`)
La salida estructurada se valida mediante el siguiente modelo de datos:

```python
class AnalisisOpinion(BaseModel):
    es_pertinente: bool = Field(
        description="True si el texto evalúa o comenta sobre el futbolista indicado"
    )
    sentimiento: Optional[Literal["positivo", "negativo", "neutro", "mixto"]] = Field(
        default=None,
        description="Categoría de percepción (obligatoria si es_pertinente=True, None si es False)"
    )
    razon: str = Field(
        description="Explicación concisa y fundamentada de la clasificación asignada"
    )
```

### Garantía de Integridad y Persistencia Atómica
Para evitar corrupciones de datos ante cierres inesperados de proceso:
1. El registro se escribe inicialmente en un archivo temporal (`opiniones.json.tmp`).
2. Se fuerza la sincronización a nivel de hardware con `os.fsync()`.
3. Se reemplaza el archivo final de forma atómica mediante `os.replace()`.

---

## Verificacion y Pruebas Automatizadas

El proyecto cuenta con una batería de pruebas de integración y aceptación que cubren los requisitos de la rúbrica al 100%:

```bash
py -m pytest tribuna_global/tests/test_aceptacion.py -v
```

### Matriz de Pruebas Cubiertas:
1. `test_actividad1_catalogo_20_jugadores_sin_duplicados`: Verifica 20 futbolistas únicos en catálogo.
2. `test_actividad1_rechaza_ids_repetidos`: Valida integridad referencial contra duplicados.
3. `test_entregable_opiniones_semilla_minimo_12_registros_4_jugadores_4_categorias`: Verifica la presencia de semillas representativas.
4. `test_actividad3_prompt_roles_system_y_human`: Valida la asignación explícita de roles para blindaje de prompt.
5. `test_actividad4_contrato_pydantic_relacion_pertinencia_sentimiento`: Comprueba la consistencia cruzada de tipos en Pydantic.
6. `test_casos_clasificacion_aceptacion_tabla_seccion_9`: Ejecuta casos límite de ironía, sarcasmo y comentarios mixtos.
7. `test_caso_entrada_vacia_o_fuera_de_rango_no_llama_al_modelo`: Asegura validación previa en front sin gastar tokens.
8. `test_caso_fallo_de_api_no_genera_registro_neutro`: Garantiza que fallos de red no generen datos corruptos.
9. `test_persistencia_conservacion_cambio_jugador_y_no_duplicacion_por_id`: Prueba la estabilidad de la clave primaria UUID.
10. `test_caso_archivo_danado_se_conserva_intacto`: Protege archivos dañados evitando su sobreescritura accidental.
11. `test_interfaz_streamlit_limpieza_chat_y_consulta_sin_duplicar`: Simula la interacción en `AppTest` y el botón de nueva conversación.

---

## Solucion de Problemas Frecuentes

- **Puerto 8501 ocupado:**
  Si Streamlit indica que el puerto está en uso por otro proceso:
  ```powershell
  Get-Process | Where-Object { $_.ProcessName -match "python" } | Stop-Process -Force
  ```
  O lanza la aplicación en un puerto alternativo:
  ```bash
  py -m streamlit run tribuna_global/app.py --server.port 8502
  ```

- **Fallo de cuota en OpenAI (Error 429 / 401):**
  La aplicación conmuta automáticamente al motor `ClasificadorEstructuradoRespaldo`, permitiendo continuar el uso y las pruebas sin interrupción.

- **Los cambios de tema o logos no se reflejan:**
  Presiona **Ctrl + F5** (o Cmd + Shift + R en Mac) en el navegador para forzar la recarga de la caché estática.
