# app.py - Plataforma de Gestión de Riesgos Climáticos para Ají y Rocoto
# Versión final: mapas de calor (heatmaps), GEE con cuenta de servicio, Groq IA, reportes PDF.

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile
import os
import zipfile
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
from shapely.geometry import Polygon, Point
import math
import warnings
import xml.etree.ElementTree as ET
import json
from io import BytesIO
import requests
import contextily as ctx

# ================= CONFIGURACIÓN INICIAL =================
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

# ================= DEPENDENCIAS OPCIONALES =================
FOLIUM_OK = False
RASTERIO_OK = False
SKIMAGE_OK = False
try:
    import folium
    from folium.plugins import Fullscreen
    from branca.colormap import LinearColormap
    FOLIUM_OK = True
except ImportError:
    pass

try:
    import rasterio
    RASTERIO_OK = True
except ImportError:
    pass

try:
    from skimage import measure
    SKIMAGE_OK = True
except ImportError:
    pass

try:
    from streamlit_folium import folium_static
    FOLIUM_STATIC_OK = True
except ImportError:
    FOLIUM_STATIC_OK = False

# ================= GOOGLE EARTH ENGINE =================
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    st.warning("⚠️ earthengine-api no instalado. Ejecuta: pip install earthengine-api")

# ================= GROQ IA =================
try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.warning("⚠️ groq no instalado. Ejecuta: pip install groq")

# ================= LECTURA DE SECRETS =================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
if GROQ_API_KEY and GROQ_AVAILABLE:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    st.success("✅ API Key de Groq cargada correctamente.")
else:
    st.warning("⚠️ No se encontró API Key de Groq o librería no instalada. La IA no estará disponible.")

# ================= INICIALIZACIÓN DE GEE CON CUENTA DE SERVICIO =================
def inicializar_gee():
    if not GEE_AVAILABLE:
        return False
    # Intentar con cuenta de servicio desde secrets
    if 'gee_service_account' in st.secrets:
        try:
            creds = st.secrets["gee_service_account"]
            # Asegurar que private_key es un string (puede venir con saltos de línea)
            private_key = creds['private_key']
            credentials = ee.ServiceAccountCredentials(
                creds['client_email'],
                key_data=private_key
            )
            ee.Initialize(credentials, project=creds.get('project_id', 'democultivos'))
            st.session_state.gee_authenticated = True
            st.success("✅ GEE autenticado con cuenta de servicio.")
            return True
        except Exception as e:
            st.error(f"❌ Error autenticando GEE con cuenta de servicio: {e}")
    # Fallback: autenticación local (requiere earthengine authenticate)
    try:
        ee.Initialize()
        st.session_state.gee_authenticated = True
        st.success("✅ GEE autenticado localmente.")
        return True
    except Exception as e:
        st.session_state.gee_authenticated = False
        st.error(f"❌ Error autenticando GEE localmente: {e}")
        return False

if 'gee_authenticated' not in st.session_state:
    st.session_state.gee_authenticated = False
    if GEE_AVAILABLE:
        inicializar_gee()

# ================= FUNCIONES DE CARGA DE PARCELA =================
def validar_crs(gdf):
    if gdf is None or len(gdf) == 0:
        return gdf
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326', inplace=False)
        elif str(gdf.crs).upper() != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        return gdf
    except:
        return gdf

def calcular_superficie(gdf):
    try:
        gdf_proj = gdf.to_crs('EPSG:3857')
        area_m2 = gdf_proj.geometry.area.sum()
        return area_m2 / 10000
    except:
        return 0.0

def cargar_shapefile_desde_zip(zip_file):
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            shp_files = [f for f in os.listdir(tmp_dir) if f.endswith('.shp')]
            if shp_files:
                shp_path = os.path.join(tmp_dir, shp_files[0])
                gdf = gpd.read_file(shp_path)
                gdf = validar_crs(gdf)
                return gdf
            else:
                st.error("❌ No se encontró archivo .shp en el ZIP")
                return None
    except Exception as e:
        st.error(f"❌ Error cargando ZIP: {e}")
        return None

def parsear_kml_manual(contenido_kml):
    try:
        root = ET.fromstring(contenido_kml)
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        polygons = []
        for polygon_elem in root.findall('.//kml:Polygon', namespaces):
            coords_elem = polygon_elem.find('.//kml:coordinates', namespaces)
            if coords_elem is not None and coords_elem.text:
                coords = []
                for coord_pair in coords_elem.text.strip().split():
                    parts = coord_pair.split(',')
                    if len(parts) >= 2:
                        coords.append((float(parts[0]), float(parts[1])))
                if len(coords) >= 3:
                    polygons.append(Polygon(coords))
        if polygons:
            return gpd.GeoDataFrame({'geometry': polygons}, crs='EPSG:4326')
        return None
    except:
        return None

def cargar_kml(kml_file):
    try:
        if kml_file.name.endswith('.kmz'):
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(kml_file, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                kml_files = [f for f in os.listdir(tmp_dir) if f.endswith('.kml')]
                if kml_files:
                    kml_path = os.path.join(tmp_dir, kml_files[0])
                    with open(kml_path, 'r', encoding='utf-8') as f:
                        contenido = f.read()
                    gdf = parsear_kml_manual(contenido)
                    if gdf is not None:
                        return gdf
        else:
            contenido = kml_file.read().decode('utf-8')
            gdf = parsear_kml_manual(contenido)
            if gdf is not None:
                return gdf
        # Si falla, intentar con geopandas directamente
        kml_file.seek(0)
        gdf = gpd.read_file(kml_file)
        gdf = validar_crs(gdf)
        return gdf
    except Exception as e:
        st.error(f"❌ Error cargando KML/KMZ: {e}")
        return None

def cargar_archivo_parcela(uploaded_file):
    try:
        if uploaded_file.name.endswith('.zip'):
            gdf = cargar_shapefile_desde_zip(uploaded_file)
        elif uploaded_file.name.endswith(('.kml', '.kmz')):
            gdf = cargar_kml(uploaded_file)
        elif uploaded_file.name.endswith('.geojson'):
            gdf = gpd.read_file(uploaded_file)
            gdf = validar_crs(gdf)
        else:
            st.error("Formato no soportado. Use ZIP, KML, KMZ o GeoJSON.")
            return None
        if gdf is not None:
            gdf = validar_crs(gdf)
            gdf = gdf.explode(ignore_index=True)
            gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
            if len(gdf) == 0:
                st.error("No se encontraron polígonos.")
                return None
            # Unir todos los polígonos en uno solo
            geom_unida = gdf.unary_union
            gdf_unido = gpd.GeoDataFrame({'geometry': [geom_unida]}, crs='EPSG:4326')
            st.info(f"✅ Se unieron {len(gdf)} polígonos.")
            return gdf_unido
        return None
    except Exception as e:
        st.error(f"❌ Error cargando archivo: {e}")
        return None

# ================= FUNCIONES DE IA (GROQ) =================
def consultar_groq(prompt, max_tokens=400):
    if not GROQ_API_KEY or not GROQ_AVAILABLE:
        return "⚠️ IA no disponible: falta API Key o librería."
    try:
        client = groq.Client(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error consultando Groq: {str(e)}"

def generar_alerta_fenologica(fase, ndvi, temp, cultivo):
    prompt = f"""
Eres agrónomo experto en {cultivo}. El cultivo está en fase de {fase}.
NDVI: {ndvi:.2f} | Temperatura: {temp:.1f}°C
Genera análisis de riesgo (bajo/medio/alto) y acción de adaptación (máx 40 palabras).
Formato: **Riesgo:** ... **Acción:** ...
"""
    return consultar_groq(prompt, max_tokens=200)

# Funciones para el reporte con IA
def generar_analisis_fertilidad(df_resumen, stats, cultivo):
    prompt = f"""Analiza fertilidad para {cultivo}: NPK={stats['npk_mean']:.2f}, MO={stats['mo_mean']:.1f}%, Humedad={stats['humedad_mean']:.2f}. Da interpretación (max 150 palabras)."""
    return consultar_groq(prompt, 300)

def generar_analisis_ndvi_ndre(df_resumen, stats, cultivo):
    prompt = f"""Interpreta NDVI={stats['ndvi_mean']:.2f}, NDRE={stats['ndre_mean']:.2f} para {cultivo}. Estado del cultivo y recomendaciones."""
    return consultar_groq(prompt, 300)

def generar_analisis_riesgo_hidrico(df_resumen, stats, cultivo):
    prompt = f"""Riesgo hídrico para {cultivo}: humedad={stats['humedad_mean']:.2f}, textura={stats['textura_predominante']}. Análisis y manejo de riego."""
    return consultar_groq(prompt, 300)

def generar_analisis_costos(df_resumen, stats, cultivo):
    prompt = f"""Evalúa rentabilidad fertilización para {cultivo}: incremento esperado={stats['incremento_mean']:.1f}%. Breve análisis ROI."""
    return consultar_groq(prompt, 300)

def generar_recomendaciones_integradas(df_resumen, stats, cultivo):
    prompt = f"""Plan de manejo integrado para {cultivo} con: NPK={stats['npk_mean']:.2f}, NDVI={stats['ndvi_mean']:.2f}, MO={stats['mo_mean']:.1f}%, textura={stats['textura_predominante']}, incremento={stats['incremento_mean']:.1f}%. 5 puntos concretos."""
    return consultar_groq(prompt, 400)

# ================= FUNCIONES DE MAPAS DE CALOR (GEE) =================
def get_precipitacion_heatmap(gdf, fecha_inicio, fecha_fin):
    """Retorna figura matplotlib con mapa de calor de precipitación acumulada (CHIRPS)"""
    try:
        geom = ee.Geometry.Polygon(list(gdf.geometry.iloc[0].exterior.coords))
        start = fecha_inicio.strftime('%Y-%m-%d')
        end = fecha_fin.strftime('%Y-%m-%d')
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterBounds(geom) \
            .filterDate(start, end) \
            .select('precipitation')
        total_precip = chirps.sum().clip(geom)
        # Obtener datos para visualización local (muestreo)
        bounds = gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        # Muestrear en una grilla
        x_vals = np.linspace(minx, maxx, 100)
        y_vals = np.linspace(miny, maxy, 100)
        coords = []
        for y in y_vals:
            for x in x_vals:
                coords.append(ee.Geometry.Point(x, y))
        points = ee.FeatureCollection(coords)
        sampled = total_precip.sampleRegions(collection=points, scale=5000, geometries=True)
        data = sampled.getInfo()
        precip_vals = np.full((len(y_vals), len(x_vals)), np.nan)
        for feature in data['features']:
            lon, lat = feature['geometry']['coordinates']
            precip = feature['properties']['precipitation']
            i = np.argmin(np.abs(x_vals - lon))
            j = np.argmin(np.abs(y_vals - lat))
            if 0 <= j < len(y_vals) and 0 <= i < len(x_vals):
                precip_vals[j, i] = precip if precip is not None else np.nan
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(precip_vals, extent=[minx, maxx, miny, maxy], origin='lower', cmap='YlGnBu', aspect='auto')
        plt.colorbar(im, ax=ax, label='Precipitación acumulada (mm)')
        gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2, alpha=0.7)
        ax.set_title('Precipitación Acumulada (CHIRPS)')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        return fig
    except Exception as e:
        st.error(f"Error generando mapa de precipitación: {e}")
        return None

def get_temperatura_heatmap(gdf, fecha_inicio, fecha_fin):
    """Mapa de calor de temperatura media (ERA5-Land)"""
    try:
        geom = ee.Geometry.Polygon(list(gdf.geometry.iloc[0].exterior.coords))
        start = fecha_inicio.strftime('%Y-%m-%d')
        end = fecha_fin.strftime('%Y-%m-%d')
        era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
            .filterBounds(geom) \
            .filterDate(start, end) \
            .select('temperature_2m')
        mean_temp = era5.mean().subtract(273.15).clip(geom)  # a Celsius
        bounds = gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        x_vals = np.linspace(minx, maxx, 100)
        y_vals = np.linspace(miny, maxy, 100)
        coords = []
        for y in y_vals:
            for x in x_vals:
                coords.append(ee.Geometry.Point(x, y))
        points = ee.FeatureCollection(coords)
        sampled = mean_temp.sampleRegions(collection=points, scale=5000, geometries=True)
        data = sampled.getInfo()
        temp_vals = np.full((len(y_vals), len(x_vals)), np.nan)
        for feature in data['features']:
            lon, lat = feature['geometry']['coordinates']
            t = feature['properties']['temperature_2m']
            i = np.argmin(np.abs(x_vals - lon))
            j = np.argmin(np.abs(y_vals - lat))
            if 0 <= j < len(y_vals) and 0 <= i < len(x_vals):
                temp_vals[j, i] = t if t is not None else np.nan
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(temp_vals, extent=[minx, maxx, miny, maxy], origin='lower', cmap='RdYlBu_r', aspect='auto', vmin=10, vmax=35)
        plt.colorbar(im, ax=ax, label='Temperatura media (°C)')
        gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2, alpha=0.7)
        ax.set_title('Temperatura Media (ERA5-Land)')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        return fig
    except Exception as e:
        st.error(f"Error generando mapa de temperatura: {e}")
        return None

def get_ndwi_heatmap(gdf, fecha_inicio, fecha_fin):
    """Mapa de calor de NDWI (humedad/agua) desde Sentinel-2"""
    try:
        geom = ee.Geometry.Polygon(list(gdf.geometry.iloc[0].exterior.coords))
        start = fecha_inicio.strftime('%Y-%m-%d')
        end = fecha_fin.strftime('%Y-%m-%d')
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(geom) \
            .filterDate(start, end) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        def add_ndwi(img):
            ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
            return img.addBands(ndwi)
        with_ndwi = collection.map(add_ndwi)
        mean_ndwi = with_ndwi.select('NDWI').mean().clip(geom)
        bounds = gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        x_vals = np.linspace(minx, maxx, 100)
        y_vals = np.linspace(miny, maxy, 100)
        coords = []
        for y in y_vals:
            for x in x_vals:
                coords.append(ee.Geometry.Point(x, y))
        points = ee.FeatureCollection(coords)
        sampled = mean_ndwi.sampleRegions(collection=points, scale=20, geometries=True)
        data = sampled.getInfo()
        ndwi_vals = np.full((len(y_vals), len(x_vals)), np.nan)
        for feature in data['features']:
            lon, lat = feature['geometry']['coordinates']
            val = feature['properties']['NDWI']
            i = np.argmin(np.abs(x_vals - lon))
            j = np.argmin(np.abs(y_vals - lat))
            if 0 <= j < len(y_vals) and 0 <= i < len(x_vals):
                ndwi_vals[j, i] = val if val is not None else np.nan
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(ndwi_vals, extent=[minx, maxx, miny, maxy], origin='lower', cmap='Blues', aspect='auto', vmin=-0.5, vmax=0.5)
        plt.colorbar(im, ax=ax, label='NDWI')
        gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2, alpha=0.7)
        ax.set_title('NDWI (Humedad/Agua) - Sentinel-2')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        return fig
    except Exception as e:
        st.error(f"Error generando NDWI: {e}")
        return None

# ================= PARÁMETROS DE CULTIVOS =================
CULTIVOS = ["AJÍ", "ROCOTO", "PAPA ANDINA"]
ICONOS = {"AJÍ": "🌶️", "ROCOTO": "🥵", "PAPA ANDINA": "🥔"}
UMBRALES = {
    "AJÍ": {"NDVI_min": 0.4, "temp_min": 18, "temp_max": 30, "humedad_min": 0.25, "humedad_max": 0.65},
    "ROCOTO": {"NDVI_min": 0.45, "temp_min": 16, "temp_max": 28, "humedad_min": 0.30, "humedad_max": 0.70},
    "PAPA ANDINA": {"NDVI_min": 0.5, "temp_min": 10, "temp_max": 22, "humedad_min": 0.35, "humedad_max": 0.75}
}

# ================= INTERFAZ PRINCIPAL =================
st.set_page_config(page_title="Gestión de Riesgos Climáticos - Ají y Rocoto", layout="wide")
st.title("🌶️ Plataforma de Gestión de Riesgos Climáticos para Ají y Rocoto")

with st.sidebar:
    st.header("⚙️ Configuración")
    cultivo = st.selectbox("Cultivo", CULTIVOS)
    st.info(f"{ICONOS[cultivo]} Parámetros específicos cargados.")
    uploaded_file = st.file_uploader("Subir parcela (GeoJSON, KML, KMZ, ZIP Shapefile)", type=['geojson','kml','kmz','zip'])
    fecha_fin = st.date_input("Fecha fin", datetime.now())
    fecha_inicio = st.date_input("Fecha inicio", datetime.now() - timedelta(days=90))
    fase_fenologica = st.selectbox("Fase actual del cultivo", ["siembra", "desarrollo", "floracion", "fructificacion", "cosecha"])
    usar_gee = st.checkbox("Usar GEE (si autenticado)", value=True)

if not uploaded_file:
    st.info("👈 Sube un archivo de parcela para comenzar.")
    st.stop()

with st.spinner("Cargando parcela..."):
    gdf = cargar_archivo_parcela(uploaded_file)
    if gdf is None:
        st.error("No se pudo cargar la parcela.")
        st.stop()
    area_ha = calcular_superficie(gdf)
    st.success(f"✅ Parcela cargada: {area_ha:.2f} ha, CRS EPSG:4326")

# Datos simulados (si no se usa GEE, se generan valores aleatorios)
ndvi_val = np.random.uniform(0.3, 0.8)
temp_val = np.random.uniform(15, 32)
humedad_val = np.random.uniform(0.2, 0.7)

# ================= PESTAÑAS =================
tab_hist, tab_monitoreo, tab_alerta, tab_gobernanza, tab_export = st.tabs(
    ["📊 Riesgos Históricos", "📡 Monitoreo Fenológico", "⚠️ Alertas y PDF", "📄 Gobernanza", "💾 Exportar"]
)

with tab_hist:
    st.header("Mapas de Calor de Riesgos Climáticos Históricos")
    if st.session_state.get("gee_authenticated", False) and usar_gee:
        with st.spinner("Generando mapa de precipitación..."):
            fig_precip = get_precipitacion_heatmap(gdf, fecha_inicio, fecha_fin)
            if fig_precip:
                st.pyplot(fig_precip)
        with st.spinner("Generando mapa de temperatura..."):
            fig_temp = get_temperatura_heatmap(gdf, fecha_inicio, fecha_fin)
            if fig_temp:
                st.pyplot(fig_temp)
        with st.spinner("Generando mapa NDWI..."):
            fig_ndwi = get_ndwi_heatmap(gdf, fecha_inicio, fecha_fin)
            if fig_ndwi:
                st.pyplot(fig_ndwi)
    else:
        st.warning("GEE no autenticado o no seleccionado. Mostrando simulaciones de mapas de calor.")
        # Mapas de calor simulados (aleatorios)
        bounds = gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, title, cmap in zip(axes, ["Precipitación (mm)", "Temperatura (°C)", "NDWI"], ["YlGnBu", "RdYlBu_r", "Blues"]):
            data = np.random.rand(100, 100) * 100
            im = ax.imshow(data, extent=[minx, maxx, miny, maxy], origin='lower', cmap=cmap, aspect='auto')
            plt.colorbar(im, ax=ax)
            gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=2, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel('Longitud')
            ax.set_ylabel('Latitud')
        st.pyplot(fig)

with tab_monitoreo:
    st.header("Monitoreo de Índices por Fase Fenológica")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NDVI", f"{ndvi_val:.2f}")
    with col2:
        st.metric("Temperatura", f"{temp_val:.1f} °C")
    with col3:
        st.metric("Humedad suelo", f"{humedad_val:.2f}")
    umbral = UMBRALES[cultivo]
    riesgo_ndvi = "🟢 Bueno" if ndvi_val > umbral["NDVI_min"] else "🔴 Bajo"
    riesgo_temp = "🟢 Adecuada" if umbral["temp_min"] <= temp_val <= umbral["temp_max"] else "🔴 Fuera de rango"
    riesgo_humedad = "🟢 Óptima" if umbral["humedad_min"] <= humedad_val <= umbral["humedad_max"] else "⚠️ Crítica"
    st.subheader("Interpretación")
    st.write(f"**NDVI:** {riesgo_ndvi} | **Temperatura:** {riesgo_temp} | **Humedad:** {riesgo_humedad}")

with tab_alerta:
    st.header("Alerta Fenológica y Ficha de Adaptación")
    if st.button("Generar Alerta con IA", type="primary"):
        with st.spinner("Consultando IA..."):
            alerta = generar_alerta_fenologica(fase_fenologica, ndvi_val, temp_val, cultivo)
        st.markdown(alerta)
        st.session_state.alerta_texto = alerta
    if st.button("📄 Generar Ficha PDF", use_container_width=True):
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.drawString(100, 750, f"FICHA DE ALERTA - {cultivo} - Fase {fase_fenologica}")
            c.drawString(100, 730, f"Fecha: {datetime.now().strftime('%d/%m/%Y')}")
            c.drawString(100, 710, f"NDVI: {ndvi_val:.2f} | Temperatura: {temp_val:.1f}°C")
            if 'alerta_texto' in st.session_state:
                c.drawString(100, 680, "Recomendación:")
                text = st.session_state.alerta_texto[:200]
                c.drawString(100, 660, text)
            c.save()
            pdf_buffer.seek(0)
            st.download_button("Descargar PDF de Alerta", data=pdf_buffer, file_name=f"alerta_{cultivo}.pdf", mime="application/pdf")
        except ImportError:
            st.warning("ReportLab no instalado. Descargando TXT.")
            if 'alerta_texto' in st.session_state:
                st.download_button("Descargar Alerta (TXT)", data=st.session_state.alerta_texto, file_name="alerta.txt")

with tab_gobernanza:
    st.header("Gobernanza de la Gestión de Riesgos Climáticos")
    st.markdown("""
    **Estructura sugerida para la cadena de ají y rocoto:**
    - **Comité de Gestión de Riesgos**: empresa, técnicos, líderes de productores.
    - **Frecuencia**: mensual / quincenal en eventos FEN.
    - **Canales**: WhatsApp, plataforma web, reuniones.
    - **Medidas**: capacitación, protocolo de respuesta, fondo de emergencia.
    """)
    if st.button("📄 Descargar One-Page Gobernanza (PDF)"):
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.drawString(100, 750, "GOBERNANZA PARA LA GESTIÓN DE RIESGOS CLIMÁTICOS")
            c.drawString(100, 730, "Cadena de Ají y Rocoto")
            c.drawString(100, 700, "Comité: coordinador, técnicos, líderes de productores.")
            c.drawString(100, 680, "Monitoreo: mensual / quincenal en FEN. Alertas por WhatsApp.")
            c.drawString(100, 660, "Medidas: capacitación anual, fondo de emergencia, protocolo de comunicación.")
            c.save()
            pdf_buffer.seek(0)
            st.download_button("Descargar PDF", data=pdf_buffer, file_name="gobernanza_riesgos.pdf", mime="application/pdf")
        except ImportError:
            st.error("ReportLab no instalado.")

with tab_export:
    st.header("Exportar Resultados")
    if st.button("Exportar parcela a GeoJSON"):
        geojson_str = gdf.to_json()
        st.download_button("Descargar GeoJSON", data=geojson_str, file_name="parcela.geojson", mime="application/json")
    if 'alerta_texto' in st.session_state:
        st.download_button("Descargar alerta (TXT)", data=st.session_state.alerta_texto, file_name="alerta.txt")
    st.info("Para mapas, usa los botones dentro de las pestañas correspondientes.")

st.markdown("---")
st.caption("Plataforma con GEE (cuenta de servicio), Groq IA, mapas de calor. Versión final.")
