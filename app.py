# app.py - Plataforma de Gestión de Riesgos Climáticos para Ají y Rocoto
# Versión con mapa único, cambio de fondo (Google Hybrid / Esri Satellite) y ZOOM AUTOMÁTICO garantizado.

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import tempfile
import os
import zipfile
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
import xml.etree.ElementTree as ET
from io import BytesIO
from shapely.geometry import Polygon

# ================= DEPENDENCIAS OPCIONALES =================
try:
    import folium
    from folium.plugins import Fullscreen
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

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
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.warning("⚠️ groq no instalado. Ejecuta: pip install groq")

# ================= LECTURA DE SECRETS =================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
if GROQ_API_KEY and GROQ_AVAILABLE:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ================= INICIALIZACIÓN DE GEE =================
def inicializar_gee():
    if not GEE_AVAILABLE:
        return False
    if 'gee_service_account' in st.secrets:
        try:
            creds = st.secrets["gee_service_account"]
            credentials = ee.ServiceAccountCredentials(
                creds['client_email'],
                key_data=creds['private_key']
            )
            ee.Initialize(credentials, project=creds.get('project_id', 'democultivos'))
            st.session_state.gee_authenticated = True
            return True
        except Exception as e:
            st.error(f"❌ Error con cuenta de servicio: {e}")
    try:
        ee.Initialize(project='applied-oxygen-459415-e2')
        st.session_state.gee_authenticated = True
        return True
    except Exception as e:
        st.session_state.gee_authenticated = False
        st.error(f"❌ Error autenticando GEE: {e}")
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
            geom_unida = gdf.unary_union
            gdf_unido = gpd.GeoDataFrame({'geometry': [geom_unida]}, crs='EPSG:4326')
            st.info(f"✅ Se unieron {len(gdf)} polígonos.")
            return gdf_unido
        return None
    except Exception as e:
        st.error(f"❌ Error cargando archivo: {e}")
        return None

# ================= FUNCIONES PARA MAPA CON ZOOM AUTOMÁTICO =================
def obtener_zoom_automatico(bounds):
    """
    Calcula un nivel de zoom apropiado para Folium basado en el bounding box (minx, miny, maxx, maxy).
    """
    minx, miny, maxx, maxy = bounds
    lat_diff = maxy - miny
    lon_diff = maxx - minx
    # Área en grados cuadrados (aproximación)
    area_deg = lat_diff * lon_diff
    # Heurística mejorada
    if area_deg < 0.00001:   # muy pequeño
        zoom = 19
    elif area_deg < 0.0001:
        zoom = 18
    elif area_deg < 0.001:
        zoom = 16
    elif area_deg < 0.01:
        zoom = 14
    elif area_deg < 0.1:
        zoom = 12
    elif area_deg < 1:
        zoom = 10
    else:
        zoom = 8
    # Limitar entre 7 y 19
    zoom = max(7, min(19, zoom))
    return zoom

def crear_mapa_con_zoom_automatico(gdf, basemap='google_hybrid'):
    """
    Crea un mapa Folium centrado en el polígono con zoom calculado automáticamente.
    """
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    zoom_start = obtener_zoom_automatico(bounds)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, control_scale=True)
    
    # Añadir el polígono de la parcela
    folium.GeoJson(
        gdf.__geo_interface__,
        name='Parcela',
        style_function=lambda x: {'color': 'yellow', 'weight': 3, 'fillOpacity': 0.1}
    ).add_to(m)
    
    # Configurar el basemap
    if basemap == 'google_hybrid':
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Hybrid',
            overlay=False,
            control=True
        ).add_to(m)
    elif basemap == 'esri_satellite':
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True
        ).add_to(m)
    else:
        folium.TileLayer('openstreetmap', name='OSM').add_to(m)
    
    return m

def obtener_tile_url_gee(image, vis_params):
    try:
        map_id = image.getMapId(vis_params)
        return map_id['tile_fetcher'].url_format
    except Exception as e:
        st.warning(f"Error generando tile URL: {e}")
        return None

def agregar_capa_gee(m, image, vis_params, nombre_capa):
    tile_url = obtener_tile_url_gee(image, vis_params)
    if tile_url:
        folium.TileLayer(
            tiles=tile_url,
            attr='Google Earth Engine',
            name=nombre_capa,
            overlay=True,
            control=True
        ).add_to(m)
        return True
    return False

# Funciones para obtener imágenes EE
def get_ndvi_image(gdf, fecha):
    region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filterDate(fecha.strftime('%Y-%m-%d'), (fecha + timedelta(days=30)).strftime('%Y-%m-%d'))
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
           .sort('CLOUDY_PIXEL_PERCENTAGE'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(region)
               .filterDate((fecha - timedelta(days=60)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
               .sort('CLOUDY_PIXEL_PERCENTAGE'))
    return col.first().normalizedDifference(['B8', 'B4']).clip(region)

def get_ndre_image(gdf, fecha):
    region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filterDate(fecha.strftime('%Y-%m-%d'), (fecha + timedelta(days=30)).strftime('%Y-%m-%d'))
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
           .sort('CLOUDY_PIXEL_PERCENTAGE'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(region)
               .filterDate((fecha - timedelta(days=60)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
               .sort('CLOUDY_PIXEL_PERCENTAGE'))
    return col.first().normalizedDifference(['B8A', 'B5']).clip(region)

def get_ndwi_image(gdf, fecha):
    region = ee.Geometry.Rectangle(gdf.total_bounds.tolist())
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filterDate(fecha.strftime('%Y-%m-%d'), (fecha + timedelta(days=30)).strftime('%Y-%m-%d'))
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
           .sort('CLOUDY_PIXEL_PERCENTAGE'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(region)
               .filterDate((fecha - timedelta(days=60)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
               .sort('CLOUDY_PIXEL_PERCENTAGE'))
    return col.first().normalizedDifference(['B3', 'B8']).clip(region)

def get_temperature_image(gdf, fecha):
    bounds = gdf.total_bounds
    delta = 0.5
    region_ampliada = ee.Geometry.Rectangle([
        bounds[0] - delta, bounds[1] - delta,
        bounds[2] + delta, bounds[3] + delta
    ])
    col = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
           .filterBounds(region_ampliada)
           .filterDate((fecha - timedelta(days=10)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
           .select('temperature_2m'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
               .filterBounds(region_ampliada)
               .filterDate((fecha - timedelta(days=30)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .select('temperature_2m'))
    temp_k = col.mean().select('temperature_2m')
    temp_c = temp_k.subtract(273.15).clip(region_ampliada)
    stats = temp_c.reduceRegion(reducer=ee.Reducer.minMax(), geometry=region_ampliada, scale=11132, maxPixels=1e9).getInfo()
    t_min = stats.get('temperature_2m_min', 5)
    t_max = stats.get('temperature_2m_max', 35)
    if t_min is None: t_min = 5
    if t_max is None: t_max = 35
    return temp_c, {'min': float(t_min), 'max': float(t_max), 'palette': ['#313695','#4575b4','#74add1','#abd9e9','#e0f3f8','#ffffbf','#fee090','#fdae61','#f46d43','#d73027','#a50026']}

def get_precipitation_image(gdf, fecha):
    bounds = gdf.total_bounds
    delta = 1.0
    region_ampliada = ee.Geometry.Rectangle([
        bounds[0] - delta, bounds[1] - delta,
        bounds[2] + delta, bounds[3] + delta
    ])
    col = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
           .filterBounds(region_ampliada)
           .filterDate((fecha - timedelta(days=30)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
           .select('precipitation'))
    if col.size().getInfo() == 0:
        col = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
               .filterBounds(region_ampliada)
               .filterDate((fecha - timedelta(days=60)).strftime('%Y-%m-%d'), fecha.strftime('%Y-%m-%d'))
               .select('precipitation'))
    img = col.sort('system:time_start', False).first().clip(region_ampliada)
    stats = img.reduceRegion(reducer=ee.Reducer.max(), geometry=region_ampliada, scale=5566, maxPixels=1e9).getInfo()
    p_max = stats.get('precipitation_max', 1.0)
    p_max = float(p_max) if p_max else 1.0
    vis_max = max(round(p_max * 1.1, 1), 1.0)
    return img, {'min': 0, 'max': vis_max, 'palette': ['#f0f9e8', '#bae4bc', '#7bccc4', '#2b8cbe', '#084081']}

# ================= FUNCIONES DE IA =================
def consultar_groq(prompt, max_tokens=600, model="llama-3.3-70b-versatile"):
    if not GROQ_API_KEY or not GROQ_AVAILABLE:
        return "⚠️ IA no disponible."
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

def generar_alerta_detallada(fase, ndvi, temp, precip_actual, humedad, cultivo, umbrales):
    prompt = f"""
Eres un agrónomo experto en {cultivo}. Genera una alerta agronómica detallada usando estos datos:

- Fase fenológica: {fase}
- NDVI actual: {ndvi:.2f} (umbral mínimo {umbrales['NDVI_min']:.2f})
- Temperatura: {temp:.1f}°C (rango óptimo {umbrales['temp_min']:.0f}-{umbrales['temp_max']:.0f}°C)
- Precipitación reciente (mm): {precip_actual:.1f}
- Humedad del suelo (índice SAR): {humedad:.2f} (rango óptimo {umbrales['humedad_min']:.2f}-{umbrales['humedad_max']:.2f})

Instrucciones:
1. Evalúa el nivel de riesgo para esta fase (CRÍTICO / ALTO / MEDIO / BAJO).
2. Explica las causas principales (estrés hídrico, térmico, nutricional, etc.).
3. Proporciona 3 recomendaciones concretas.
4. Si hay riesgo de helada o golpe de calor, menciónalo.
5. Formato claro, máximo 250 palabras.
"""
    return consultar_groq(prompt, max_tokens=600)

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
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    cultivo = st.selectbox("Cultivo", CULTIVOS)
    st.info(f"{ICONOS[cultivo]} Parámetros específicos cargados.")
    uploaded_file = st.file_uploader("Subir parcela (GeoJSON, KML, KMZ, ZIP Shapefile)", type=['geojson','kml','kmz','zip'])
    fecha_fin = st.date_input("Fecha fin", datetime.now())
    fecha_inicio = st.date_input("Fecha inicio", datetime.now() - timedelta(days=90))
    fase_fenologica = st.selectbox("Fase actual del cultivo", ["siembra", "desarrollo", "floracion", "fructificacion", "cosecha"])
    usar_gee = st.checkbox("Usar GEE (si autenticado)", value=True)
    st.markdown("---")
    st.caption("📊 Datos satelitales: Sentinel-2, CHIRPS, ERA5-Land")
    gee_ok = st.session_state.get('gee_authenticated', False)
    st.caption(f"GEE: {'✅ Autenticado' if gee_ok else '❌ No autenticado'}")
    if st.button("🔄 Reintentar autenticación GEE"):
        inicializar_gee()
        st.rerun()

if not uploaded_file:
    st.info("👈 Sube un archivo de parcela para comenzar el análisis.")
    st.stop()

# Cargar parcela
with st.spinner("Cargando parcela..."):
    gdf = cargar_archivo_parcela(uploaded_file)
    if gdf is None:
        st.error("No se pudo cargar la parcela.")
        st.stop()
    area_ha = calcular_superficie(gdf)
    st.success(f"✅ Parcela cargada: {area_ha:.2f} ha, CRS EPSG:4326")

# Datos simulados (se reemplazarán si hay GEE)
ndvi_val = np.random.uniform(0.3, 0.8)
temp_val = np.random.uniform(15, 32)
humedad_val = np.random.uniform(0.2, 0.7)
precip_actual = np.random.uniform(0, 20)

df_ndvi = pd.DataFrame()
df_precip = pd.DataFrame()
df_temp = pd.DataFrame()
if st.session_state.get("gee_authenticated", False) and usar_gee:
    with st.spinner("Descargando series temporales desde GEE..."):
        try:
            from agroia_gee import (
                obtener_serie_temporal_ndvi, obtener_serie_temporal_temperatura,
                obtener_serie_temporal_precipitacion
            )
            df_ndvi = obtener_serie_temporal_ndvi(gdf, fecha_inicio.strftime('%Y-%m-%d'), fecha_fin.strftime('%Y-%m-%d'))
            df_precip = obtener_serie_temporal_precipitacion(gdf, fecha_inicio.strftime('%Y-%m-%d'), fecha_fin.strftime('%Y-%m-%d'))
            df_temp = obtener_serie_temporal_temperatura(gdf, fecha_inicio.strftime('%Y-%m-%d'), fecha_fin.strftime('%Y-%m-%d'))
            if not df_ndvi.empty:
                ndvi_val = df_ndvi['ndvi'].iloc[-1]
            if not df_temp.empty:
                temp_val = df_temp['temp'].iloc[-1]
            if not df_precip.empty:
                precip_actual = df_precip['precip'].iloc[-1]
        except ImportError:
            st.info("Módulo agroia_gee no encontrado. Usando datos simulados.")
else:
    st.info("GEE no autenticado o no seleccionado. Se usan datos simulados.")

# ================= PESTAÑAS =================
tab_dashboard, tab_mapas, tab_monitoreo, tab_alerta, tab_gobernanza, tab_export = st.tabs(
    ["📊 Dashboard General", "🗺️ Mapa de Riesgo", "📈 Monitoreo Fenológico", "⚠️ Alertas IA", "📄 Gobernanza", "💾 Exportar"]
)

# ================= DASHBOARD GENERAL =================
with tab_dashboard:
    st.header("Dashboard de Indicadores Clave")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌱 NDVI actual", f"{ndvi_val:.2f}", delta=f"{ndvi_val - UMBRALES[cultivo]['NDVI_min']:.2f}" if ndvi_val > UMBRALES[cultivo]['NDVI_min'] else "crítico")
    with col2:
        st.metric("🌡️ Temperatura", f"{temp_val:.1f} °C", delta="óptima" if UMBRALES[cultivo]['temp_min'] <= temp_val <= UMBRALES[cultivo]['temp_max'] else "alerta")
    with col3:
        st.metric("💧 Humedad suelo", f"{humedad_val:.2f}", delta="normal" if UMBRALES[cultivo]['humedad_min'] <= humedad_val <= UMBRALES[cultivo]['humedad_max'] else "crítica")
    with col4:
        st.metric("📅 Fase fenológica", fase_fenologica.capitalize())
    
    st.subheader("Evolución de Índices")
    if not df_ndvi.empty and not df_temp.empty and not df_precip.empty:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(df_ndvi['date'], df_ndvi['ndvi'], 'g-', linewidth=2)
        axes[0].axhline(UMBRALES[cultivo]['NDVI_min'], color='red', linestyle='--')
        axes[0].set_ylabel('NDVI')
        axes[1].plot(df_temp['date'], df_temp['temp'], 'r-')
        axes[1].axhline(UMBRALES[cultivo]['temp_min'], color='blue', linestyle='--')
        axes[1].axhline(UMBRALES[cultivo]['temp_max'], color='orange', linestyle='--')
        axes[1].set_ylabel('Temperatura (°C)')
        axes[2].bar(df_precip['date'], df_precip['precip'], color='cyan')
        axes[2].set_ylabel('Precipitación (mm)')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Datos históricos insuficientes. Se muestran simulaciones.")
        fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
        ndvi_sim = np.random.uniform(0.3, 0.8, len(fechas))
        temp_sim = np.random.uniform(15, 32, len(fechas))
        precip_sim = np.random.exponential(5, len(fechas))
        fig, axes = plt.subplots(3,1,figsize=(12,10))
        axes[0].plot(fechas, ndvi_sim, 'g-')
        axes[0].set_ylabel('NDVI sim')
        axes[1].plot(fechas, temp_sim, 'r-')
        axes[1].set_ylabel('Temp sim')
        axes[2].bar(fechas, precip_sim, color='cyan')
        axes[2].set_ylabel('Precip sim')
        st.pyplot(fig)

# ================= MAPA DE RIESGO CON ZOOM AUTOMÁTICO =================
with tab_mapas:
    st.header("🗺️ Mapa de Riesgo Climático Interactivo")
    st.markdown("El mapa se centra automáticamente en tu parcela. Selecciona el índice y el fondo deseado.")
    
    if not (st.session_state.get("gee_authenticated", False) and usar_gee and FOLIUM_OK and FOLIUM_STATIC_OK):
        st.warning("⚠️ Se requiere GEE autenticado, folium y streamlit-folium. Instala: pip install folium streamlit-folium")
        st.stop()
    
    indice = st.selectbox("Elige el índice a visualizar", ["NDVI", "NDRE", "NDWI", "Temperatura", "Precipitación"])
    fondo = st.radio("Fondo del mapa", ["Google Hybrid", "Esri Satellite"], horizontal=True)
    basemap = 'google_hybrid' if fondo == "Google Hybrid" else 'esri_satellite'
    
    with st.spinner(f"Obteniendo datos de {indice} desde GEE... Esto puede tomar unos segundos."):
        if indice == "NDVI":
            image = get_ndvi_image(gdf, fecha_fin)
            vis = {'min': -0.2, 'max': 0.8, 'palette': ['red', 'yellow', 'green']}
            nombre_capa = "NDVI"
        elif indice == "NDRE":
            image = get_ndre_image(gdf, fecha_fin)
            vis = {'min': -0.2, 'max': 0.8, 'palette': ['red', 'yellow', 'green']}
            nombre_capa = "NDRE"
        elif indice == "NDWI":
            image = get_ndwi_image(gdf, fecha_fin)
            vis = {'min': -0.5, 'max': 0.5, 'palette': ['brown', 'white', 'blue']}
            nombre_capa = "NDWI"
        elif indice == "Temperatura":
            image, vis = get_temperature_image(gdf, fecha_fin)
            nombre_capa = "Temperatura (°C)"
        elif indice == "Precipitación":
            image, vis = get_precipitation_image(gdf, fecha_fin)
            nombre_capa = "Precipitación (mm)"
    
    # Crear mapa con zoom automático
    mapa = crear_mapa_con_zoom_automatico(gdf, basemap=basemap)
    success = agregar_capa_gee(mapa, image, vis, nombre_capa)
    if success:
        folium.LayerControl().add_to(mapa)
        folium_static(mapa, width=800, height=600)
        st.success("✅ Mapa generado con zoom automático a la parcela.")
    else:
        st.error("No se pudo agregar la capa de GEE. Verifica autenticación y conectividad.")

# ================= MONITOREO FENOLÓGICO =================
with tab_monitoreo:
    st.header("Monitoreo Detallado")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("NDVI", f"{ndvi_val:.2f}")
        st.metric("Temperatura", f"{temp_val:.1f} °C")
        st.metric("Humedad suelo", f"{humedad_val:.2f}")
        st.metric("Precipitación reciente", f"{precip_actual:.1f} mm")
    with col2:
        st.subheader("Comparativa con Umbrales")
        umbral = UMBRALES[cultivo]
        st.write(f"**NDVI:** {'🟢' if ndvi_val > umbral['NDVI_min'] else '🔴'} Mínimo {umbral['NDVI_min']}")
        st.write(f"**Temperatura:** {'🟢' if umbral['temp_min'] <= temp_val <= umbral['temp_max'] else '🔴'} Rango {umbral['temp_min']}-{umbral['temp_max']} °C")
        st.write(f"**Humedad:** {'🟢' if umbral['humedad_min'] <= humedad_val <= umbral['humedad_max'] else '🔴'} Rango {umbral['humedad_min']:.2f}-{umbral['humedad_max']:.2f}")
    
    if not df_ndvi.empty:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_ndvi['date'], df_ndvi['ndvi'], 'g-o', markersize=3)
        ax.axhline(umbral['NDVI_min'], color='red', linestyle='--')
        ax.set_ylabel('NDVI')
        st.pyplot(fig)

# ================= ALERTAS IA =================
with tab_alerta:
    if st.button("🤖 Generar Alerta Detallada", type="primary"):
        with st.spinner("Consultando IA..."):
            alerta = generar_alerta_detallada(fase_fenologica, ndvi_val, temp_val, precip_actual, humedad_val, cultivo, UMBRALES[cultivo])
        st.markdown(alerta)
        st.download_button("Descargar alerta", data=alerta, file_name=f"alerta_{cultivo}.txt")

# ================= GOBERNANZA Y EXPORTACIÓN =================
with tab_gobernanza:
    st.markdown("""
    **Estructura sugerida para la cadena de ají y rocoto:**
    - Comité de Gestión de Riesgos
    - Frecuencia de monitoreo: mensual / quincenal
    - Canales: WhatsApp, plataforma web
    - Medidas: capacitación, fondo de emergencia
    """)
    if st.button("Descargar Gobernanza (PDF)"):
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.drawString(100, 750, "GOBERNANZA PARA RIESGOS CLIMÁTICOS")
            c.drawString(100, 730, "Cadena de Ají y Rocoto")
            c.save()
            pdf_buffer.seek(0)
            st.download_button("PDF", data=pdf_buffer, file_name="gobernanza.pdf")
        except ImportError:
            st.error("Instalar reportlab")

with tab_export:
    if st.button("Exportar parcela a GeoJSON"):
        st.download_button("Descargar", data=gdf.to_json(), file_name="parcela.geojson")
    if not df_ndvi.empty:
        st.download_button("Serie NDVI CSV", data=df_ndvi.to_csv(index=False), file_name="ndvi.csv")

st.caption("Versión con zoom automático garantizado. El mapa se centra en tu polígono al cargar.")
