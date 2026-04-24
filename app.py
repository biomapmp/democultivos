# app.py - Plataforma de Gestión de Riesgos Climáticos para Ají y Rocoto
# Incluye visualización NDVI y NDRE (con GEE o simulado)

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
from PIL import Image

# ================= CONFIGURACIÓN INICIAL =================
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

# Intentar importar dependencias opcionales
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
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
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

# Google Earth Engine
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    st.warning("⚠️ earthengine-api no instalado.")

# Groq IA
try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.warning("⚠️ groq no instalado. La IA no estará disponible.")

# ================= CONFIGURACIÓN DE CLAVES =================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
if GROQ_API_KEY and GROQ_AVAILABLE:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    st.success("✅ API Key de Groq cargada.")
else:
    st.warning("⚠️ No se encontró API Key de Groq o librería no instalada. La IA no estará disponible.")

# ================= INICIALIZACIÓN GEE =================
def inicializar_gee():
    if not GEE_AVAILABLE:
        return False
    try:
        gee_secret = os.environ.get('GEE_SERVICE_ACCOUNT')
        if gee_secret:
            creds = json.loads(gee_secret)
            credentials = ee.ServiceAccountCredentials(creds['client_email'], key_data=json.dumps(creds))
            ee.Initialize(credentials, project=creds.get('project_id', 'democultivos'))
            st.session_state.gee_authenticated = True
            st.success("✅ GEE autenticado con cuenta de servicio.")
            return True
        ee.Initialize()
        st.session_state.gee_authenticated = True
        st.success("✅ GEE autenticado localmente.")
        return True
    except Exception as e:
        st.session_state.gee_authenticated = False
        st.error(f"❌ Error autenticando GEE: {str(e)}")
        return False

if 'gee_authenticated' not in st.session_state:
    st.session_state.gee_authenticated = False
    if GEE_AVAILABLE:
        inicializar_gee()

# ================= PARÁMETROS DE CULTIVOS =================
CULTIVOS = ["AJÍ", "ROCOTO", "PAPA ANDINA"]
ICONOS = {"AJÍ": "🌶️", "ROCOTO": "🥵", "PAPA ANDINA": "🥔"}

UMBRALES = {
    "AJÍ": {
        "NDVI_min": 0.4, "NDVI_opt": 0.7,
        "temp_min": 18, "temp_max": 30,
        "humedad_suelo_min": 0.25, "humedad_suelo_max": 0.65,
    },
    "ROCOTO": {
        "NDVI_min": 0.45, "NDVI_opt": 0.75,
        "temp_min": 16, "temp_max": 28,
        "humedad_suelo_min": 0.30, "humedad_suelo_max": 0.70,
    },
    "PAPA ANDINA": {
        "NDVI_min": 0.5, "NDVI_opt": 0.8,
        "temp_min": 10, "temp_max": 22,
        "humedad_suelo_min": 0.35, "humedad_suelo_max": 0.75,
    }
}

# ================= FUNCIONES DE CARGA DE PARCELA =================
def validar_crs(gdf):
    if gdf is None or len(gdf) == 0:
        return gdf
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326', inplace=False)
            st.info("ℹ️ Se asignó EPSG:4326 al archivo (no tenía CRS)")
        elif str(gdf.crs).upper() != 'EPSG:4326':
            original_crs = str(gdf.crs)
            gdf = gdf.to_crs('EPSG:4326')
            st.info(f"ℹ️ Transformado de {original_crs} a EPSG:4326")
        return gdf
    except Exception as e:
        st.warning(f"⚠️ Error al corregir CRS: {str(e)}")
        return gdf

def calcular_superficie(gdf):
    try:
        if gdf is None or len(gdf) == 0:
            return 0.0
        gdf = validar_crs(gdf)
        gdf_projected = gdf.to_crs('EPSG:3857')
        area_m2 = gdf_projected.geometry.area.sum()
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
                st.error("❌ No se encontró ningún archivo .shp en el ZIP")
                return None
    except Exception as e:
        st.error(f"❌ Error cargando shapefile desde ZIP: {str(e)}")
        return None

def parsear_kml_manual(contenido_kml):
    try:
        root = ET.fromstring(contenido_kml)
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        polygons = []
        for polygon_elem in root.findall('.//kml:Polygon', namespaces):
            coords_elem = polygon_elem.find('.//kml:coordinates', namespaces)
            if coords_elem is not None and coords_elem.text:
                coord_text = coords_elem.text.strip()
                coord_list = []
                for coord_pair in coord_text.split():
                    parts = coord_pair.split(',')
                    if len(parts) >= 2:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        coord_list.append((lon, lat))
                if len(coord_list) >= 3:
                    polygons.append(Polygon(coord_list))
        if not polygons:
            for multi_geom in root.findall('.//kml:MultiGeometry', namespaces):
                for polygon_elem in multi_geom.findall('.//kml:Polygon', namespaces):
                    coords_elem = polygon_elem.find('.//kml:coordinates', namespaces)
                    if coords_elem is not None and coords_elem.text:
                        coord_text = coords_elem.text.strip()
                        coord_list = []
                        for coord_pair in coord_text.split():
                            parts = coord_pair.split(',')
                            if len(parts) >= 2:
                                lon = float(parts[0])
                                lat = float(parts[1])
                                coord_list.append((lon, lat))
                        if len(coord_list) >= 3:
                            polygons.append(Polygon(coord_list))
        if polygons:
            gdf = gpd.GeoDataFrame({'geometry': polygons}, crs='EPSG:4326')
            return gdf
        else:
            for placemark in root.findall('.//kml:Placemark', namespaces):
                for elem_name in ['Polygon', 'LineString', 'Point', 'LinearRing']:
                    elem = placemark.find(f'.//kml:{elem_name}', namespaces)
                    if elem is not None:
                        coords_elem = elem.find('.//kml:coordinates', namespaces)
                        if coords_elem is not None and coords_elem.text:
                            coord_text = coords_elem.text.strip()
                            coord_list = []
                            for coord_pair in coord_text.split():
                                parts = coord_pair.split(',')
                                if len(parts) >= 2:
                                    lon = float(parts[0])
                                    lat = float(parts[1])
                                    coord_list.append((lon, lat))
                            if len(coord_list) >= 3:
                                polygons.append(Polygon(coord_list))
                            break
        if polygons:
            gdf = gpd.GeoDataFrame({'geometry': polygons}, crs='EPSG:4326')
            return gdf
        return None
    except Exception as e:
        st.error(f"❌ Error parseando KML manualmente: {str(e)}")
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
                        try:
                            gdf = gpd.read_file(kml_path)
                            gdf = validar_crs(gdf)
                            return gdf
                        except:
                            st.error("❌ No se pudo cargar el archivo KML/KMZ")
                            return None
                else:
                    st.error("❌ No se encontró ningún archivo .kml en el KMZ")
                    return None
        else:
            contenido = kml_file.read().decode('utf-8')
            gdf = parsear_kml_manual(contenido)
            if gdf is not None:
                return gdf
            else:
                kml_file.seek(0)
                gdf = gpd.read_file(kml_file)
                gdf = validar_crs(gdf)
                return gdf
    except Exception as e:
        st.error(f"❌ Error cargando archivo KML/KMZ: {str(e)}")
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
            st.error("❌ Formato de archivo no soportado. Use ZIP (Shapefile), KML, KMZ o GeoJSON.")
            return None
        
        if gdf is not None:
            gdf = validar_crs(gdf)
            gdf = gdf.explode(ignore_index=True)
            gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
            if len(gdf) == 0:
                st.error("❌ No se encontraron polígonos en el archivo")
                return None
            geometria_unida = gdf.unary_union
            gdf_unido = gpd.GeoDataFrame([{'geometry': geometria_unida}], crs='EPSG:4326')
            gdf_unido = validar_crs(gdf_unido)
            st.info(f"✅ Se unieron {len(gdf)} polígono(s) en una sola geometría.")
            gdf_unido['id_zona'] = 1
            return gdf_unido
        return gdf
    except Exception as e:
        st.error(f"❌ Error cargando archivo: {str(e)}")
        import traceback
        st.error(f"Detalle: {traceback.format_exc()}")
        return None

# ================= FUNCIONES DE VISUALIZACIÓN NDVI/NDRE =================
def visualizar_indices_gee_estatico(gdf, satelite, fecha_inicio, fecha_fin):
    """Retorna diccionario con imágenes NDVI y NDRE desde GEE"""
    if not GEE_AVAILABLE or not st.session_state.gee_authenticated:
        return None, "GEE no autenticado"
    
    try:
        bounds = gdf.total_bounds
        min_lon, min_lat, max_lon, max_lat = bounds
        geometry = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
        start_date = fecha_inicio.strftime('%Y-%m-%d')
        end_date = fecha_fin.strftime('%Y-%m-%d')
        
        if 'SENTINEL' in satelite.upper():
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            ndvi_bands = ['B8', 'B4']
            ndre_bands = ['B8', 'B5']
            title = "Sentinel-2"
        elif 'LANDSAT' in satelite.upper():
            collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            ndvi_bands = ['SR_B5', 'SR_B4']
            ndre_bands = ['SR_B5', 'SR_B6']
            title = "Landsat"
        else:
            return None, "Satélite no soportado"
        
        filtered = (collection
                   .filterBounds(geometry)
                   .filterDate(start_date, end_date)
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)))
        
        count = filtered.size().getInfo()
        if count == 0:
            return None, "No hay imágenes disponibles"
        
        image = filtered.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        ndvi = image.normalizedDifference(ndvi_bands).rename('NDVI')
        ndre = image.normalizedDifference(ndre_bands).rename('NDRE')
        
        region_params = {'dimensions': 800, 'region': geometry, 'format': 'png'}
        ndvi_url = ndvi.getThumbURL({'min': -0.2, 'max': 0.8, 'palette': ['red', 'yellow', 'green'], **region_params})
        ndre_url = ndre.getThumbURL({'min': -0.1, 'max': 0.6, 'palette': ['blue', 'white', 'green'], **region_params})
        
        import requests
        ndvi_resp = requests.get(ndvi_url)
        ndre_resp = requests.get(ndre_url)
        if ndvi_resp.status_code != 200 or ndre_resp.status_code != 200:
            return None, "Error descargando imágenes"
        
        ndvi_bytes = BytesIO(ndvi_resp.content)
        ndre_bytes = BytesIO(ndre_resp.content)
        
        return {
            'ndvi_bytes': ndvi_bytes,
            'ndre_bytes': ndre_bytes,
            'title': title,
            'cloud_percent': image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo() if image.get('CLOUDY_PIXEL_PERCENTAGE') else 0,
        }, "Imágenes generadas correctamente"
    except Exception as e:
        return None, f"Error: {str(e)}"

def generar_imagen_ndvi_simulada(gdf, ancho=800, alto=800):
    """Genera una imagen simulada de NDVI (para cuando no hay GEE)"""
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    # Crear una imagen de ruido aleatorio con forma de polígono
    x = np.linspace(minx, maxx, ancho)
    y = np.linspace(miny, maxy, alto)
    X, Y = np.meshgrid(x, y)
    # Simular valores NDVI entre 0.2 y 0.8
    ndvi = 0.3 + 0.5 * np.random.rand(alto, ancho)
    # Aplicar máscara del polígono
    points = np.vstack([X.ravel(), Y.ravel()]).T
    from shapely.geometry import Point
    mask = gdf.geometry.unary_union.contains([Point(p) for p in points])
    mask = mask.reshape(alto, ancho)
    ndvi[~mask] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ndvi, extent=[minx, maxx, miny, maxy], origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='NDVI')
    gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=2)
    ax.set_title('NDVI Simulado')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    return buf

def exportar_mapa_tiff(imagen_bytes, gdf, nombre_base):
    """Convierte una imagen PNG a GeoTIFF (requiere rasterio)"""
    if not RASTERIO_OK:
        return None, None
    try:
        img = Image.open(imagen_bytes)
        gdf_proj = gdf.to_crs(epsg=3857)
        bounds = gdf_proj.total_bounds
        width, height = img.size
        transform = from_origin(bounds[0], bounds[3], (bounds[2]-bounds[0])/width, (bounds[3]-bounds[1])/height)
        img_array = np.array(img)
        if img_array.shape[2] == 4:
            img_array = img_array[:,:,:3]
        img_array = np.transpose(img_array, (2,0,1))
        tiff_buf = BytesIO()
        with rasterio.open(tiff_buf, 'w', driver='GTiff', height=height, width=width, count=3, dtype=img_array.dtype, crs=CRS.from_epsg(3857), transform=transform, compress='lzw') as dst:
            dst.write(img_array)
        tiff_buf.seek(0)
        return tiff_buf, f"{nombre_base}.tiff"
    except Exception as e:
        st.error(f"Error exportando a TIFF: {e}")
        return None, None

# ================= FUNCIONES DE IA =================
def consultar_groq(prompt, max_tokens=400):
    if not GROQ_API_KEY or not GROQ_AVAILABLE:
        return "⚠️ IA no disponible: falta API Key o librería."
    try:
        client = groq.Client(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error consultando Groq: {str(e)}"

def generar_alerta_fenologica(fase, ndvi, temp, cultivo):
    prompt = f"""
Eres un agrónomo experto en {cultivo}. El cultivo está en fase de {fase}.
Valores actuales:
- NDVI: {ndvi:.2f}
- Temperatura: {temp:.1f}°C

Genera un análisis de riesgo (bajo/medio/alto) para esta fase y una acción de adaptación concreta (máximo 40 palabras). Usa formato: **Riesgo:** ... **Acción:** ...
"""
    return consultar_groq(prompt, max_tokens=200)

# ================= INTERFAZ PRINCIPAL =================
st.set_page_config(page_title="Gestión de Riesgos Climáticos - Ají y Rocoto", layout="wide")
st.title("🌶️ Plataforma de Gestión de Riesgos Climáticos para Ají y Rocoto")

with st.sidebar:
    st.header("⚙️ Configuración")
    cultivo = st.selectbox("Cultivo", CULTIVOS)
    st.info(f"{ICONOS[cultivo]} Parámetros específicos cargados.")
    
    uploaded_file = st.file_uploader("Subir parcela (GeoJSON, KML, KMZ, ZIP Shapefile)", 
                                     type=['geojson','kml','kmz','zip'])
    
    st.subheader("📅 Período de análisis")
    fecha_fin = st.date_input("Fecha fin", datetime.now())
    fecha_inicio = st.date_input("Fecha inicio", datetime.now() - timedelta(days=30))
    
    st.subheader("🌿 Fenología")
    fase_fenologica = st.selectbox("Fase actual del cultivo", 
                                   ["siembra", "desarrollo", "floracion", "fructificacion", "cosecha"])
    
    st.subheader("🛰️ Fuente de datos")
    usar_gee = st.checkbox("Usar GEE (si autenticado)", value=True)
    
    st.subheader("📡 Datos in situ (opcional)")
    archivo_estacion = st.file_uploader("Subir CSV de estación", type=['csv'])

if not uploaded_file:
    st.info("👈 Sube un archivo de parcela para comenzar el análisis.")
    st.stop()

# Cargar parcela
with st.spinner("Cargando parcela..."):
    gdf = cargar_archivo_parcela(uploaded_file)
    if gdf is None:
        st.error("No se pudo cargar la parcela. Verifica el formato del archivo.")
        st.stop()
    area_ha = calcular_superficie(gdf)
    st.success(f"✅ Parcela cargada: {area_ha:.2f} ha. CRS: {gdf.crs}")

# ================= SIMULACIÓN DE DATOS SATELITALES =================
# Si GEE está autenticado y se desea usar, se pueden obtener índices reales. Por simplicidad, simulamos.
ndvi_val = np.random.uniform(0.3, 0.8)
temp_val = np.random.uniform(15, 32)
humedad_val = np.random.uniform(0.2, 0.7)

# ================= PESTAÑAS =================
tab_hist, tab_monitoreo, tab_alerta, tab_gobernanza, tab_export = st.tabs(
    ["📊 Riesgos Históricos", "📡 Monitoreo Fenológico", "⚠️ Alertas y PDF", "📄 Gobernanza", "💾 Exportar"]
)

with tab_hist:
    st.header("Mapa de Riesgos Climáticos Históricos")
    st.info("Visualización de índices históricos (precipitación, temperatura, NDWI) utilizando GEE.")
    if st.session_state.get("gee_authenticated", False):
        st.success("Con GEE autenticado, aquí se mostrarían mapas interactivos reales.")
        # Aquí se podría implementar la visualización histórica
    else:
        st.warning("GEE no autenticado. Mostrando datos simulados.")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(np.random.rand(100,100), cmap='Blues')
        axes[0].set_title("Precipitación (simulada)")
        axes[1].imshow(np.random.rand(100,100), cmap='RdYlBu')
        axes[1].set_title("NDWI (simulado)")
        axes[2].imshow(np.random.rand(100,100), cmap='RdYlGn')
        axes[2].set_title("Temperatura (simulada)")
        for ax in axes: ax.axis('off')
        st.pyplot(fig)

with tab_monitoreo:
    st.header("Monitoreo de Índices por Fase Fenológica")
    
    # Indicadores numéricos
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NDVI", f"{ndvi_val:.2f}")
    with col2:
        st.metric("Temperatura", f"{temp_val:.1f} °C")
    with col3:
        st.metric("Humedad suelo (SAR)", f"{humedad_val:.2f}")
    
    # Comparación con umbrales
    umbral = UMBRALES[cultivo]
    riesgo_ndvi = "🟢 Bueno" if ndvi_val > umbral["NDVI_min"] else "🔴 Bajo"
    riesgo_temp = "🟢 Adecuada" if umbral["temp_min"] <= temp_val <= umbral["temp_max"] else "🔴 Fuera de rango"
    riesgo_humedad = "🟢 Óptima" if umbral["humedad_suelo_min"] <= humedad_val <= umbral["humedad_suelo_max"] else "⚠️ Crítica"
    
    st.subheader("Interpretación automática")
    st.write(f"**NDVI:** {riesgo_ndvi}")
    st.write(f"**Temperatura:** {riesgo_temp}")
    st.write(f"**Humedad del suelo:** {riesgo_humedad}")
    
    # ============= VISUALIZACIÓN DE MAPAS NDVI Y NDRE =============
    st.subheader("🗺️ Mapas de Índices de Vegetación (NDVI y NDRE)")
    
    col_map1, col_map2 = st.columns(2)
    
    # Botón para generar mapas
    if st.button("🔄 Generar Mapas NDVI/NDRE", use_container_width=True):
        with st.spinner("Generando imágenes..."):
            if usar_gee and st.session_state.get("gee_authenticated", False):
                # Usar GEE real
                satelite = "SENTINEL-2"  # Podría ser configurable
                resultado, mensaje = visualizar_indices_gee_estatico(gdf, satelite, fecha_inicio, fecha_fin)
                if resultado:
                    st.session_state.ndvi_img = resultado['ndvi_bytes']
                    st.session_state.ndre_img = resultado['ndre_bytes']
                    st.session_state.img_title = resultado['title']
                    st.success(mensaje)
                else:
                    st.error(mensaje)
                    # Fallback a simulación
                    st.session_state.ndvi_img = generar_imagen_ndvi_simulada(gdf)
                    st.session_state.ndre_img = generar_imagen_ndvi_simulada(gdf)  # Simulada
                    st.warning("Usando imágenes simuladas por fallo de GEE")
            else:
                # Datos simulados
                st.session_state.ndvi_img = generar_imagen_ndvi_simulada(gdf)
                st.session_state.ndre_img = generar_imagen_ndvi_simulada(gdf)
                st.info("Usando datos simulados (GEE no autenticado o no seleccionado)")
    
    # Mostrar mapas si existen en sesión
    if 'ndvi_img' in st.session_state and 'ndre_img' in st.session_state:
        with col_map1:
            st.image(st.session_state.ndvi_img, caption="NDVI", use_column_width=True)
            # Botones de descarga para NDVI
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("📥 PNG", data=st.session_state.ndvi_img, file_name=f"ndvi_{cultivo}.png", mime="image/png", key="ndvi_png")
            with col_d2:
                if RASTERIO_OK:
                    tiff_buf, tiff_name = exportar_mapa_tiff(st.session_state.ndvi_img, gdf, f"ndvi_{cultivo}")
                    if tiff_buf:
                        st.download_button("📥 GeoTIFF", data=tiff_buf, file_name=tiff_name, mime="image/tiff", key="ndvi_tiff")
        with col_map2:
            st.image(st.session_state.ndre_img, caption="NDRE", use_column_width=True)
            col_d3, col_d4 = st.columns(2)
            with col_d3:
                st.download_button("📥 PNG", data=st.session_state.ndre_img, file_name=f"ndre_{cultivo}.png", mime="image/png", key="ndre_png")
            with col_d4:
                if RASTERIO_OK:
                    tiff_buf2, tiff_name2 = exportar_mapa_tiff(st.session_state.ndre_img, gdf, f"ndre_{cultivo}")
                    if tiff_buf2:
                        st.download_button("📥 GeoTIFF", data=tiff_buf2, file_name=tiff_name2, mime="image/tiff", key="ndre_tiff")
        
        # Paquete ZIP
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zf:
            zf.writestr(f"ndvi_{cultivo}.png", st.session_state.ndvi_img.getvalue())
            zf.writestr(f"ndre_{cultivo}.png", st.session_state.ndre_img.getvalue())
        zip_buf.seek(0)
        st.download_button("📦 Descargar ambos mapas (ZIP)", data=zip_buf, file_name=f"indices_{cultivo}.zip", mime="application/zip")
    
    if archivo_estacion:
        st.subheader("📊 Datos de estación in situ")
        df_est = pd.read_csv(archivo_estacion)
        st.dataframe(df_est)

with tab_alerta:
    st.header("Alerta Fenológica y Ficha de Adaptación")
    if st.button("Generar Alerta con IA", type="primary"):
        with st.spinner("Consultando IA (Groq)..."):
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
            st.warning("ReportLab no instalado. Se descargará un archivo TXT.")
            if 'alerta_texto' in st.session_state:
                st.download_button("Descargar Alerta (TXT)", data=st.session_state.alerta_texto, file_name="alerta.txt")

with tab_gobernanza:
    st.header("Gobernanza de la Gestión de Riesgos Climáticos")
    st.markdown("""
    **Estructura sugerida para la cadena de ají y rocoto:**
    
    - **Comité de Gestión de Riesgos**: integrado por representantes de la empresa, técnicos agrónomos y líderes de productores.
    - **Frecuencia de monitoreo**: mensual, con alertas quincenales durante eventos FEN.
    - **Canales de comunicación**: WhatsApp (alertas), plataforma web (dashboard), reuniones presenciales.
    - **Medidas administrativas**:
        * Capacitación en uso de la plataforma.
        * Protocolo de respuesta ante alertas.
    """)
    if st.button("📄 Descargar One-Page Gobernanza (PDF)"):
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.drawString(100, 750, "GOBERNANZA PARA LA GESTIÓN DE RIESGOS CLIMÁTICOS")
            c.drawString(100, 730, "Cadena de Ají y Rocoto")
            c.drawString(100, 700, "Comité de Gestión de Riesgos: coordinador, técnicos, líderes de productores.")
            c.drawString(100, 680, "Monitoreo: mensual / quincenal en FEN. Alertas por WhatsApp.")
            c.drawString(100, 660, "Medidas: capacitación anual, fondo de emergencia, protocolo de comunicación.")
            c.save()
            pdf_buffer.seek(0)
            st.download_button("Descargar PDF", data=pdf_buffer, file_name="gobernanza_riesgos.pdf", mime="application/pdf")
        except ImportError:
            st.error("ReportLab no instalado. No se puede generar PDF.")

with tab_export:
    st.header("Exportar Resultados")
    if st.button("Exportar parcela a GeoJSON"):
        geojson_str = gdf.to_json()
        st.download_button("Descargar GeoJSON", data=geojson_str, file_name="parcela.geojson", mime="application/json")
    if 'alerta_texto' in st.session_state:
        st.download_button("Descargar alerta (TXT)", data=st.session_state.alerta_texto, file_name="alerta.txt")
    if 'ndvi_img' in st.session_state:
        st.download_button("Descargar NDVI (PNG)", data=st.session_state.ndvi_img, file_name=f"ndvi_{cultivo}.png", mime="image/png")
        st.download_button("Descargar NDRE (PNG)", data=st.session_state.ndre_img, file_name=f"ndre_{cultivo}.png", mime="image/png")

st.markdown("---")
st.caption("Plataforma desarrollada con Streamlit, Google Earth Engine y Groq. Visualización NDVI/NDRE incluida.")
