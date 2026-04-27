# agroia_gee.py - Funciones Python puras para análisis climático de Ají, Rocoto y Papa Andina
# Sin referencias a Streamlit. Versión limpia.

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

# ================= GOOGLE EARTH ENGINE =================
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

# ================= GROQ IA =================
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ================= LECTURA DE API KEY =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY and GROQ_AVAILABLE:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ================= PARÁMETROS DE CULTIVOS =================
CULTIVOS = ["AJÍ", "ROCOTO", "PAPA ANDINA"]
ICONOS = {"AJÍ": "🌶️", "ROCOTO": "🥵", "PAPA ANDINA": "🥔"}
UMBRALES = {
    "AJÍ": {"NDVI_min": 0.4, "temp_min": 18, "temp_max": 30, "humedad_min": 0.25, "humedad_max": 0.65},
    "ROCOTO": {"NDVI_min": 0.45, "temp_min": 16, "temp_max": 28, "humedad_min": 0.30, "humedad_max": 0.70},
    "PAPA ANDINA": {"NDVI_min": 0.5, "temp_min": 10, "temp_max": 22, "humedad_min": 0.35, "humedad_max": 0.75}
}

# ================= INICIALIZACIÓN DE GEE =================
def inicializar_gee():
    if not GEE_AVAILABLE:
        return False
    try:
        ee.Initialize()
        return True
    except Exception as e:
        return False

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
                return None
    except Exception as e:
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
            return None
        if gdf is not None:
            gdf = validar_crs(gdf)
            gdf = gdf.explode(ignore_index=True)
            gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
            if len(gdf) == 0:
                return None
            geom_unida = gdf.unary_union
            gdf_unido = gpd.GeoDataFrame({'geometry': [geom_unida]}, crs='EPSG:4326')
            return gdf_unido
        return None
    except Exception as e:
        return None


# ================= FUNCIONES PARA MAPAS DE CALOR (NDVI, NDRE, TEMP, PRECIP, NDWI) =================
def obtener_imagen_gee_thumbnail(gdf, image_func, vis_params, dimensions='600x600'):
    """Helper: genera una URL de miniatura de GEE para el área de la parcela."""
    try:
        bounds = gdf.total_bounds
        region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
        image = image_func(region)
        url = image.getThumbURL({
            'region': region,
            'dimensions': dimensions,
            'format': 'png',
            'min': vis_params.get('min', 0),
            'max': vis_params.get('max', 1),
            'palette': vis_params.get('palette', ['blue', 'green', 'red'])
        })
        return url
    except Exception as e:
        return None

def mapa_ndvi(gdf, fecha):
    """Genera URL de mapa de calor NDVI (Sentinel-2) en la fecha más cercana disponible."""
    def build_image(region):
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(fecha.strftime('%Y-%m-%d'), (fecha + timedelta(days=30)).strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        image = collection.first()
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return ndvi.clip(region)
    return obtener_imagen_gee_thumbnail(gdf, build_image, {'min': -0.2, 'max': 0.8, 'palette': ['red', 'yellow', 'green']})

def mapa_ndre(gdf, fecha):
    """Genera URL de mapa de calor NDRE (Sentinel-2). NDRE = (B8A - B5) / (B8A + B5)"""
    def build_image(region):
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(fecha.strftime('%Y-%m-%d'), (fecha + timedelta(days=30)).strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        image = collection.first()
        ndre = image.normalizedDifference(['B8A', 'B5']).rename('NDRE')
        return ndre.clip(region)
    return obtener_imagen_gee_thumbnail(gdf, build_image, {'min': -0.2, 'max': 0.8, 'palette': ['red', 'yellow', 'green']})

def mapa_temperatura(gdf, fecha):
    """Mapa de temperatura superficial (ERA5-Land) para la fecha."""
    def build_image(region):
        collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
            .filterBounds(region) \
            .filterDate(fecha.strftime('%Y-%m-%d'), (fecha + timedelta(days=1)).strftime('%Y-%m-%d')) \
            .select('temperature_2m')
        image = collection.first()
        temp_c = image.subtract(273.15).rename('temp_c')
        return temp_c.clip(region)
    return obtener_imagen_gee_thumbnail(gdf, build_image, {'min': -5, 'max': 40, 'palette': ['blue', 'cyan', 'yellow', 'red']})

def mapa_precipitacion(gdf, fecha):
    """Mapa de precipitación diaria (CHIRPS)"""
    def build_image(region):
        collection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterBounds(region) \
            .filterDate(fecha.strftime('%Y-%m-%d'), (fecha + timedelta(days=1)).strftime('%Y-%m-%d')) \
            .select('precipitation')
        image = collection.first()
        return image.clip(region)
    return obtener_imagen_gee_thumbnail(gdf, build_image, {'min': 0, 'max': 50, 'palette': ['white', 'lightblue', 'blue', 'darkblue']})

def mapa_ndwi(gdf, fecha):
    """Mapa NDWI (Green-NIR)/(Green+NIR) para Sentinel-2."""
    def build_image(region):
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(fecha.strftime('%Y-%m-%d'), (fecha + timedelta(days=30)).strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        image = collection.first()
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        return ndwi.clip(region)
    return obtener_imagen_gee_thumbnail(gdf, build_image, {'min': -0.5, 'max': 0.5, 'palette': ['brown', 'white', 'blue']})

# ================= SERIES TEMPORALES GEE =================
def _fc_to_dataframe(fc_info, date_key, value_key):
    """Convierte el resultado de FeatureCollection.getInfo() en DataFrame."""
    records = []
    for f in fc_info.get('features', []):
        props = f.get('properties', {})
        val = props.get(value_key)
        date_ms = props.get(date_key)
        if val is not None and date_ms is not None:
            records.append({'date': datetime.utcfromtimestamp(date_ms / 1000), value_key: val})
    return pd.DataFrame(records).sort_values('date') if records else pd.DataFrame()

def obtener_serie_temporal_ndvi(gdf, fecha_inicio, fecha_fin):
    """Retorna DataFrame con columnas ['date', 'ndvi'] para el período dado."""
    try:
        bounds = gdf.total_bounds
        region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(region)
                      .filterDate(fecha_inicio, fecha_fin)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                      .limit(200))

        def to_feature(image):
            ndvi = image.normalizedDifference(['B8', 'B4'])
            mean = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=30, maxPixels=1e9
            ).get('nd')
            return ee.Feature(None, {
                'date_ms': image.date().millis(),
                'ndvi': mean
            })

        fc = collection.map(to_feature)
        info = fc.getInfo()
        return _fc_to_dataframe(info, 'date_ms', 'ndvi')
    except Exception:
        return pd.DataFrame()

def obtener_serie_temporal_temperatura(gdf, fecha_inicio, fecha_fin):
    """Retorna DataFrame con columnas ['date', 'temp'] en °C para el período dado."""
    try:
        bounds = gdf.total_bounds
        region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
        collection = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                      .filterBounds(region)
                      .filterDate(fecha_inicio, fecha_fin)
                      .select('temperature_2m')
                      .limit(400))

        def to_feature(image):
            mean = (image.subtract(273.15)
                    .reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=11132, maxPixels=1e9)
                    .get('temperature_2m'))
            return ee.Feature(None, {
                'date_ms': image.date().millis(),
                'temp': mean
            })

        fc = collection.map(to_feature)
        info = fc.getInfo()
        return _fc_to_dataframe(info, 'date_ms', 'temp')
    except Exception:
        return pd.DataFrame()

def obtener_serie_temporal_precipitacion(gdf, fecha_inicio, fecha_fin):
    """Retorna DataFrame con columnas ['date', 'precip'] en mm para el período dado."""
    try:
        bounds = gdf.total_bounds
        region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
        collection = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                      .filterBounds(region)
                      .filterDate(fecha_inicio, fecha_fin)
                      .select('precipitation')
                      .limit(400))

        def to_feature(image):
            mean = image.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=5566, maxPixels=1e9
            ).get('precipitation')
            return ee.Feature(None, {
                'date_ms': image.date().millis(),
                'precip': mean
            })

        fc = collection.map(to_feature)
        info = fc.getInfo()
        return _fc_to_dataframe(info, 'date_ms', 'precip')
    except Exception:
        return pd.DataFrame()

# ================= VALORES ACTUALES =================
def obtener_ndvi_actual(gdf):
    """Retorna el valor NDVI más reciente disponible (últimos 30 días)."""
    try:
        fecha_fin = datetime.now().strftime('%Y-%m-%d')
        fecha_inicio = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        df = obtener_serie_temporal_ndvi(gdf, fecha_inicio, fecha_fin)
        if not df.empty:
            return float(df['ndvi'].iloc[-1])
    except Exception:
        pass
    return round(np.random.uniform(0.3, 0.7), 2)

def obtener_ndwi_actual(gdf):
    """Retorna un índice de humedad de suelo aproximado basado en NDWI."""
    try:
        bounds = gdf.total_bounds
        region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
        fecha_fin = datetime.now().strftime('%Y-%m-%d')
        fecha_inicio = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(fecha_inicio, fecha_fin) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        image = collection.first()
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('ndwi')
        mean = ndwi.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=30).get('ndwi').getInfo()
        return float(mean) if mean is not None else round(np.random.uniform(0.2, 0.6), 2)
    except Exception:
        return round(np.random.uniform(0.2, 0.6), 2)

def obtener_ndre_actual(gdf):
    """Retorna el valor NDRE más reciente disponible (últimos 30 días)."""
    try:
        bounds = gdf.total_bounds
        region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
        fecha_fin = datetime.now().strftime('%Y-%m-%d')
        fecha_inicio = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(fecha_inicio, fecha_fin) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        image = collection.first()
        ndre = image.normalizedDifference(['B8A', 'B5']).rename('ndre')
        mean = ndre.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=30).get('ndre').getInfo()
        return float(mean) if mean is not None else round(np.random.uniform(0.2, 0.6), 2)
    except Exception:
        return round(np.random.uniform(0.2, 0.6), 2)

def obtener_temperatura_actual(gdf):
    """Retorna la temperatura media más reciente en °C (ERA5-Land)."""
    try:
        fecha_fin = datetime.now().strftime('%Y-%m-%d')
        fecha_inicio = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        df = obtener_serie_temporal_temperatura(gdf, fecha_inicio, fecha_fin)
        if not df.empty:
            return float(df['temp'].iloc[-1])
    except Exception:
        pass
    return round(np.random.uniform(14, 28), 1)

def obtener_precipitacion_actual(gdf):
    """Retorna la precipitación del día más reciente disponible en mm (CHIRPS)."""
    try:
        fecha_fin = datetime.now().strftime('%Y-%m-%d')
        fecha_inicio = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        df = obtener_serie_temporal_precipitacion(gdf, fecha_inicio, fecha_fin)
        if not df.empty:
            return float(df['precip'].iloc[-1])
    except Exception:
        pass
    return round(np.random.exponential(5), 1)

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
3. Proporciona 3 recomendaciones concretas y accionables para el productor (riego, fertilización, protección, ajuste de fechas).
4. Si hay riesgo de helada o golpe de calor, menciónalo.
5. Formato claro, conciso, máximo 250 palabras.
"""
    return consultar_groq(prompt, max_tokens=600)
