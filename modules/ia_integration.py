# modules/ia_integration.py - Versión simplicada para agricultores
# Usa Groq pero con prompts en lenguaje claro, sin tecnicismos

import os
import time
import pandas as pd
from typing import Dict, Tuple, Optional
from groq import Groq
import streamlit as st

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================== CLIENTE GROQ ==================
def _get_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)

def llamar_groq(prompt: str, system_prompt: str = None, temperature: float = 0.3, max_retries: int = 2) -> Optional[str]:
    client = _get_groq_client()
    if client is None:
        return None
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    for intento in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=temperature,
                max_tokens=1500,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception:
            time.sleep(2 ** intento)
    return None

# Para compatibilidad
llamar_deepseek = llamar_groq

# ================== PREPARACIÓN DE DATOS (igual que antes) ==================
def preparar_resumen_zonas(gdf_completo, cultivo: str, max_zonas: int = 3) -> Tuple[pd.DataFrame, Dict]:
    cols = ['id_zona', 'area_ha', 'fert_npk_actual', 'fert_ndvi', 'fert_ndre',
            'fert_materia_organica', 'fert_humedad_suelo', 'rec_N', 'rec_P', 'rec_K',
            'costo_costo_total', 'proy_rendimiento_sin_fert', 'proy_rendimiento_con_fert',
            'proy_incremento_esperado', 'textura_suelo', 'arena', 'limo', 'arcilla']
    for col in cols:
        if col not in gdf_completo.columns:
            gdf_completo[col] = 0.0
    df = gdf_completo[cols].copy()
    df.columns = ['Zona', 'Area_ha', 'NPK', 'NDVI', 'NDRE', 'MO_%', 'Humedad',
                  'N_rec', 'P_rec', 'K_rec', 'Costo_total', 'Rend_sin_fert',
                  'Rend_con_fert', 'Inc_%', 'Textura', 'Arena_%', 'Limo_%', 'Arcilla_%']
    stats = {
        'total_area': df['Area_ha'].sum(),
        'num_zonas': len(df),
        'npk_prom': df['NPK'].mean(),
        'npk_min': df['NPK'].min(),
        'npk_max': df['NPK'].max(),
        'mo_prom': df['MO_%'].mean(),
        'mo_min': df['MO_%'].min(),
        'mo_max': df['MO_%'].max(),
        'humedad_prom': df['Humedad'].mean(),
        'humedad_min': df['Humedad'].min(),
        'humedad_max': df['Humedad'].max(),
        'ndvi_prom': df['NDVI'].mean(),
        'ndvi_min': df['NDVI'].min(),
        'ndvi_max': df['NDVI'].max(),
        'ndre_prom': df['NDRE'].mean(),
        'ndre_min': df['NDRE'].min(),
        'ndre_max': df['NDRE'].max(),
        'rend_sin_prom': df['Rend_sin_fert'].mean(),
        'rend_con_prom': df['Rend_con_fert'].mean(),
        'inc_prom': df['Inc_%'].mean(),
        'costo_total': df['Costo_total'].sum(),
        'textura_dominante': df['Textura'].mode()[0] if not df['Textura'].empty else 'No determinada'
    }
    df_sorted = df.sort_values('NPK')
    n = max_zonas
    indices = [0, len(df)//2, -1] if len(df) >= 3 else list(range(len(df)))
    df_muestra = df_sorted.iloc[indices].head(n)
    return df_muestra, stats

# ================== ANÁLISIS EN LENGUAJE CAMPESINO ==================

def generar_analisis_fertilidad(df_resumen: pd.DataFrame, stats: Dict, cultivo: str) -> str:
    system = """Eres un agricultor experto que habla con palabras sencillas.
    Explica la fertilidad del suelo usando comparaciones como "bueno", "regular", "malo".
    Usa ejemplos de la vida diaria: "como si la tierra estuviera cansada", "como cuando una persona no come bien".
    Da recomendaciones prácticas que se puedan hacer con materiales locales (compost, estiércol, ceniza, etc.)."""
    
    # Convertir números a calificaciones cualitativas
    npk = stats['npk_prom']
    if npk >= 0.7:
        cal_npk = "MUY BUENA 🌟"
        color = "verde"
    elif npk >= 0.4:
        cal_npk = "REGULAR 🟡"
        color = "amarillo"
    else:
        cal_npk = "MALA 🔴"
        color = "rojo"
    
    mo = stats['mo_prom']
    if mo >= 4:
        cal_mo = "BUENA (tierra viva)"
    elif mo >= 2.5:
        cal_mo = "REGULAR (necesita más abono)"
    else:
        cal_mo = "BAJA (tierra cansada)"
    
    prompt = f"""
Nuestro cultivo es {cultivo}.

**Estado del suelo:**
- Fertilidad general (NPK): {cal_npk} (valor {stats['npk_prom']:.2f} de 1)
- Materia orgánica: {cal_mo} ({stats['mo_prom']:.1f}%)
- Tipo de tierra: {stats['textura_dominante']}

**¿Qué significa esto?**
- Si la fertilidad es MALA: la tierra no tiene suficiente comida para las plantas.
- Si es REGULAR: hay algo de comida pero falta.
- Si es BUENA: la tierra está sana.

**Recomendaciones fáciles de hacer:**
- Para mejorar la materia orgánica: haga montones de hojas secas, restos de cosecha y estiércol. Déjelos descomponer 3 meses y luego esparza.
- Si la tierra es arenosa: agregue arcilla o estiércol bien descompuesto para que retenga agua.
- Si es arcillosa: agregue arena o ceniza para que no se apelmace.
- Use abonos verdes: siembre frijol o haba entre surcos y entiérrelos antes de que den semilla.

**No gaste dinero en químicos caros.** Con compost y rotación de cultivos la tierra se recupera sola.
"""
    resultado = llamar_groq(prompt, system_prompt=system, temperature=0.5)
    if resultado is None:
        return "⚠️ No se pudo conectar con la IA. Revise su conexión a internet."
    return resultado

def generar_analisis_ndvi_ndre(df_resumen: pd.DataFrame, stats: Dict, cultivo: str) -> str:
    system = "Eres un agricultor sabio que explica usando analogías simples. Habla de 'plantas verdes y felices' o 'plantas amarillas y tristes'."
    ndvi = stats['ndvi_prom']
    if ndvi >= 0.7:
        estado = "MUY VERDE Y SANO 🌿"
    elif ndvi >= 0.4:
        estado = "NORMAL, PERO PUEDE MEJORAR 🌱"
    else:
        estado = "AMARILLO, ESTRESADO 🍂"
    
    ndre = stats['ndre_prom']
    if ndre >= 0.4:
        nitro = "BUEN NIVEL DE NITRÓGENO (como si tuviera buen abono)"
    elif ndre >= 0.25:
        nitro = "FALTA UN POCO DE NITRÓGENO"
    else:
        nitro = "MUY POCO NITRÓGENO (las plantas están flacas)"
    
    prompt = f"""
El satélite nos muestra cómo está el cultivo de {cultivo}:

- **Vigor general (NDVI):** {estado} (valor {stats['ndvi_prom']:.2f})
- **Nitrógeno (NDRE):** {nitro} (valor {stats['ndre_prom']:.2f})

**Explicación para el campesino:**
- Si el mapa sale verde oscuro: las plantas están felices, bien alimentadas.
- Si sale amarillo o rojo: hay problemas (falta agua, nutrientes o plaga).

**¿Qué hacer según el mapa?**
- En las zonas VERDES: siga haciendo lo mismo.
- En las zonas AMARILLAS: riegue un poco más o aplique abono líquido (té de compost).
- En las zonas ROJAS: revise si hay plagas, si el suelo está muy duro o si falta abono.

**Consejo práctico:** Camine por las zonas más feas y cave un hueco. Si la tierra está seca o muy dura, necesita materia orgánica. Si hay bichos, use cal o ceniza.
"""
    resultado = llamar_groq(prompt, system_prompt=system, temperature=0.5)
    if resultado is None:
        return "⚠️ No se pudo analizar el vigor del cultivo por error de red."
    return resultado

def generar_analisis_riesgo_hidrico(df_resumen: pd.DataFrame, stats: Dict, cultivo: str) -> str:
    system = "Eres un campesino que conoce el agua. Usa frases como 'la tierra bebe agua como esponja' o 'se seca rápido como ladrillo'."
    hum = stats['humedad_prom']
    if hum >= 0.4:
        agua = "SUELO HÚMEDO (bueno para el cultivo)"
    elif hum >= 0.25:
        agua = "HUMEDAD MEDIA (peligra si no llueve pronto)"
    else:
        agua = "SUELO SECO (urge regar o capturar agua)"
    
    text = stats['textura_dominante']
    if "arenoso" in text.lower():
        consejo = "La tierra arenosa es como un colador: el agua se va rápido. Ponga mucha materia orgánica (estiércol, paja) para que retenga humedad."
    elif "arcilloso" in text.lower():
        consejo = "La tierra arcillosa se encharca. Haga surcos en curva para que el agua no se estanque y pudra las raíces."
    else:
        consejo = "Su tierra es buena para retener agua, pero vigile que no se seque en verano."
    
    prompt = f"""
**Estado del agua en el suelo:**
- {agua} (índice {stats['humedad_prom']:.2f})
- Tipo de tierra: {text}

**Explicación sencilla:**
- Si el suelo está seco: las plantas sufren como nosotros sin agua.
- Si está muy mojado: las raíces se ahogan.

**Qué hacer según su tierra:**
{consejo}

**Recomendaciones para ahorrar agua:**
- Ponga hojas secas o paja alrededor de las plantas (acolchado). Así el sol no seca la tierra.
- Haga hoyos o terrazas para que el agua de lluvia no se escurra.
- Riegue temprano en la mañana o al atardecer para que no se evapore.

**Si tiene poco dinero:** Use botellas de plástico con agujeritos enterradas cerca de la raíz (riego por goteo casero).
"""
    resultado = llamar_groq(prompt, system_prompt=system, temperature=0.5)
    if resultado is None:
        return "⚠️ No se pudo analizar el riesgo de agua."
    return resultado

def generar_analisis_costos(df_resumen: pd.DataFrame, stats: Dict, cultivo: str) -> str:
    system = "Eres un contador campesino. Habla de ahorros y gastos con números redondos y consejos para no gastar mucho."
    costo = stats['costo_total']
    inc = stats['inc_prom']
    if costo < 500:
        ahorro = "gasta poco dinero en fertilizantes"
    elif costo < 1500:
        ahorro = "gasta un dinero moderado"
    else:
        ahorro = "gasta mucho dinero en productos químicos"
    
    prompt = f"""
**Su gasto actual en fertilizantes y aplicaciones:** ${costo:,.0f} en total.
**El fertilizante le aumenta el rendimiento en un {inc:.1f}% aproximadamente.**

**Explicación para no desperdiciar plata:**
- {ahorro}.
- Si deja de comprar fertilizantes caros y usa compost, puede ahorrar hasta un 60% de ese dinero.

**Opciones más baratas que funcionan:**
1. **Compost casero:** con cáscaras, estiércol de gallina/cabra y restos de cocina.
2. **Bocashi:** mezcle tierra, ceniza, afrecho, melaza (si consigue) y déjelo fermentar 15 días.
3. **Purín de ortiga o de cola de caballo:** remoje maleza en agua 10 días y luego diluya (1 litro por 10 de agua) y riegue.

**Invierta su dinero en:** semillas criollas, herramientas manuales y cercas vivas. Eso dura años.
"""
    resultado = llamar_groq(prompt, system_prompt=system, temperature=0.5)
    if resultado is None:
        return "⚠️ No se pudo analizar los costos."
    return resultado

def generar_recomendaciones_integradas(df_resumen: pd.DataFrame, stats: Dict, cultivo: str) -> str:
    system = "Eres un tío sabio que da consejos prácticos para la finca. Habla como si estuvieras conversando en la cocina."
    prompt = f"""
Después de revisar la tierra y las plantas de {cultivo}, estos son mis **5 consejos fáciles y baratos** para que mejore la cosecha:

1. **Abone con compost** ─ Haga un montón de hojas secas, estiércol y restos de cocina. En 2 meses ya puede usarlo.
2. **Siembre diferentes cultivos** ─ No ponga solo {cultivo}. Alterne con frijol, haba o maíz para que la tierra no se canse.
3. **Acolchado (paja o plástico viejo)** ─ Cubra el suelo para que no pierda agua y no crezca maleza.
4. **Hoyos profundos** ─ Al sembrar, haga hoyos de 30 cm y eche abajo un puñado de compost. Las raíces crecerán más fuertes.
5. **Observe las hormigas y las hojas** ─ Si ve hormigas coloradas, hay pulgones. Si las hojas tienen manchas blancas, es hongo. A tiempo se controla con cal o ceniza.

**Recuerde:** La tierra es como una olla: si la alimenta bien, ella le da buena comida. No necesita químicos caros.
"""
    resultado = llamar_groq(prompt, system_prompt=None, temperature=0.7)
    if resultado is None:
        return "⚠️ No se pudieron generar recomendaciones. Intente más tarde."
    return resultado
