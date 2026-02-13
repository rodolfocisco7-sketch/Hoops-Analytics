import pandas as pd
import json
from datetime import datetime, timedelta
import os
import requests 
import time
import streamlit as st

class DataManager:
    """Gestiona datos con lectura remota desde GitHub para Streamlit Cloud"""
    
    # CONFIGURACIÓN
    USER = "rodolfocisco7-sketch"
    REPO = "Hoops-Analytics" 
    BASE_RAW_URL = f"https://raw.githubusercontent.com/{USER}/{REPO}/main/data"
    
    DATA_DIR = 'data'
    STATS_FILE = f'{DATA_DIR}/stats_latest.parquet'
    LESIONADOS_FILE = f'{DATA_DIR}/lesionados_latest.parquet'
    METADATA_FILE = f'{DATA_DIR}/metadata.json'
    
    DIAS_RETENER = 15
    PARTIDOS_POR_JUGADOR = 10 
    
    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        # ✅ FORZAR MODO CLOUD (siempre lee de GitHub)
        self.is_cloud = True
        
    def _get_url(self, filename):
        """Genera URL Raw con bypass AGRESIVO de caché"""
        # Timestamp en milisegundos + hash aleatorio
        timestamp = int(time.time() * 1000)
        cache_buster = hash(str(timestamp)) % 10000
        return f"{self.BASE_RAW_URL}/{filename}?nocache={timestamp}&v={cache_buster}"

    def _fetch_with_retry(self, url, max_retries=3):
        """Intenta fetch con reintentos y headers anti-caché"""
        headers = {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)
        
        return None

    @st.cache_data(ttl=300)  # Cache por 5 minutos
    def cargar_stats(_self):
        """Lee de GitHub con cache busting agresivo"""
        try:
            if _self.is_cloud:
                url = _self._get_url('stats_latest.parquet')
                response = _self._fetch_with_retry(url)
                return pd.read_parquet(response.content)
            
            if not os.path.exists(_self.STATS_FILE):
                return pd.DataFrame()
            return pd.read_parquet(_self.STATS_FILE)
        except Exception as e:
            print(f"❌ Error cargando stats: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=300)
    def cargar_lesionados(_self):
        """Lee de GitHub con cache busting agresivo"""
        try:
            if _self.is_cloud:
                url = _self._get_url('lesionados_latest.parquet')
                response = _self._fetch_with_retry(url)
                return pd.read_parquet(response.content)
            
            if not os.path.exists(_self.LESIONADOS_FILE):
                return pd.DataFrame()
            return pd.read_parquet(_self.LESIONADOS_FILE)
        except Exception as e:
            print(f"❌ Error cargando lesionados: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=60)  # Metadata: cache 1 minuto
    def cargar_metadata(_self):
        """Lee metadata de GitHub con cache busting"""
        try:
            if _self.is_cloud:
                url = _self._get_url('metadata.json')
                response = _self._fetch_with_retry(url)
                return response.json()
            
            if not os.path.exists(_self.METADATA_FILE):
                return {}
            with open(_self.METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error cargando metadata: {e}")
            return {}

    # ========== MÉTODOS DEL SCRAPER (sin cambios) ==========
    
    def guardar_stats(self, df_nuevo):
        """Usado por scraper_automatico.py"""
        fecha_limite = datetime.now() - timedelta(days=self.DIAS_RETENER)
        df_filtrado = df_nuevo[df_nuevo['Fecha'] >= fecha_limite].copy()
        df_final = (
            df_filtrado
            .sort_values('Fecha', ascending=False)
            .groupby('Jugador')
            .head(self.PARTIDOS_POR_JUGADOR)
            .reset_index(drop=True)
        )
        df_final.to_parquet(
            self.STATS_FILE, 
            engine='pyarrow', 
            compression='snappy', 
            index=False
        )
        return len(df_final)

    def guardar_lesionados(self, df_lesionados):
        """Usado por scraper_automatico.py"""
        if not df_lesionados.empty:
            df_lesionados.to_parquet(
                self.LESIONADOS_FILE,
                engine='pyarrow',
                compression='snappy',
                index=False
            )

    def actualizar_metadata(self, stats):
        """Usado por scraper_automatico.py"""
        metadata = {
            'ultima_actualizacion': datetime.now().isoformat(),
            'total_jugadores': stats.get('total_jugadores', 0),
            'total_registros': stats.get('total_registros', 0),
            'equipos_procesados': stats.get('equipos', 0),
            'errores': stats.get('errores', 0),
            'duracion_minutos': stats.get('duracion_minutos', 0)
        }
        with open(self.METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)

    def obtener_stats_equipo(self, equipo):
        """Filtrar stats por equipo"""
        df = self.cargar_stats()
        if df.empty:
            return pd.DataFrame()
        return df[df['Equipo'] == equipo].sort_values('Fecha', ascending=False)

    def obtener_lesionados_equipo(self, equipo):
        """Filtrar lesionados por equipo"""
        df = self.cargar_lesionados()
        if df.empty:
            return pd.DataFrame()
        return df[df['Equipo'] == equipo]
    
    def estadisticas_almacenamiento(self):
        """Info de dónde se leen los datos"""
        stats = {
            "modo": "🌐 GitHub Raw" if self.is_cloud else "💻 Local"
        }
        
        # Si hay datos, agregar info adicional
        df = self.cargar_stats()
        if not df.empty:
            stats.update({
                'total_registros': len(df),
                'jugadores_unicos': df['Jugador'].nunique(),
                'equipos_unicos': df['Equipo'].nunique(),
                'fecha_mas_antigua': df['Fecha'].min().isoformat() if 'Fecha' in df.columns else 'N/A',
                'fecha_mas_reciente': df['Fecha'].max().isoformat() if 'Fecha' in df.columns else 'N/A'
            })
        
        return stats