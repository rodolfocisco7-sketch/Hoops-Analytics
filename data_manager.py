import pandas as pd
import json
from datetime import datetime, timedelta
import os
import requests 
import time

class DataManager:
    USER = "rodolfocisco7-sketch"
    REPO = "Hoops-Analytics" 
    BASE_RAW_URL = f"https://raw.githubusercontent.com/{USER}/{REPO}/main/data"
    
    DATA_DIR = 'data'
    STATS_FILE = os.path.join(DATA_DIR, 'stats_latest.parquet')
    LESIONADOS_FILE = os.path.join(DATA_DIR, 'lesionados_latest.parquet')
    METADATA_FILE = os.path.join(DATA_DIR, 'metadata.json')
    
    DIAS_RETENER = 15
    PARTIDOS_POR_JUGADOR = 10 

    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        # Detectar si estamos en Streamlit o en un script normal
        self.is_streamlit = self._check_streamlit()
        # En la nube siempre usamos GitHub para LEER
        self.is_cloud = self.is_streamlit 

    def _check_streamlit(self):
        """Detecta si Streamlit está corriendo"""
        try:
            import streamlit as st
            return True
        except ImportError:
            return False

    def _get_url(self, filename):
        timestamp = int(time.time() * 1000)
        return f"{self.BASE_RAW_URL}/{filename}?v={timestamp}"

    # --- MÉTODOS DE CARGA (Usados por la APP) ---

    def cargar_stats(self):
        """Carga datos. Usa cache solo si está en Streamlit"""
        if self.is_streamlit:
            import streamlit as st
            @st.cache_data(ttl=300)
            def _internal_load():
                return self._leer_datos('stats_latest.parquet', self.STATS_FILE)
            return _internal_load()
        return self._leer_datos('stats_latest.parquet', self.STATS_FILE)

    def _leer_datos(self, filename, local_path):
        try:
            if self.is_cloud:
                url = self._get_url(filename)
                headers = {'Cache-Control': 'no-cache'}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                from io import BytesIO
                return pd.read_parquet(BytesIO(response.content))
            
            if os.path.exists(local_path):
                return pd.read_parquet(local_path)
        except Exception as e:
            print(f"❌ Error leyendo {filename}: {e}")
        return pd.DataFrame()

    # --- MÉTODOS DE GUARDADO (Usados por el SCRAPER) ---

    def guardar_stats(self, df_nuevo):
        if df_nuevo.empty:
            print("⚠️ Intento de guardar DataFrame vacío.")
            return 0
            
        # Asegurar que 'Fecha' sea datetime
        df_nuevo['Fecha'] = pd.to_datetime(df_nuevo['Fecha'])
        
        fecha_limite = datetime.now() - timedelta(days=self.DIAS_RETENER)
        df_filtrado = df_nuevo[df_nuevo['Fecha'] >= fecha_limite].copy()
        
        if df_filtrado.empty:
            print(f"⚠️ Todos los datos son anteriores a {self.DIAS_RETENER} días. Guardando datos crudos.")
            df_final = df_nuevo
        else:
            df_final = (
                df_filtrado
                .sort_values('Fecha', ascending=False)
                .groupby('Jugador')
                .head(self.PARTIDOS_POR_JUGADOR)
                .reset_index(drop=True)
            )
            
        df_final.to_parquet(self.STATS_FILE, index=False)
        print(f"✅ Archivo guardado en {self.STATS_FILE} - {len(df_final)} registros")
        return len(df_final)

    def actualizar_metadata(self, stats):
        metadata = {
            'ultima_actualizacion': datetime.now().strftime("%d/%m %H:%M"),
            'total_jugadores': stats.get('total_jugadores', 0),
            'total_registros': stats.get('total_registros', 0),
            'equipos_procesados': stats.get('equipos', 0)
        }
        with open(self.METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✅ Metadata actualizada localmente.")