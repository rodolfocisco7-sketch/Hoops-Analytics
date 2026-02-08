import pandas as pd
import json
from datetime import datetime, timedelta
import os
import requests # Necesario para leer la metadata desde la URL
import time

class DataManager:
    """Gestiona datos con lectura remota desde GitHub para Streamlit Cloud"""
    
    # CONFIGURACIÓN: Ajusta estas URLs con tu usuario de GitHub
    USER = "rodolfocisco7-sketch"
    REPO = "hoops-analytics" # Confirma si este es el nombre de tu repo
    BASE_RAW_URL = f"https://raw.githubusercontent.com/{USER}/{REPO}/main/data"
    
    DATA_DIR = 'data'
    STATS_FILE = f'{DATA_DIR}/stats_latest.parquet'
    LESIONADOS_FILE = f'{DATA_DIR}/lesionados_latest.parquet'
    METADATA_FILE = f'{DATA_DIR}/metadata.json'
    
    DIAS_RETENER = 15
    PARTIDOS_POR_JUGADOR = 10 
    
    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        # Detectamos si estamos en Streamlit Cloud o local
        self.is_cloud = os.getenv('STREAMLIT_SERVER_PORT') is not None

    def _get_url(self, filename):
        """Genera URL Raw con bypass de caché"""
        return f"{self.BASE_RAW_URL}/{filename}?v={int(time.time())}"

    def cargar_stats(self):
        """Lee de GitHub en la nube o local en PC"""
        try:
            if self.is_cloud:
                return pd.read_parquet(self._get_url('stats_latest.parquet'))
            
            if not os.path.exists(self.STATS_FILE):
                return pd.DataFrame()
            return pd.read_parquet(self.STATS_FILE)
        except Exception:
            return pd.DataFrame()

    def cargar_lesionados(self):
        """Lee de GitHub en la nube o local en PC"""
        try:
            if self.is_cloud:
                return pd.read_parquet(self._get_url('lesionados_latest.parquet'))
            
            if not os.path.exists(self.LESIONADOS_FILE):
                return pd.DataFrame()
            return pd.read_parquet(self.LESIONADOS_FILE)
        except Exception:
            return pd.DataFrame()

    def cargar_metadata(self):
        """Lee metadata de GitHub o local"""
        try:
            if self.is_cloud:
                response = requests.get(self._get_url('metadata.json'))
                return response.json()
            
            if not os.path.exists(self.METADATA_FILE):
                return {}
            with open(self.METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    # --- Los métodos de GUARDAR se quedan igual (solo los usa el scraper) ---
    def guardar_stats(self, df_nuevo):
        fecha_limite = datetime.now() - timedelta(days=self.DIAS_RETENER)
        df_filtrado = df_nuevo[df_nuevo['Fecha'] >= fecha_limite].copy()
        df_final = df_filtrado.sort_values('Fecha', ascending=False).groupby('Jugador').head(self.PARTIDOS_POR_JUGADOR).reset_index(drop=True)
        df_final.to_parquet(self.STATS_FILE, engine='pyarrow', compression='snappy', index=False)
        return len(df_final)

    def guardar_lesionados(self, df_lesionados):
        if not df_lesionados.empty:
            df_lesionados.to_parquet(self.LESIONADOS_FILE, engine='pyarrow', compression='snappy', index=False)

    def actualizar_metadata(self, stats):
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

    # --- Filtros (no cambian porque llaman a cargar_stats) ---
    def obtener_stats_equipo(self, equipo):
        df = self.cargar_stats()
        if df.empty: return pd.DataFrame()
        return df[df['Equipo'] == equipo].sort_values('Fecha', ascending=False)

    def obtener_lesionados_equipo(self, equipo):
        df = self.cargar_lesionados()
        if df.empty: return pd.DataFrame()
        return df[df['Equipo'] == equipo]
    
    def estadisticas_almacenamiento(self):
        # Simplificado para evitar errores en nube
        return {"modo": "Nube (GitHub Raw)" if self.is_cloud else "Local"}