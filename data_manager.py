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

    # ============================================================================
    # MÉTODOS DE CARGA (Usados por la APP)
    # ============================================================================

    def cargar_stats(self):
        """Carga datos. Usa cache solo si está en Streamlit"""
        if self.is_streamlit:
            import streamlit as st
            @st.cache_data(ttl=300)
            def _internal_load():
                return self._leer_datos('stats_latest.parquet', self.STATS_FILE)
            return _internal_load()
        return self._leer_datos('stats_latest.parquet', self.STATS_FILE)

    def cargar_lesionados(self):
        """Carga datos de lesionados"""
        if self.is_streamlit:
            import streamlit as st
            @st.cache_data(ttl=300)
            def _internal_load():
                return self._leer_datos('lesionados_latest.parquet', self.LESIONADOS_FILE)
            return _internal_load()
        return self._leer_datos('lesionados_latest.parquet', self.LESIONADOS_FILE)

    def cargar_metadata(self):
        """Carga metadata del último scraping"""
        try:
            if self.is_cloud:
                # Leer desde GitHub
                url = self._get_url('metadata.json')
                response = requests.get(url, headers={'Cache-Control': 'no-cache'}, timeout=10)
                response.raise_for_status()
                return response.json()
            
            # Leer localmente
            if os.path.exists(self.METADATA_FILE):
                with open(self.METADATA_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Error cargando metadata: {e}")
        
        # Retornar metadata vacía si falla
        return {
            'ultima_actualizacion': 'Sin datos',
            'total_jugadores': 0,
            'total_registros': 0,
            'equipos_procesados': 0
        }

    def obtener_stats_equipo(self, nombre_equipo):
        """Obtiene estadísticas de un equipo específico"""
        df_all = self.cargar_stats()
        if df_all.empty:
            return pd.DataFrame()
        
        df_equipo = df_all[df_all['Equipo'] == nombre_equipo].copy()
        
        # Asegurar que Fecha sea datetime
        if not df_equipo.empty and 'Fecha' in df_equipo.columns:
            df_equipo['Fecha'] = pd.to_datetime(df_equipo['Fecha'])
        
        return df_equipo

    def obtener_lesionados_equipo(self, nombre_equipo):
        """Obtiene jugadores lesionados de un equipo específico"""
        df_lesionados = self.cargar_lesionados()
        if df_lesionados.empty:
            return pd.DataFrame()
        
        return df_lesionados[df_lesionados['Equipo'] == nombre_equipo].copy()

    def estadisticas_almacenamiento(self):
        """Retorna estadísticas del almacenamiento actual"""
        stats = {
            'stats_exists': os.path.exists(self.STATS_FILE),
            'lesionados_exists': os.path.exists(self.LESIONADOS_FILE),
            'metadata_exists': os.path.exists(self.METADATA_FILE),
        }
        
        if stats['stats_exists']:
            stats['stats_size_mb'] = round(os.path.getsize(self.STATS_FILE) / 1024 / 1024, 2)
        
        if stats['lesionados_exists']:
            stats['lesionados_size_mb'] = round(os.path.getsize(self.LESIONADOS_FILE) / 1024 / 1024, 2)
        
        return stats

    def _leer_datos(self, filename, local_path):
        """Método interno para leer datos desde GitHub o local"""
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

    # ============================================================================
    # MÉTODOS DE GUARDADO (Usados por el SCRAPER)
    # ============================================================================

    def guardar_stats(self, df_nuevo):
        """Guarda estadísticas aplicando filtros de retención"""
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
        print(f"✅ Stats guardadas en {self.STATS_FILE} - {len(df_final)} registros")
        return len(df_final)

    def guardar_stats_jugador(self, equipo, jugador, df_stats):
        """
        Guarda estadísticas de un jugador individual (usado por scraper_automatico.py)
        Acumula en el archivo principal
        """
        if df_stats.empty:
            return
        
        # Cargar datos existentes
        df_existente = pd.DataFrame()
        if os.path.exists(self.STATS_FILE):
            df_existente = pd.read_parquet(self.STATS_FILE)
        
        # Agregar nuevos datos
        df_combinado = pd.concat([df_existente, df_stats], ignore_index=True)
        
        # Eliminar duplicados basados en Jugador + Timestamp
        df_combinado = df_combinado.drop_duplicates(subset=['Jugador', 'Timestamp'], keep='last')
        
        # Guardar
        df_combinado.to_parquet(self.STATS_FILE, index=False)

    def guardar_lesionados(self, df_lesionados):
        """Guarda información de jugadores lesionados"""
        if df_lesionados.empty:
            print("⚠️ No hay datos de lesionados para guardar.")
            return
        
        df_lesionados.to_parquet(self.LESIONADOS_FILE, index=False)
        print(f"✅ Lesionados guardados en {self.LESIONADOS_FILE} - {len(df_lesionados)} registros")

    def guardar_lesionados_equipo(self, equipo, df_lesionados):
        """
        Guarda lesionados de un equipo específico (usado por scraper_automatico.py)
        Acumula en el archivo principal
        """
        if df_lesionados.empty:
            return
        
        # Cargar datos existentes
        df_existente = pd.DataFrame()
        if os.path.exists(self.LESIONADOS_FILE):
            df_existente = pd.read_parquet(self.LESIONADOS_FILE)
        
        # Eliminar lesionados antiguos de este equipo
        if not df_existente.empty:
            df_existente = df_existente[df_existente['Equipo'] != equipo]
        
        # Agregar nuevos lesionados
        df_combinado = pd.concat([df_existente, df_lesionados], ignore_index=True)
        
        # Guardar
        df_combinado.to_parquet(self.LESIONADOS_FILE, index=False)

    def actualizar_metadata(self, stats):
        """Actualiza el archivo de metadata con información del scraping"""
        metadata = {
            'ultima_actualizacion': datetime.now().isoformat(),
            'total_jugadores': stats.get('total_jugadores', 0),
            'total_registros': stats.get('total_registros', 0),
            'equipos_procesados': stats.get('equipos', 0),
            'duracion_minutos': stats.get('duracion_minutos', 0),
            'errores': stats.get('errores', 0)
        }
        
        with open(self.METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Metadata actualizada: {metadata}")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================
if __name__ == "__main__":
    dm = DataManager()
    
    print("\n📊 ESTADÍSTICAS DE ALMACENAMIENTO:")
    print(json.dumps(dm.estadisticas_almacenamiento(), indent=2))
    
    print("\n📋 METADATA:")
    print(json.dumps(dm.cargar_metadata(), indent=2))
    
    print("\n🏀 CARGANDO STATS:")
    df = dm.cargar_stats()
    if not df.empty:
        print(f"  ✅ {len(df)} registros cargados")
        print(f"  ✅ {df['Jugador'].nunique()} jugadores únicos")
        print(f"  ✅ {df['Equipo'].nunique()} equipos")
    else:
        print("  ⚠️ No hay datos disponibles")