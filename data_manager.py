import pandas as pd
import json
from datetime import datetime, timedelta
import os
import time

# ============================================================================
# CACHE A NIVEL DE MÓDULO — Esto SÍ funciona en Streamlit Cloud
# (el patrón @st.cache_data dentro de métodos de instancia NO cachea)
# ============================================================================
try:
    import streamlit as st

    @st.cache_data(ttl=1800, show_spinner=False)
    def _cargar_parquet_github(url: str) -> pd.DataFrame:
        """Cache real: se ejecuta 1 sola vez por sesión (30 min TTL)"""
        import requests
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            from io import BytesIO
            return pd.read_parquet(BytesIO(response.content))
        except Exception as e:
            print(f"❌ Error cargando {url}: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=1800, show_spinner=False)
    def _cargar_metadata_github(url: str) -> dict:
        """Cache para metadata JSON"""
        import requests
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error cargando metadata: {e}")
            return {}

    _STREAMLIT_DISPONIBLE = True

except ImportError:
    _STREAMLIT_DISPONIBLE = False

    def _cargar_parquet_github(url: str) -> pd.DataFrame:
        import requests
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            from io import BytesIO
            return pd.read_parquet(BytesIO(response.content))
        except Exception as e:
            print(f"❌ Error cargando {url}: {e}")
            return pd.DataFrame()

    def _cargar_metadata_github(url: str) -> dict:
        import requests
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {}


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
        self.is_streamlit = _STREAMLIT_DISPONIBLE
        # En Streamlit Cloud no tenemos escritura local confiable → siempre GitHub
        self.is_cloud = self.is_streamlit

    # ============================================================================
    # MÉTODOS DE CARGA (usados por la app)
    # ============================================================================

    def cargar_stats(self) -> pd.DataFrame:
        """
        Carga stats desde GitHub con cache de 30 min (Streamlit Cloud)
        o desde archivo local (scraper).
        Primera carga: ~1-2s. Siguientes: instantáneo.
        """
        if self.is_cloud:
            # Sin timestamp para que el cache funcione correctamente
            url = f"{self.BASE_RAW_URL}/stats_latest.parquet"
            return _cargar_parquet_github(url)

        if os.path.exists(self.STATS_FILE):
            return pd.read_parquet(self.STATS_FILE)
        return pd.DataFrame()

    def cargar_lesionados(self) -> pd.DataFrame:
        """Carga lesionados con cache de 30 min"""
        if self.is_cloud:
            url = f"{self.BASE_RAW_URL}/lesionados_latest.parquet"
            return _cargar_parquet_github(url)

        if os.path.exists(self.LESIONADOS_FILE):
            return pd.read_parquet(self.LESIONADOS_FILE)
        return pd.DataFrame()

    def cargar_metadata(self) -> dict:
        """Carga metadata con cache"""
        if self.is_cloud:
            url = f"{self.BASE_RAW_URL}/metadata.json"
            return _cargar_metadata_github(url)

        if os.path.exists(self.METADATA_FILE):
            with open(self.METADATA_FILE, 'r') as f:
                return json.load(f)

        return {
            'ultima_actualizacion': 'Sin datos',
            'total_jugadores': 0,
            'total_registros': 0,
            'equipos_procesados': 0
        }

    def obtener_stats_equipo(self, nombre_equipo: str) -> pd.DataFrame:
        """
        Filtra el DataFrame global por equipo.
        Rápido porque cargar_stats() usa cache — no re-descarga.
        """
        df_all = self.cargar_stats()
        if df_all.empty:
            return pd.DataFrame()

        df_equipo = df_all[df_all['Equipo'] == nombre_equipo].copy()

        if not df_equipo.empty and 'Fecha' in df_equipo.columns:
            df_equipo['Fecha'] = pd.to_datetime(df_equipo['Fecha'])

        return df_equipo

    def obtener_lesionados_equipo(self, nombre_equipo: str) -> pd.DataFrame:
        """Filtra lesionados por equipo"""
        df_lesionados = self.cargar_lesionados()
        if df_lesionados.empty:
            return pd.DataFrame()
        return df_lesionados[df_lesionados['Equipo'] == nombre_equipo].copy()

    def estadisticas_almacenamiento(self) -> dict:
        stats = {
            'stats_exists': os.path.exists(self.STATS_FILE),
            'lesionados_exists': os.path.exists(self.LESIONADOS_FILE),
            'metadata_exists': os.path.exists(self.METADATA_FILE),
            'modo': 'Streamlit Cloud (GitHub)' if self.is_cloud else 'Local'
        }
        if stats['stats_exists']:
            stats['stats_size_mb'] = round(os.path.getsize(self.STATS_FILE) / 1024 / 1024, 2)
        if stats['lesionados_exists']:
            stats['lesionados_size_mb'] = round(os.path.getsize(self.LESIONADOS_FILE) / 1024 / 1024, 2)
        return stats

    # ============================================================================
    # MÉTODOS DE GUARDADO (usados por el scraper — corre LOCAL, no en Cloud)
    # ============================================================================

    def guardar_stats(self, df_nuevo: pd.DataFrame) -> int:
        if df_nuevo.empty:
            print("⚠️ Intento de guardar DataFrame vacío.")
            return 0

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
        print(f"✅ Stats guardadas: {len(df_final)} registros")
        return len(df_final)

    def guardar_stats_jugador(self, equipo: str, jugador: str, df_stats: pd.DataFrame):
        if df_stats.empty:
            return

        df_existente = pd.DataFrame()
        if os.path.exists(self.STATS_FILE):
            df_existente = pd.read_parquet(self.STATS_FILE)

        df_combinado = pd.concat([df_existente, df_stats], ignore_index=True)
        df_combinado = df_combinado.drop_duplicates(subset=['Jugador', 'Timestamp'], keep='last')
        df_combinado.to_parquet(self.STATS_FILE, index=False)

    def guardar_lesionados(self, df_lesionados: pd.DataFrame):
        if df_lesionados.empty:
            print("⚠️ No hay datos de lesionados para guardar.")
            return
        df_lesionados.to_parquet(self.LESIONADOS_FILE, index=False)
        print(f"✅ Lesionados guardados: {len(df_lesionados)} registros")

    def guardar_lesionados_equipo(self, equipo: str, df_lesionados: pd.DataFrame):
        if df_lesionados.empty:
            return

        df_existente = pd.DataFrame()
        if os.path.exists(self.LESIONADOS_FILE):
            df_existente = pd.read_parquet(self.LESIONADOS_FILE)

        if not df_existente.empty:
            df_existente = df_existente[df_existente['Equipo'] != equipo]

        df_combinado = pd.concat([df_existente, df_lesionados], ignore_index=True)
        df_combinado.to_parquet(self.LESIONADOS_FILE, index=False)

    def actualizar_metadata(self, stats: dict):
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
        print(f"✅ Metadata actualizada")


# ============================================================================
# TEST RÁPIDO
# ============================================================================
if __name__ == "__main__":
    dm = DataManager()

    print("\n📊 ALMACENAMIENTO:")
    print(json.dumps(dm.estadisticas_almacenamiento(), indent=2))

    print("\n📋 METADATA:")
    print(json.dumps(dm.cargar_metadata(), indent=2))

    print("\n🏀 STATS:")
    import time
    t = time.time()
    df = dm.cargar_stats()
    elapsed = time.time() - t
    if not df.empty:
        print(f"  ✅ {len(df)} registros | {df['Jugador'].nunique()} jugadores | {elapsed:.2f}s")
    else:
        print("  ⚠️ Sin datos")