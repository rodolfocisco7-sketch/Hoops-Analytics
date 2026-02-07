# data_manager.py
import pandas as pd
import json
from datetime import datetime, timedelta
import os

class DataManager:
    """Gestiona datos con estrategia de reemplazo total"""
    
    DATA_DIR = 'data'
    STATS_FILE = f'{DATA_DIR}/stats_latest.parquet'
    LESIONADOS_FILE = f'{DATA_DIR}/lesionados_latest.parquet'
    METADATA_FILE = f'{DATA_DIR}/metadata.json'
    
    # CONFIGURACIÓN: Solo guardar últimos N días
    DIAS_RETENER = 15  # 15 días de histórico
    PARTIDOS_POR_JUGADOR = 10  # Máximo por jugador
    
    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
    
    def guardar_stats(self, df_nuevo):
        """
        Reemplaza archivo completo - solo últimos 15 días
        """
        # Filtrar solo datos recientes
        fecha_limite = datetime.now() - timedelta(days=self.DIAS_RETENER)
        df_filtrado = df_nuevo[df_nuevo['Fecha'] >= fecha_limite].copy()
        
        # Limitar partidos por jugador
        df_final = (
            df_filtrado
            .sort_values('Fecha', ascending=False)
            .groupby('Jugador')
            .head(self.PARTIDOS_POR_JUGADOR)
            .reset_index(drop=True)
        )
        
        # Guardar en formato Parquet (10x más comprimido que CSV)
        df_final.to_parquet(
            self.STATS_FILE,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        return len(df_final)
    
    def cargar_stats(self):
        """Lee archivo Parquet - súper rápido"""
        if not os.path.exists(self.STATS_FILE):
            return pd.DataFrame()
        
        return pd.read_parquet(self.STATS_FILE)
    
    def guardar_lesionados(self, df_lesionados):
        """Reemplaza lesionados completo"""
        if df_lesionados.empty:
            return
            
        df_lesionados.to_parquet(
            self.LESIONADOS_FILE,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
    
    def cargar_lesionados(self):
        """Lee lesionados"""
        if not os.path.exists(self.LESIONADOS_FILE):
            return pd.DataFrame()
        
        return pd.read_parquet(self.LESIONADOS_FILE)
    
    def actualizar_metadata(self, stats):
        """Guarda info de última actualización"""
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
    
    def cargar_metadata(self):
        """Lee metadata"""
        if not os.path.exists(self.METADATA_FILE):
            return {}
        
        with open(self.METADATA_FILE, 'r') as f:
            return json.load(f)
    
    def obtener_stats_jugador(self, jugador, equipo=None):
        """Filtra stats de un jugador"""
        df = self.cargar_stats()
        
        if df.empty:
            return pd.DataFrame()
        
        mask = df['Jugador'] == jugador
        if equipo:
            mask &= df['Equipo'] == equipo
        
        return df[mask].sort_values('Fecha', ascending=False)
    
    def obtener_stats_equipo(self, equipo):
        """Filtra stats de un equipo"""
        df = self.cargar_stats()
        
        if df.empty:
            return pd.DataFrame()
        
        return df[df['Equipo'] == equipo].sort_values('Fecha', ascending=False)
    
    def obtener_lesionados_equipo(self, equipo):
        """Filtra lesionados de un equipo"""
        df = self.cargar_lesionados()
        
        if df.empty:
            return pd.DataFrame()
        
        return df[df['Equipo'] == equipo]
    
    def estadisticas_almacenamiento(self):
        """Info de uso de espacio"""
        stats = {}
        
        if os.path.exists(self.STATS_FILE):
            size_mb = os.path.getsize(self.STATS_FILE) / (1024 * 1024)
            stats['stats_size_mb'] = round(size_mb, 2)
        
        if os.path.exists(self.LESIONADOS_FILE):
            size_kb = os.path.getsize(self.LESIONADOS_FILE) / 1024
            stats['lesionados_size_kb'] = round(size_kb, 2)
        
        df = self.cargar_stats()
        if not df.empty:
            stats['total_registros'] = len(df)
            stats['jugadores_unicos'] = df['Jugador'].nunique()
            stats['equipos_unicos'] = df['Equipo'].nunique()
            stats['fecha_mas_antigua'] = df['Fecha'].min().isoformat()
            stats['fecha_mas_reciente'] = df['Fecha'].max().isoformat()
        
        return stats