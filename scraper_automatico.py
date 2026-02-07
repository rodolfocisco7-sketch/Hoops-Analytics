# scraper_automatico.py
import pandas as pd
import time
import logging
from datetime import datetime
from data_manager import DataManager
from tqdm import tqdm
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Importar funciones del app.py
try:
    from app import (
        scrapear_jugador, 
        obtener_jugadores_lesionados, 
        JUGADORES_DB,
        TEAM_IDS
    )
except ImportError:
    logging.error("No se pudo importar desde app.py")
    sys.exit(1)

class ScraperOptimizado:
    def __init__(self):
        self.dm = DataManager()
    
    def scrapear_todo(self):
        """
        Scraping completo - reemplaza todo
        """
        inicio = datetime.now()
        logging.info("="*60)
        logging.info(f"SCRAPING INICIADO: {inicio}")
        
        all_stats = []
        all_lesionados = []
        
        total_jugadores = 0
        errores = 0
        
        equipos = list(JUGADORES_DB.keys())
        
        for equipo in tqdm(equipos, desc="Equipos"):
            logging.info(f"Procesando: {equipo}")
            
            try:
                # 1. SCRAPEAR STATS
                jugadores = JUGADORES_DB[equipo]
                
                for nombre, info in tqdm(jugadores.items(), 
                                        desc=f"  {equipo}", 
                                        leave=False):
                    try:
                        df_jug = scrapear_jugador(
                            info['id'],
                            nombre,
                            equipo,
                            cantidad=10  # Últimos 10 partidos
                        )
                        
                        if not df_jug.empty:
                            all_stats.append(df_jug)
                            total_jugadores += 1
                        
                        time.sleep(0.3)  # Rate limiting
                        
                    except Exception as e:
                        logging.error(f"  ✗ {nombre}: {e}")
                        errores += 1
                
                # 2. SCRAPEAR LESIONADOS
                try:
                    df_les = obtener_jugadores_lesionados(equipo)
                    
                    if not df_les.empty:
                        df_les['Equipo'] = equipo
                        all_lesionados.append(df_les)
                        logging.info(f"  → {len(df_les)} lesionados")
                
                except Exception as e:
                    logging.warning(f"  ⚠ Lesionados no disponibles: {e}")
                
                time.sleep(1)  # Pausa entre equipos
                
            except Exception as e:
                logging.error(f"ERROR CRÍTICO en {equipo}: {e}")
                errores += 1
        
        # 3. CONSOLIDAR Y GUARDAR
        if all_stats:
            df_stats_final = pd.concat(all_stats, ignore_index=True)
            total_registros = self.dm.guardar_stats(df_stats_final)
            logging.info(f"✓ Stats guardadas: {total_registros} registros")
        else:
            df_stats_final = pd.DataFrame()
            total_registros = 0
        
        if all_lesionados:
            df_lesionados_final = pd.concat(all_lesionados, ignore_index=True)
            self.dm.guardar_lesionados(df_lesionados_final)
            logging.info(f"✓ Lesionados guardados: {len(df_lesionados_final)}")
        
        # 4. METADATA
        fin = datetime.now()
        duracion = (fin - inicio).total_seconds() / 60
        
        self.dm.actualizar_metadata({
            'total_jugadores': total_jugadores,
            'total_registros': total_registros,
            'equipos': len(equipos),
            'errores': errores,
            'duracion_minutos': round(duracion, 2)
        })
        
        # 5. ESTADÍSTICAS
        storage_stats = self.dm.estadisticas_almacenamiento()
        
        logging.info("="*60)
        logging.info(f"SCRAPING COMPLETADO en {duracion:.2f} min")
        logging.info(f"Jugadores: {total_jugadores} | Errores: {errores}")
        logging.info(f"Tamaño archivo: {storage_stats.get('stats_size_mb', 0)} MB")
        logging.info("="*60)
        
        return {
            'exito': True,
            'jugadores': total_jugadores,
            'registros': total_registros,
            'errores': errores,
            'duracion': duracion
        }

if __name__ == "__main__":
    scraper = ScraperOptimizado()
    resultado = scraper.scrapear_todo()
    
    print("\n" + "="*60)
    print("RESULTADO DEL SCRAPING:")
    print(f"  ✓ Jugadores procesados: {resultado['jugadores']}")
    print(f"  ✓ Registros guardados: {resultado['registros']}")
    print(f"  ✗ Errores: {resultado['errores']}")
    print(f"  ⏱ Duración: {resultado['duracion']:.2f} min")
    print("="*60)