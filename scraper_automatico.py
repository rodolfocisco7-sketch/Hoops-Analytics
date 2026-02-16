# scraper_automatico.py - VERSIÓN OPTIMIZADA
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

# Importar funciones
try:
    from config_nba import JUGADORES_DB, TEAM_IDS
    from logic_nba import scrapear_jugador, obtener_jugadores_lesionados
except ImportError as e:
    logging.error(f"Error de importación: {e}")
    sys.exit(1)

class ScraperOptimizado:
    def __init__(self):
        self.dm = DataManager()
        # ⚡ CONFIGURACIÓN OPTIMIZADA
        self.config = {
            'partidos_por_jugador': 10,     
            'delay_jugadores': 0.25,         
            'delay_equipos': 0.8,            
            'timeout_minutos': 80           
        }
        self.tiempo_inicio = None
    
    def _check_timeout(self):
        """Verifica si se excedió el timeout"""
        if self.tiempo_inicio:
            minutos_transcurridos = (time.time() - self.tiempo_inicio) / 60
            if minutos_transcurridos > self.config['timeout_minutos']:
                logging.warning(f"⏱️ TIMEOUT: {minutos_transcurridos:.1f} min alcanzados")
                return True
        return False
    
    def scrapear_todo(self):
        """
        Scraping completo optimizado para GitHub Actions
        """
        self.tiempo_inicio = time.time()
        inicio = datetime.now()
        
        logging.info("="*60)
        logging.info(f"⚡ SCRAPING OPTIMIZADO INICIADO: {inicio}")
        logging.info(f"📊 Configuración:")
        logging.info(f"  - Partidos por jugador: {self.config['partidos_por_jugador']}")
        logging.info(f"  - Timeout máximo: {self.config['timeout_minutos']} min")
        logging.info("="*60)
        
        all_stats = []
        all_lesionados = []
        
        total_jugadores = 0
        jugadores_con_datos = 0
        jugadores_omitidos = 0
        errores = 0
        equipos_procesados = 0
        timeout_alcanzado = False
        
        equipos = list(JUGADORES_DB.keys())
        
        for idx_equipo, equipo in enumerate(equipos, 1):
            # ⏱️ CHECK TIMEOUT
            if self._check_timeout():
                logging.warning(f"⏱️ Deteniendo scraping por timeout en equipo {idx_equipo}/{len(equipos)}")
                timeout_alcanzado = True
                break
            
            tiempo_transcurrido = (time.time() - self.tiempo_inicio) / 60
            logging.info(f"\n[{idx_equipo}/{len(equipos)}] {equipo} | Tiempo: {tiempo_transcurrido:.1f}min")
            
            try:
                # 1. SCRAPEAR STATS
                jugadores = JUGADORES_DB[equipo]
                jugadores_equipo_exitosos = 0
                
                for nombre, info in tqdm(jugadores.items(), 
                                        desc=f"  {equipo}", 
                                        leave=False):
                    # ⏱️ CHECK TIMEOUT cada 10 jugadores
                    if total_jugadores % 10 == 0 and self._check_timeout():
                        timeout_alcanzado = True
                        break
                    
                    try:
                        player_id = info.get('id')
                        if not player_id:
                            logging.warning(f"  ⚠️ {nombre} - Sin ID, omitiendo")
                            jugadores_omitidos += 1
                            continue
                        
                        # ⚡ SCRAPING OPTIMIZADO (5 partidos en vez de 10)
                        df_jug = scrapear_jugador(
                            player_id,
                            nombre,
                            equipo,
                            cantidad=self.config['partidos_por_jugador']
                        )
                        
                        if not df_jug.empty:
                            all_stats.append(df_jug)
                            jugadores_con_datos += 1
                            jugadores_equipo_exitosos += 1
                        
                        total_jugadores += 1
                        
                        # ⚡ Delay optimizado
                        time.sleep(self.config['delay_jugadores'])
                        
                    except Exception as e:
                        logging.error(f"  ✗ {nombre}: {e}")
                        errores += 1
                
                if timeout_alcanzado:
                    break
                
                logging.info(f"  ✅ {jugadores_equipo_exitosos}/{len(jugadores)} jugadores con datos")
                
                # 2. SCRAPEAR LESIONADOS
                try:
                    df_les = obtener_jugadores_lesionados(equipo)
                    
                    if not df_les.empty:
                        df_les['Equipo'] = equipo
                        all_lesionados.append(df_les)
                        logging.info(f"  🏥 {len(df_les)} lesionado(s)")
                
                except Exception as e:
                    logging.warning(f"  ⚠️ Lesionados: {e}")
                
                equipos_procesados += 1
                
                # ⚡ Delay optimizado entre equipos
                if idx_equipo < len(equipos):
                    time.sleep(self.config['delay_equipos'])
                
            except Exception as e:
                logging.error(f"❌ ERROR CRÍTICO en {equipo}: {e}")
                errores += 1
        
        # 3. CONSOLIDAR Y GUARDAR
        duracion_scraping = (time.time() - self.tiempo_inicio) / 60
        
        if all_stats:
            df_stats_final = pd.concat(all_stats, ignore_index=True)
            total_registros = self.dm.guardar_stats(df_stats_final)
            logging.info(f"✅ Stats guardadas: {total_registros} registros")
        else:
            logging.warning("⚠️ No se recolectaron estadísticas")
            df_stats_final = pd.DataFrame()
            total_registros = 0
        
        if all_lesionados:
            df_lesionados_final = pd.concat(all_lesionados, ignore_index=True)
            self.dm.guardar_lesionados(df_lesionados_final)
            logging.info(f"✅ Lesionados guardados: {len(df_lesionados_final)}")
        
        # 4. METADATA
        fin = datetime.now()
        duracion = (fin - inicio).total_seconds() / 60
        
        self.dm.actualizar_metadata({
            'total_jugadores': jugadores_con_datos,
            'total_registros': total_registros,
            'equipos': equipos_procesados,
            'errores': errores,
            'duracion_minutos': round(duracion, 2),
            'timeout': timeout_alcanzado,
            'partidos_por_jugador': self.config['partidos_por_jugador']
        })
        
        # 5. ESTADÍSTICAS
        storage_stats = self.dm.estadisticas_almacenamiento()
        
        logging.info("="*60)
        logging.info("📊 RESUMEN FINAL")
        logging.info("="*60)
        logging.info(f"✅ Equipos procesados: {equipos_procesados}/{len(equipos)}")
        logging.info(f"✅ Jugadores procesados: {total_jugadores}")
        logging.info(f"✅ Jugadores con datos: {jugadores_con_datos}")
        logging.info(f"⚠️ Jugadores omitidos: {jugadores_omitidos}")
        logging.info(f"❌ Errores: {errores}")
        logging.info(f"⏱️ Duración: {duracion:.2f} min")
        
        if timeout_alcanzado:
            logging.warning("⏱️ ADVERTENCIA: Scraping detenido por timeout")
        
        if storage_stats.get('stats_size_mb'):
            logging.info(f"💾 Tamaño archivo: {storage_stats['stats_size_mb']} MB")
        
        logging.info("="*60)
        logging.info("🎉 SCRAPING COMPLETADO" if not timeout_alcanzado else "⚠️ SCRAPING PARCIAL COMPLETADO")
        logging.info("="*60)
        
        return {
            'exito': True,
            'jugadores': jugadores_con_datos,
            'registros': total_registros,
            'errores': errores,
            'duracion': duracion,
            'timeout': timeout_alcanzado,
            'equipos_procesados': equipos_procesados
        }

if __name__ == "__main__":
    try:
        scraper = ScraperOptimizado()
        resultado = scraper.scrapear_todo()
        
        print("\n" + "="*60)
        print("📊 RESULTADO DEL SCRAPING:")
        print(f"  ✅ Equipos procesados: {resultado['equipos_procesados']}")
        print(f"  ✅ Jugadores procesados: {resultado['jugadores']}")
        print(f"  ✅ Registros guardados: {resultado['registros']}")
        print(f"  ❌ Errores: {resultado['errores']}")
        print(f"  ⏱️ Duración: {resultado['duracion']:.2f} min")
        
        if resultado['timeout']:
            print(f"  ⚠️ TIMEOUT: Scraping parcial (datos guardados)")
        
        print("="*60)
        
        # Exit code apropiado
        if resultado['jugadores'] == 0:
            sys.exit(1)  # Fallo crítico
        elif resultado['timeout']:
            sys.exit(0)  # Timeout pero datos guardados (no fallar)
        else:
            sys.exit(0)  # Éxito
            
    except Exception as e:
        logging.error(f"❌ ERROR FATAL: {e}", exc_info=True)
        sys.exit(1)