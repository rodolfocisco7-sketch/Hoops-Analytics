# logic_nba.py
import pandas as pd
import time
import random
import numpy as np
from curl_cffi import requests
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scoreboardv2
import pytz
from datetime import datetime, timedelta

# Configuración básica para el scraper
CONFIG = {
    'TIMEOUT_SCRAPING': 20,
    'HORAS_OFFSET_PANAMA': 5,
    'DELAY_MIN': 0.5,
    'DELAY_MAX': 1.5
}

def obtener_fecha_panama():
    return datetime.now(pytz.timezone('America/Panama'))

def scrapear_jugador(player_id, nombre_jugador, equipo_sel, cantidad=7):
    """
    Extrae estadísticas del jugador (Versión para Scraper Automático)
    
    Args:
        player_id: ID del jugador en SofaScore
        nombre_jugador: Nombre del jugador
        equipo_sel: Nombre del equipo
        cantidad: Número de partidos a extraer (default: 7)
    
    Returns:
        DataFrame con estadísticas del jugador
    """
    # Importar JUGADORES_DB aquí para evitar imports circulares
    from config_nba import JUGADORES_DB
    
    lista_stats = []
    scraper = requests.Session() 
    
    info_jugador = JUGADORES_DB.get(equipo_sel, {}).get(nombre_jugador, {})
    altura = info_jugador.get("alt", 0)
    posicion = info_jugador.get("pos", "N/A")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Referer': 'https://www.sofascore.com/',
    }
    
    try:
        url_lista = f"https://api.sofascore.com/api/v1/player/{player_id}/events/last/0"
        response = scraper.get(url_lista, headers=headers, impersonate="chrome120", timeout=CONFIG['TIMEOUT_SCRAPING'])
        data = response.json()
        
        eventos_todos = data.get('events', [])[::-1]
        eventos = [ev for ev in eventos_todos if ev.get('tournament', {}).get('name') == 'NBA'][:cantidad]
        
        for ev in eventos:
            ev_id = ev.get('id')
            fecha_utc = pd.to_datetime(ev.get('startTimestamp'), unit='s')
            fecha_local = fecha_utc - pd.Timedelta(hours=CONFIG['HORAS_OFFSET_PANAMA'])
            
            es_local = equipo_sel in ev.get('homeTeam', {}).get('name', '')
            localia = "Local" if es_local else "Visitante"
            
            url_stats = f"https://api.sofascore.com/api/v1/event/{ev_id}/player/{player_id}/statistics"
            time.sleep(random.uniform(CONFIG['DELAY_MIN'], CONFIG['DELAY_MAX']))
            
            response_stats = scraper.get(url_stats, headers=headers, impersonate="chrome120", timeout=CONFIG['TIMEOUT_SCRAPING'])
            stat_data = response_stats.json()
            
            s = stat_data.get('statistics', {})
            if s and s.get('secondsPlayed', 0) > 0:
                puntos = s.get('points', 0)
                tiros = s.get('fieldGoalsAttempted', 0)
                eficiencia = puntos / tiros if tiros > 0 else 0
                
                lista_stats.append({
                    "Jugador": nombre_jugador,
                    "Equipo": equipo_sel,
                    "Posicion": posicion,
                    "Altura": altura,
                    "Fecha": fecha_local,
                    "Puntos": puntos,
                    "Rebotes": s.get('rebounds', 0),
                    "Asistencias": s.get('assists', 0),
                    "Minutos": round(s.get('secondsPlayed', 0) / 60, 1),
                    "Tiros": tiros,
                    "Eficiencia": round(eficiencia, 2),
                    "Localia": localia,
                    "Timestamp": ev.get('startTimestamp')
                })
    except Exception as e:
        print(f"❌ Error en {nombre_jugador}: {str(e)}")
        
    return pd.DataFrame(lista_stats)

def obtener_jugadores_lesionados(team_name):
    """
    Obtiene jugadores lesionados desde SofaScore
    
    Args:
        team_name: Nombre del equipo
    
    Returns:
        DataFrame con jugadores lesionados
    """
    from config_nba import TEAM_IDS
    
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    team_id = TEAM_IDS.get(team_name)
    if not team_id: 
        return pd.DataFrame()
    
    lesionados = []
    try:
        url_next = f"https://api.sofascore.com/api/v1/team/{team_id}/events/next/0"
        data = requests.get(url_next, headers=headers).json()
        if data.get('events'):
            event_id = data['events'][0]['id']
            lineup_data = requests.get(f"https://api.sofascore.com/api/v1/event/{event_id}/lineups", headers=headers).json()
            is_home = data['events'][0]['homeTeam']['id'] == team_id
            team_key = 'home' if is_home else 'away'
            missing = lineup_data.get(team_key, {}).get('missingPlayers', [])
            for player in missing:
                p_info = player.get('player', {})
                lesionados.append({
                    'Jugador': p_info.get('name', 'N/A'),
                    'Razon': player.get('reason', 'Unknown'),
                    'Equipo': team_name
                })
    except: 
        pass
    return pd.DataFrame(lesionados)