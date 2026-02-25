# logic_nba.py
import pandas as pd
import time
import random
import numpy as np
from curl_cffi import requests
from datetime import datetime, timedelta
import pytz

CONFIG = {
    'TIMEOUT_SCRAPING': 20,
    'HORAS_OFFSET_PANAMA': 5,
    'DELAY_MIN': 0.5,
    'DELAY_MAX': 1.5
}

def obtener_fecha_panama():
    return datetime.now(pytz.timezone('America/Panama'))


# ============================================================================
# PRÓXIMO PARTIDO 
# ============================================================================

def obtener_proximo_partido(team_name: str) -> dict:
    """
    Obtiene el próximo partido del equipo desde SofaScore.
    Más confiable que nba_api en Streamlit Cloud.
    
    Returns:
        dict con: hay_juego, rival, rival_sofascore_id, localia, fecha, event_id
    """
    from config_nba import TEAM_IDS

    resultado_vacio = {
        "hay_juego": False,
        "rival": None,
        "rival_sofascore_id": None,
        "localia": None,
        "fecha": None,
        "event_id": None
    }

    team_id = TEAM_IDS.get(team_name)
    if not team_id:
        return resultado_vacio

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.sofascore.com/'
    }

    try:
        session = requests.Session()
        url = f"https://api.sofascore.com/api/v1/team/{team_id}/events/next/0"
        response = session.get(url, headers=headers, impersonate="chrome120", timeout=10)
        data = response.json()

        eventos = data.get('events', [])

        # Filtrar solo partidos NBA
        nba_eventos = [
            ev for ev in eventos
            if ev.get('tournament', {}).get('name') == 'NBA'
        ]

        if not nba_eventos:
            return resultado_vacio

        ev = nba_eventos[0]

        home_id = ev.get('homeTeam', {}).get('id')
        es_local = (home_id == team_id)

        rival_key = 'awayTeam' if es_local else 'homeTeam'
        rival_info = ev.get(rival_key, {})
        rival_nombre = rival_info.get('name', 'Desconocido')
        rival_sofascore_id = rival_info.get('id')

        # Mapear nombre SofaScore → nombre en JUGADORES_DB
        rival_nombre_db = _mapear_nombre_equipo(rival_nombre)

        # Fecha en hora
        ts = ev.get('startTimestamp', 0)
        fecha_utc = datetime.utcfromtimestamp(ts)
        fecha_panama = fecha_utc - timedelta(hours=CONFIG['HORAS_OFFSET_PANAMA'])
        ahora_panama = datetime.utcnow() - timedelta(hours=CONFIG['HORAS_OFFSET_PANAMA'])

        dias_diff = (fecha_panama.date() - ahora_panama.date()).days
        if dias_diff == 0:
            fecha_texto = f"Hoy {fecha_panama.strftime('%H:%M')}"
        elif dias_diff == 1:
            fecha_texto = f"Mañana {fecha_panama.strftime('%H:%M')}"
        elif dias_diff < 0:
            fecha_texto = f"En curso"
        else:
            fecha_texto = fecha_panama.strftime('%d/%m %H:%M')

        return {
            "hay_juego": True,
            "rival": rival_nombre_db,           
            "rival_display": rival_nombre,       
            "rival_sofascore_id": rival_sofascore_id,
            "localia": "Local" if es_local else "Visitante",
            "fecha": fecha_texto,
            "event_id": ev.get('id')
        }

    except Exception as e:
        print(f"❌ Error obteniendo próximo partido para {team_name}: {e}")
        return resultado_vacio


def _mapear_nombre_equipo(nombre_sofascore: str) -> str:
    """
    Mapea nombres de equipos de SofaScore al formato de JUGADORES_DB.
    SofaScore a veces usa nombres cortos o distintos.
    """
    MAPA = {
        "LA Clippers": "Los Angeles Clippers",
        "LA Lakers": "Los Angeles Lakers",
        "Golden State": "Golden State Warriors",
        "Oklahoma City": "Oklahoma City Thunder",
        "New Orleans": "New Orleans Pelicans",
        "San Antonio": "San Antonio Spurs",
        "Portland": "Portland Trail Blazers",
        "Utah": "Utah Jazz",
        "Memphis": "Memphis Grizzlies",
        "Minnesota": "Minnesota Timberwolves",
        "New York": "New York Knicks",
        "Philadelphia": "Philadelphia 76ers",
        "Washington": "Washington Wizards",
        "Charlotte": "Charlotte Hornets",
        "Orlando": "Orlando Magic",
        "Cleveland": "Cleveland Cavaliers",
        "Detroit": "Detroit Pistons",
        "Indiana": "Indiana Pacers",
        "Chicago": "Chicago Bulls",
        "Atlanta": "Atlanta Hawks",
        "Brooklyn": "Brooklyn Nets",
        "Toronto": "Toronto Raptors",
        "Milwaukee": "Milwaukee Bucks",
        "Miami": "Miami Heat",
        "Boston": "Boston Celtics",
        "Denver": "Denver Nuggets",
        "Phoenix": "Phoenix Suns",
        "Sacramento": "Sacramento Kings",
        "Dallas": "Dallas Mavericks",
        "Houston": "Houston Rockets",
    }
    # Primero intenta match exacto
    for clave, valor in MAPA.items():
        if clave.lower() in nombre_sofascore.lower() or nombre_sofascore.lower() in clave.lower():
            return valor
    return nombre_sofascore


# ============================================================================
# ANÁLISIS DEFENSIVO DEL RIVAL
# ============================================================================

def calcular_stats_defensivas_rival(df_all: pd.DataFrame, equipo_rival: str, metrica: str = 'Puntos') -> dict:
    """
    Calcula cuánto permite el rival en promedio en la métrica dada.
    Basado en los datos scrapeados de los jugadores del rival.
    
    Nota: Como los datos son de jugadores propios, usamos el promedio
    del rival como referencia de su nivel ofensivo para estimar su defensa.
    
    Returns:
        dict con estadísticas defensivas del rival
    """
    df_rival = df_all[df_all['Equipo'] == equipo_rival].copy()

    if df_rival.empty:
        return {
            'disponible': False,
            'mensaje': f'Sin datos del rival ({equipo_rival})'
        }

    # Stats del rival (nivel ofensivo — referencia de qué tan bueno es el equipo)
    promedio_rival = df_rival.groupby('Jugador')[metrica].mean()
    total_equipo = promedio_rival.sum()
    top_jugadores = promedio_rival.nlargest(3)

    # Pace estimado: partidos jugados / días desde primer partido
    fechas = pd.to_datetime(df_rival['Fecha'])
    dias_rango = (fechas.max() - fechas.min()).days + 1
    partidos_totales = df_rival['Timestamp'].nunique()
    pace_aprox = round(partidos_totales / max(dias_rango, 1) * 7, 1)  # partidos por semana

    return {
        'disponible': True,
        'equipo': equipo_rival,
        'metrica': metrica,
        'produccion_total_rival': round(total_equipo, 1),
        'promedio_por_jugador': round(promedio_rival.mean(), 1),
        'top_jugadores': top_jugadores.round(1).to_dict(),
        'pace_semanal': pace_aprox,
        'total_jugadores_data': len(promedio_rival),
        'nivel_rival': _clasificar_nivel(total_equipo, metrica)
    }


def _clasificar_nivel(total: float, metrica: str) -> str:
    """Clasifica el nivel del rival según su producción total"""
    umbrales = {
        'Puntos': [(120, '🔴 Ofensa élite'), (110, '🟡 Ofensa sólida'), (0, '🟢 Ofensa débil')],
        'Rebotes': [(45, '🔴 Gran rebote'), (40, '🟡 Buen rebote'), (0, '🟢 Débil en rebote')],
        'Asistencias': [(28, '🔴 Gran asistencia'), (24, '🟡 Buen paseo'), (0, '🟢 Débil pasando')]
    }
    for umbral, label in umbrales.get(metrica, []):
        if total >= umbral:
            return label
    return '📊 Sin clasificar'


# ============================================================================
# SCRAPER DE JUGADORES
# ============================================================================

def scrapear_jugador(player_id, nombre_jugador, equipo_sel, cantidad=7):
    """
    Extrae estadísticas del jugador desde SofaScore.
    """
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


# ============================================================================
# LESIONADOS
# ============================================================================

def obtener_jugadores_lesionados(team_name):
    """
    Obtiene jugadores lesionados desde SofaScore.
    """
    from config_nba import TEAM_IDS

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/json',
        'Referer': 'https://www.sofascore.com/'
    }
    team_id = TEAM_IDS.get(team_name)
    if not team_id:
        return pd.DataFrame()

    lesionados = []
    try:
        session = requests.Session()
        url_next = f"https://api.sofascore.com/api/v1/team/{team_id}/events/next/0"
        data = session.get(url_next, headers=headers, impersonate="chrome120", timeout=10).json()

        if data.get('events'):
            event_id = data['events'][0]['id']
            home_id = data['events'][0]['homeTeam']['id']
            is_home = home_id == team_id
            team_key = 'home' if is_home else 'away'

            lineup_url = f"https://api.sofascore.com/api/v1/event/{event_id}/lineups"
            lineup_data = session.get(lineup_url, headers=headers, impersonate="chrome120", timeout=10).json()

            missing = lineup_data.get(team_key, {}).get('missingPlayers', [])
            for player in missing:
                p_info = player.get('player', {})
                lesionados.append({
                    'Jugador': p_info.get('name', 'N/A'),
                    'Razon': player.get('reason', 'Unknown'),
                    'Equipo': team_name
                })
    except Exception as e:
        print(f"⚠️ Error lesionados {team_name}: {e}")

    return pd.DataFrame(lesionados)