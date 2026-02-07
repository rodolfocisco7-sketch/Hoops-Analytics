import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
import numpy as np 
import random
from data_manager import DataManager
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scoreboardv2

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from curl_cffi import requests
import xgboost as xgb 

from datetime import datetime, timedelta
import pytz


def obtener_fecha_panama():
    return datetime.now(pytz.timezone('America/Panama'))

st.markdown("""
<style>
    @keyframes jump {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-25px); }
    }
    .jumping-logo { font-size: 80px; animation: jump 0.8s infinite; text-align: center; }
    .brand-title { text-align: center; color: #FFD700; font-family: 'Arial Black', sans-serif; margin-bottom: 0px; text-shadow: 2px 2px #000; font-size: 42px; }
    .brand-subtitle { text-align: center; color: #00D9FF; font-size: 22px; font-weight: bold; margin-bottom: 30px; }
    .progress-stats { 
        display: flex; justify-content: space-around; 
        background: rgba(0, 217, 255, 0.1); padding: 20px; 
        border-radius: 15px; border: 1px solid #00D9FF; margin: 20px 0;
    }
    .stat-value { color: #00FFA3; font-size: 24px; font-weight: bold; text-align: center; }
    .stat-label { color: #E0F4FF; font-size: 12px; text-align: center; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

def mostrar_interfaz_carga(nombre, equipo, pct, total, actual):
    restantes = total - actual
    st.markdown('<div class="jumping-logo">🏀</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="brand-title">CANAL HOOPS ANALYTICS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">by Rodolfo Cisco</p>', unsafe_allow_html=True)
    st.markdown(f"""
        <div style="text-align:center; margin-bottom:15px;">
            <p style="color:#00D9FF; font-size:20px; margin:0;">Analizando a: <b style="color:white;">{nombre}</b></p>
            <span style="color:#6B9EB0;">{equipo}</span>
        </div>
        <div class="progress-stats">
            <div class="stat-item"><div class="stat-label">Progreso</div><div class="stat-value">{pct:.1f}%</div></div>
            <div class="stat-item"><div class="stat-label">Procesados</div><div class="stat-value">{actual}/{total}</div></div>
            <div class="stat-item"><div class="stat-label">Restantes</div><div class="stat-value">{restantes}</div></div>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================
st.set_page_config(
    page_title="🏀 Hoops Analytics Pro", 
    layout="wide",
    initial_sidebar_state="collapsed"
)
dm = DataManager()
# ============================================================================
# CONSTANTES GLOBALES
# ============================================================================
CONFIG = {
    'NUM_PARTIDOS_DEFECTO': 7,
    'TIMEOUT_SCRAPING': 15,
    'DELAY_MIN': 0.2,
    'DELAY_MAX': 0.4,
    'HORAS_OFFSET_PANAMA': 6
}

TEAM_IDS = {
    # Conferencia Oeste
    "Dallas Mavericks": 3411,
    "Denver Nuggets": 3417,
    "Golden State Warriors": 3428,
    "Houston Rockets": 3412,
    "Los Angeles Clippers": 3425,
    "Los Angeles Lakers": 3427,
    "Memphis Grizzlies": 3415,
    "Minnesota Timberwolves": 3426,
    "New Orleans Pelicans": 5539,
    "Oklahoma City Thunder": 3418,
    "Phoenix Suns": 3416,
    "Portland Trail Blazers": 3414,
    "Sacramento Kings": 3413,
    "San Antonio Spurs": 3429,
    "Utah Jazz": 3434,

    # Conferencia Este
    "Atlanta Hawks": 3423,
    "Boston Celtics": 3422,
    "Brooklyn Nets": 3436,
    "Charlotte Hornets": 3430,
    "Chicago Bulls": 3409,
    "Cleveland Cavaliers": 3432,
    "Detroit Pistons": 3424,
    "Indiana Pacers": 3419,
    "Miami Heat": 3435,
    "Milwaukee Bucks": 3410,
    "New York Knicks": 3421,
    "Orlando Magic": 3437,
    "Philadelphia 76ers": 3420,
    "Toronto Raptors": 3433,
    "Washington Wizards": 3431
}

# --- BASE DE DATOS DE JUGADORES (Completa) ---
JUGADORES_DB = {
"Atlanta Hawks": {
    "Jalen Johnson":            {"id": 1133787, "pos": "PF", "alt": 206},
    "Dyson Daniels":            {"id": 1206070, "pos": "PG", "alt": 201},
    "Onyeka Okonwu":            {"id": 1092036, "pos": "C", "alt": 203},
    "Nickeil Alexander-Walker": {"id": 988994, "pos": "SG", "alt": 196},
    "CJ McCollum":              {"id": 817253, "pos": "SG", "alt": 191},
    "Vit Krejci":               {"id": 1092950, "pos": "F", "alt": 203},
    "Corey Kispert":            {"id": 1134115, "pos": "F", "alt": 198},
    "Mouhamed Gueye":           {"id": 1181310, "pos": "F", "alt": 208},
    "Christian Koloko":         {"id": 1181314, "pos": "C", "alt": 216},
    "Zaccharie Risacher":       {"id": 1163019, "pos": "F", "alt": 204},
    "Gabe Vincent":             {"id": 958896,  "pos": "PG",   "alt": 188},
    "Buddy Hield":              {"id": 846949,  "pos": "SG",    "alt": 193},
    "Jonathan Kuminga":         {"id": 1132126, "pos": "SF",    "alt": 201},

},

"Boston Celtics": {
    "Jaylen Brown":      {"id": 846898, "pos": "SG", "alt": 198},
    "Payton Pritchard":  {"id": 1092911, "pos": "PG", "alt": 185},
    "Jayson Tatum":      {"id": 885203, "pos": "SF", "alt": 203},
    "Derrick White":     {"id": 885248, "pos": "SG", "alt": 193},
    "Nikola Vucevic":    {"id": 817423,  "pos": "C",     "alt": 208},
    "Sam Hauser":        {"id": 1133791, "pos": "SF", "alt": 201},
    "Hugo Gonzalez":     {"id": 1412639, "pos": "SG/SF", "alt": 198},
    "Jordan Walsh":      {"id": 1458572, "pos": "SG/SF", "alt": 198},
    "Max Shulga":        {"id": 1178185, "pos": "SG", "alt": 193},
    "Ron Harper Jr.":    {"id": 1179581, "pos": "SG/SF", "alt": 196},
    "Xavier Tillman":    {"id": 1092873, "pos": "PF/C", "alt": 203},
    "Luka Garza":        {"id": 1133860, "pos": "C", "alt": 208},
    "Neemias Queta":     {"id": 1132138, "pos": "C", "alt": 213},
    "Baylor Scheierman": {"id": 1181786, "pos": "SG/SF", "alt": 198},
    "John Tonje":        {"id": 1178037,   "pos": "SG", "alt": 196},

},

"Brooklyn Nets": {

    "Nicola Claxton":    {"id": 989006, "pos": "C", "alt": 211},
    "Noah Clowney":      {"id": 1457792, "pos": "PF", "alt": 208},
    "Dayron Sharpe":     {"id": 1134015, "pos": "C", "alt": 206},
    "Bojan Bogdanovic":  {"id": 817112, "pos": "SF", "alt": 201},
    "Michael Porter":    {"id": 940773, "pos": "SF", "alt": 208},
    "Egor Demin":        {"id": 1492436, "pos": "G", "alt": 206},
    "Terance Man":       {"id": 989017, "pos": "GF", "alt": 196},
    "Ziaire Williams":   {"id": 1134024, "pos": "F", "alt": 206},
    "Danny Wolf":        {"id": 1510215, "pos": "FC", "alt": 213},
    "Drake Powell":      {"id": 1947101, "pos": "GF", "alt": 198},
    "Jalen Wilson":      {"id": 1182728, "pos": "F", "alt": 198},
    "Nolan Traore":      {"id": 1436219, "pos": "F", "alt": 191},
    "Ben Saraf":         {"id": 1566961, "pos": "G", "alt": 195},
    "Josh Minott":       {"id": 1179834, "pos": "PG", "alt": 185},
},

"Charlotte Hornets": {
    
  "Miles Bridges":        {"id": 940806, "pos": "F",  "alt": 201},
  "Kon Knueppel":         {"id": 1908729, "pos": "G",  "alt": 201},
  "Brandon Miller":       {"id": 1457788, "pos": "F",  "alt": 201},
  "LaMelo Ball":          {"id": 1092868, "pos": "G",  "alt": 201},
  "Grant Williams":       {"id": 989079, "pos": "F",  "alt": 198},
  "Sion James":           {"id": 1179853, "pos": "G",  "alt": 196},
  "Tre Mann":             {"id": 1133847, "pos": "G",  "alt": 191},
  "Josh Green":           {"id": 1092867, "pos": "G",  "alt": 196},
  "Ryan Kalkbrenner":     {"id": 1178307, "pos": "C",  "alt": 216},
  "Collin Sexton":        {"id": 940769, "pos": "G",  "alt": 191},
  "Tidjane Salaun":       {"id": 1476794, "pos": "F",  "alt": 205},
  "Moussa Diabate":       {"id": 1178443, "pos": "PF/C","alt": 206},
  "Hunter Tyson":       {"id": 1178826, "pos": "SF",    "alt": 203},
  "Xavier Tillman":    {"id": 1092873, "pos": "PF/C", "alt": 203},
  "Coby White":        {"id": 989015,  "pos": "PG",    "alt": 196},


},

"Chicago Bulls": {
  
  "Nikola Vucevic":    {"id": 817423,  "pos": "C",     "alt": 208},
  "Matas Buzelis":     {"id": 1629476, "pos": "SF",    "alt": 208},
  "Josh Giddey":       {"id": 1133812, "pos": "PG",    "alt": 203},
  "Tre Jones":         {"id": 1092947, "pos": "PG",    "alt": 185},
  "Isaac Okoro":       {"id": 1092915, "pos": "SF",    "alt": 196},
  "Jalen Smith":       {"id": 1092897,    "pos": "PF/C",  "alt": 206},
  "Patrick Williams":  {"id": 1092881,    "pos": "PF",    "alt": 201},
  
  "Dalen Terry":       {"id": 1181319,    "pos": "SG",    "alt": 201},
  "Zach Collins":      {"id": 885242,    "pos": "PF/C",  "alt": 211},
  "Anfernee Simons":   {"id": 940767, "pos": "SG", "alt": 191},
  "Rob Dillingham":         {"id": 1599612, "pos": "PG", "alt": 191},
  "Leonard Miller":         {"id": 1415127, "pos": "PF", "alt": 208},
  "Collin Sexton":        {"id": 940769, "pos": "G",  "alt": 191},
  "Ousmane Dieng":           {"id": 1191132, "pos": "PF", "alt": 208},
  "Guerschon Yabusele":   {"id": 846899,  "pos": "PF", "alt": 204},
  "Jaden Ivey":      {"id": 1178370, "pos": "SG", "alt": 193},


},

"Cleveland Cavaliers": {
  "Donovan Mitchell":   {"id": 885252,    "pos": "SG",    "alt": 191},
  "Evan Mobley":        {"id": 1133830,   "pos": "PF",    "alt": 211},
  "Jarrett Allen":      {"id": 885222,    "pos": "C",     "alt": 206},
  "Jaylon Tyson":       {"id": 1381333,   "pos": "SF",    "alt": 201},
  "Ty Jerome":          {"id": 988301,    "pos": "PG",    "alt": 196},
  

  "Sam Merrill":        {"id": 1396473,   "pos": "SG",    "alt": 193},
  "Dean Wade":          {"id": 987560,    "pos": "PF/C",  "alt": 206},
  "Tyrese Proctor":     {"id": 1430914,   "pos": "PG",    "alt": 196},
  "Craig Porter Jr.":   {"id": 1178247,   "pos": "PG",    "alt": 185},
  "Larry Nance Jr.":    {"id": 817288,    "pos": "PF/C",  "alt": 203},
  "Luke Travers":       {"id": 1187303,   "pos": "SG",    "alt": 200},
  "NaeQwan Tomlin":     {"id": 1458333,   "pos": "PF",    "alt": 208},
  "Thomas Bryant":      {"id": 885216,    "pos": "C",     "alt": 208},
  "James Harden":       {"id": 817133,    "pos": "PG",    "alt": 196},
  "Dennis Schroder":      {"id": 817361,  "pos": "PG", "alt": 185},
},

"Dallas Mavericks": {
  "Kyrie Irving":          {"id": 817177,  "pos": "PG",    "alt": 188},
  "Klay Thompson":       {"id": 817404,  "pos": "SG",    "alt": 198},
  "P.J. Washington Jr.": {"id": 987613,  "pos": "PF",    "alt": 201},
  "Naji Marshall":       {"id": 1093316, "pos": "SF",    "alt": 198},
  "Max Christie":        {"id": 1178651, "pos": "SG",    "alt": 196},
  "Daniel Gafford":      {"id": 988296,  "pos": "C",     "alt": 208},
  "Dereck Lively II":    {"id": 1430931, "pos": "C",     "alt": 216},
  "Caleb Martin":        {"id": 989018,  "pos": "SF",    "alt": 196},
  "Cooper Flagg":        {"id": 1908739, "pos": "PF",    "alt": 206},
  "Brandon Williams":    {"id": 1170101, "pos": "PG",    "alt": 188},
  "Ryan Nembhard":       {"id": 1178315, "pos": "PG",    "alt": 183},
  "Dwight Powell":       {"id": 817331,  "pos": "PF/C",  "alt": 208},
  "Tyus Jones":          {"id": 817204,  "pos": "PG", "alt": 185},
  "Khris Middleton":       {"id": 817265,  "pos": "SF", "alt": 201},
  "AJ Johnson":            {"id": 1540637, "pos": "SG", "alt": 196},
  "Marvin Bagley":         {"id": 940811,  "pos": "PF", "alt": 211},
  "Malaki Branham":        {"id": 1179607, "pos": "SG", "alt": 193},
},

"Detroit Pistons": {
    "Cade Cunningham": {"id": 1133854, "pos": "PG", "alt": 201},
    "Jalen Duren":     {"id": 1179838, "pos": "C",  "alt": 208},
    "Tobias Harris":   {"id": 817141,  "pos": "PF", "alt": 203},
    "Duncan Robinson": {"id": 945092,  "pos": "SF", "alt": 201},
    "Ausar Thompson":  {"id": 1507964, "pos": "SF", "alt": 201},
    "Kevin Huerter":   {"id": 940788,  "pos": "SG", "alt": 201},
    "Isaiah Stewart":  {"id": 1092901, "pos": "PF", "alt": 203},
    "Daniss Jenkins":  {"id": 1457974, "pos": "PG", "alt": 188},
    "Caris LeVert":    {"id": 846936,  "pos": "G",  "alt": 198},
    "Javonte Green":   {"id": 954825,  "pos": "GF", "alt": 196},
    "Ron Holland":     {"id": 1629477, "pos": "SF", "alt": 201},
    "Paul Reed":       {"id": 1092906, "pos": "PF", "alt": 206},
    "Marcus Sasser":   {"id": 1179506, "pos": "PG", "alt": 185},
    "Chaz Lanier":     {"id": 1180639, "pos": "G",  "alt": 193},
    "Bobi Klintman":   {"id": 1433931, "pos": "F",  "alt": 206}

},

"Denver Nuggets": {
  "Nikola Jokić":       {"id": 817199,  "pos": "C",     "alt": 211},
  "Jamal Murray":       {"id": 846917,  "pos": "PG",    "alt": 193},
  "Aaron Gordon":       {"id": 817114,  "pos": "PF",    "alt": 203},
  "Peyton Watson":      {"id": 1178030, "pos": "SF",    "alt": 201},
  "Christian Braun":    {"id": 1178657, "pos": "SG",    "alt": 198},
  "Julian Strawther":   {"id": 1436504, "pos": "SG",    "alt": 198},
  "Jonas Valanciunas":  {"id": 817417,  "pos": "C",     "alt": 211},
  "Cameron Johnson":    {"id": 988302,  "pos": "SF",    "alt": 203},
  "Bruce Brown":        {"id": 940805,  "pos": "SG",    "alt": 193},
  "Tim Hardaway Jr.":   {"id": 817132,  "pos": "SG/SF", "alt": 196},
  "Spencer Jones":      {"id": 1177993, "pos": "SF",    "alt": 201},
  "Zeke Nnaji":         {"id": 1092914,  "pos": "PF/C", "alt": 206},
  "Jalen Pickett":      {"id": 1178425, "pos": "PG",    "alt": 193},
  "Curtis Jones":       {"id": 1179003, "pos": "PG",    "alt": 193},
},

"Golden State Warriors": {
  "Stephen Curry":           {"id": 817050,  "pos": "PG",    "alt": 188},
  "Draymond Green":          {"id": 817122,  "pos": "PF",    "alt": 198},
  "Kristaps Porzingis":       {"id": 817330, "pos": "C", "alt": 221},
  "Brandin Podziemski":      {"id": 1433919, "pos": "SG",    "alt": 193},
  "Al Horford":              {"id": 817165,  "pos": "C",     "alt": 206},
  "Jimmy Butler":            {"id": 817017,  "pos": "SF",    "alt": 201},
  "Moses Moody":             {"id": 1132134, "pos": "SG",    "alt": 196},
  "Will Richard":            {"id": 1181668, "pos": "SG",    "alt": 193},
  "Quinten Post":            {"id": 1178523, "pos": "PF",    "alt": 213},
  "Pat Spencer":             {"id": 1392409, "pos": "PG",    "alt": 188},
  
  "Gary Payton II":          {"id": 855982,  "pos": "SG",    "alt": 188},
  "Gui Santos":              {"id": 1386098, "pos": "SF",    "alt": 198},
  "De'Anthony Melton":       {"id": 940779,  "pos": "PG",    "alt": 191},
},

"Houston Rockets": {
 "Alperen Sengun":        {"id": 1133822, "pos": "C",     "alt": 211},
  "Amen Thompson":         {"id": 1507963, "pos": "PG",    "alt": 201},
  "Jabari Smith Jr.":      {"id": 1178599, "pos": "PF",    "alt": 211},
  "Tari Eason":            {"id": 1178696, "pos": "SF",    "alt": 203},
  "Reed Sheppard":         {"id": 1599611, "pos": "SG",    "alt": 188},
  "Steven Adams":          {"id": 817066,  "pos": "C",     "alt": 211},
  "Dorian Finney-Smith":   {"id": 846931,  "pos": "PF",    "alt": 201},
  "Kevin Durant":          {"id": 817074,  "pos": "SF",    "alt": 211},
  "Fred VanVleet":         {"id": 846972,    "pos": "PG",    "alt": 183},
  "Josh Okogie":           {"id": 940774,    "pos": "SG",    "alt": 193},
  "Aaron Holiday":         {"id": 940789,    "pos": "PG",    "alt": 183},
  "Clint Capela":          {"id": 817022,    "pos": "C",     "alt": 208},
  "JD Davison":            {"id": 1179998,    "pos": "PG",    "alt": 191},
  "Jae'Sean Tate":         {"id": 1092887,    "pos": "SF",    "alt": 193},
  "Jeff Green":            {"id": 817126,    "pos": "PF",    "alt": 203},
  "Isaiah Crawford":       {"id": 1181088,    "pos": "SF",    "alt": 198},
},

"Indiana Pacers": {
"Tyrese Haliburton":     {"id": 1092923, "pos": "PG",    "alt": 196},
  "Pascal Siakam":       {"id": 846971,  "pos": "PF",    "alt": 203},
  "Andrew Nembhard":     {"id": 1181383, "pos": "PG",    "alt": 193},
  "Aaron Nesmith":       {"id": 1092913, "pos": "SF",    "alt": 198},
  "T.J. McConnell":      {"id": 823344,  "pos": "PG",    "alt": 185},
  "Jarace Walker":       {"id": 1430972, "pos": "PF",    "alt": 201},
  "Jay Huff":            {"id": 1415757, "pos": "C",     "alt": 216},
  "Obi Toppin":          {"id": 1092877, "pos": "PF",    "alt": 206},
  "Ben Sheppard":        {"id": 1181664, "pos": "SG",    "alt": 198},
  "Johnny Furphy":       {"id": 1599733, "pos": "SG",    "alt": 206},
  "Garrison Mathews":    {"id": 989020,  "pos": "SG",    "alt": 198},
  "Quenton Jackson":     {"id": 1178678, "pos": "SG",    "alt": 196},
  "Kam Jones":           {"id": 1179069, "pos": "SG",    "alt": 196},
  "James Wiseman":       {"id": 1092884, "pos": "C",     "alt": 211},
  "Ivica Zubac":         {"id": 846927,  "pos": "C",    "alt": 213},
  "Kobe Brown":          {"id": 1178349, "pos": "PF",   "alt": 201},
},

"Los Angeles Clippers": {
  "Darius Garland":     {"id": 987161,    "pos": "PG",    "alt": 185},
  "Kawhi Leonard":       {"id": 817229,  "pos": "SF",   "alt": 201},
  "Kris Dunn":           {"id": 846937,  "pos": "PG",   "alt": 191},
  "Nicolas Batum":       {"id": 817117,  "pos": "SF",   "alt": 203},
  "John Collins":        {"id": 885200,  "pos": "PF/C", "alt": 206},
  "Brook Lopez":         {"id": 817236,  "pos": "C",    "alt": 216},
  "Bradley Beal":        {"id": 816985,  "pos": "SG",   "alt": 193},
  "Bogdan Bogdanović":   {"id": 905908,  "pos": "SG",   "alt": 196},
  "Derrick Jones Jr.":   {"id": 855985,  "pos": "SF",   "alt": 198},
  "Isaiah Jackson":      {"id": 1133916, "pos": "PF/C",  "alt": 206},
  "Kobe Sanders":        {"id": 1182047, "pos": "SG",   "alt": 203},
  "Jordan Miller":       {"id": 1179642, "pos": "SF",   "alt": 196},
  "Cam Christie":        {"id": 1599629, "pos": "SG",   "alt": 196},
  "Bennedict Mathurin":  {"id": 1181316, "pos": "SG",    "alt": 196},
},

"Los Angeles Lakers": {
"LeBron James":          {"id": 817181,  "pos": "SF",   "alt": 206},
  "Luka Doncic":         {"id": 861608,  "pos": "PG/SG","alt": 201},
  "Marcus Smart":        {"id": 817375,  "pos": "PG",   "alt": 191},
  "Austin Reaves":       {"id": 1132140, "pos": "SG",   "alt": 196},
  "Rui Hachimura":       {"id": 988788,  "pos": "PF",   "alt": 203},
  "Deandre Ayton":       {"id": 940812,  "pos": "C",    "alt": 213},
  "Jake LaRavia":        {"id": 1179536, "pos": "SF",   "alt": 201},
  "Jarred Vanderbilt":   {"id": 940760,  "pos": "PF",   "alt": 203},
  "Luke Kennard":             {"id": 885256, "pos": "G", "alt": 196},
  "Jaxson Hayes":        {"id": 987100,  "pos": "C",    "alt": 213},
  "Dalton Knecht":       {"id": 1181979, "pos": "SG",   "alt": 198},
  "Maxi Kleber":         {"id": 861655,  "pos": "PF/C", "alt": 208},
  "Adou Thiero":         {"id": 1458373, "pos": "SF",   "alt": 203},
  "Bronny James":        {"id": 1600531, "pos": "PG",   "alt": 188}
},

"Memphis Grizzlies": {
  "Ja Morant":                {"id": 987102,  "pos": "PG",    "alt": 188},
  "Zach Edey":                {"id": 1178367, "pos": "C",     "alt": 224},
  "Santi Aldama":             {"id": 1133988, "pos": "PF",    "alt": 213},
  "Jaylen Wells":             {"id": 1600897, "pos": "SF",    "alt": 198},
  "Cedric Coward":            {"id": 1436586, "pos": "SG/SF", "alt": 198},
  "Kyle Anderson":          {"id": 816958,    "pos": "SF", "alt": 206},
  "Walter Clayton Jr.":     {"id": 1182375,   "pos": "PG", "alt": 191},
  "Taylor Hendricks":       {"id": 1431028,   "pos": "PF", "alt": 206},
  "Ty Jerome":                {"id": 988301,  "pos": "SG",    "alt": 196},
  "Kentavious Caldwell-Pope": {"id": 817020,  "pos": "SG",    "alt": 196},
  "Scotty Pippen Jr.":        {"id": 1386097, "pos": "PG",    "alt": 185},
  "Brandon Clarke":           {"id": 987934,  "pos": "PF",    "alt": 203},
  "GG Jackson II":            {"id": 1457768, "pos": "PF",    "alt": 206},
  "Cam Spencer":              {"id": 1180289, "pos": "PG",    "alt": 191},
  "Javon Small":              {"id": 1178514, "pos": "PG",    "alt": 188},
},

"Miami Heat": {
 "Bam Adebayo":            {"id": 885218,  "pos": "C",     "alt": 206},
  "Tyler Herro":           {"id": 988789,  "pos": "SG",    "alt": 196},
  "Jaime Jaquez Jr.":      {"id": 1178018, "pos": "SF",    "alt": 198},
  "Nikola Jovic":          {"id": 1084054, "pos": "PF",    "alt": 208},
  "Kel'el Ware":           {"id": 2015658, "pos": "C",     "alt": 213},
  "Pelle Larsson":         {"id": 1181318, "pos": "SG",    "alt": 198},
  "Terry Rozier":          {"id": 817351,  "pos": "PG",    "alt": 185},
  "Norman Powell":         {"id": 817332,  "pos": "SG",    "alt": 193},
  "Andrew Wiggins":        {"id": 817442,  "pos": "SF",    "alt": 201},
  "Davion Mitchell":       {"id": 1132132, "pos": "PG",    "alt": 183},
  "Simone Fontecchio":     {"id": 861512,  "pos": "SF",    "alt": 201},
  "Keshad Johnson":        {"id": 2015659, "pos": "PF",    "alt": 198},
  "Jahmir Young":          {"id": 1181073, "pos": "PG",    "alt": 188},
  "Dru Smith":             {"id": 1412285, "pos": "PG",    "alt": 188},
  "Myron Gardner":         {"id": 1181537, "pos": "SF",    "alt": 196},
  "Kasparas Jakucionis":   {"id": 1464826, "pos": "SG",    "alt": 198},
  "Vladislav Goldin":      {"id": 1179023, "pos": "C",     "alt": 216},
},

"Milwaukee Bucks": {
  "Giannis Antetokounmpo":    {"id": 816960,  "pos": "PF", "alt": 211},
  "Myles Turner":             {"id": 817414,  "pos": "C",  "alt": 211},
  "Kyle Kuzma":               {"id": 885215,  "pos": "PF", "alt": 206},
  "Bobby Portis":             {"id": 817329,  "pos": "PF", "alt": 208},
  "AJ Green":                 {"id": 1179862, "pos": "SG", "alt": 193},
  "Kevin Porter Jr.":         {"id": 987558,  "pos": "SG", "alt": 193},
  "Ryan Rollins":             {"id": 1178910, "pos": "PG", "alt": 193},
  "Taurean Prince":           {"id": 846893,  "pos": "SF", "alt": 198},
  "Gary Trent Jr.":           {"id": 940761,  "pos": "SG", "alt": 196},
  "Gary Harris":              {"id": 817139,  "pos": "SG", "alt": 193},
  
  "Jericho Sims":             {"id": 1133798, "pos": "C",  "alt": 208},

  "Andre Jackson Jr.":        {"id": 1179050, "pos": "GF", "alt": 198},
  "Thanasis Antetokounmpo":   {"id": 816961,  "pos": "PF", "alt": 201},
  "Alex Antetokounmpo":       {"id": 1132112, "pos": "PF", "alt": 202},
  "Pete Nance":               {"id": 1178272, "pos": "PF", "alt": 211},
},

"Minnesota Timberwolves": {
"Anthony Edwards":          {"id": 1092929, "pos": "SG", "alt": 193},
  "Julius Randle":          {"id": 817338,  "pos": "PF", "alt": 203},
  "Rudy Gobert":            {"id": 817111,  "pos": "C",  "alt": 216},
  "Naz Reid":               {"id": 1000618, "pos": "C",  "alt": 206},
  "Mike Conley":            {"id": 817038,  "pos": "PG", "alt": 185},
  "Jaden McDaniels":        {"id": 1092930, "pos": "SF", "alt": 206},
  "Donte DiVincenzo":       {"id": 940799,  "pos": "SG", "alt": 193},
  "Bones Hyland":           {"id": 1133811, "pos": "PG", "alt": 188},
  "Terrence Shannon Jr.":   {"id": 1180103, "pos": "SG", "alt": 198},
  "Jaylen Clark":           {"id": 1178026, "pos": "SG", "alt": 193},
  "Ayo Dosunmu":            {"id": 1133911, "pos": "SG",    "alt": 196},
  "Joan Beringer":          {"id": 1939566, "pos": "C",  "alt": 206},
  "Julian Phillips":        {"id": 1457779,    "pos": "SF", "alt": 203},
  "Joe Ingles":             {"id": 817175,  "pos": "SF", "alt": 206},
  "Johnny Juzang":          {"id": 1178022, "pos": "SF", "alt": 196},
  "Rocco Zikarsky":         {"id": 1540639, "pos": "C",  "alt": 220},
  "Enrique Freeman":        {"id": 1178464, "pos": "PF", "alt": 201},
},

"New Orleans Pelicans": {
  "Zion Williamson":     {"id": 987867,  "pos": "PF", "alt": 198},
  "Trey Murphy III":     {"id": 1133923, "pos": "SF", "alt": 203},
  "Dejounte Murray":     {"id": 846950,  "pos": "PG", "alt": 193},
  "Herbert Jones":       {"id": 1133919, "pos": "SF", "alt": 203},
  "Jordan Poole":        {"id": 989024,  "pos": "SG", "alt": 193},
  "Jeremiah Fears":      {"id": 1959659,    "pos": "PG", "alt": 193},
  "Saddiq Bey":          {"id": 1092874,    "pos": "SF", "alt": 201},
  "Derik Queen":         {"id": 1915309,    "pos": "C",  "alt": 208},
  
  "Trey Alexander":      {"id": 1178310,    "pos": "PG", "alt": 191},
  "Yves Missi":          {"id": 1602178,    "pos": "C",  "alt": 211},
  "Bryce McGowens":      {"id": 1180196,    "pos": "SG", "alt": 198},
  "Micah Peavy":         {"id": 1178716,    "pos": "SF", "alt": 203},
  "Jordan Hawkins":      {"id": 1179054,    "pos": "SG", "alt": 196},
  "Kevon Looney":        {"id": 817235,     "pos": "PF", "alt": 206},
  "Karlo Matkovic":      {"id": 982285,     "pos": "C",  "alt": 211},
  "Jaden Springer":      {"id": 1133927,    "pos": "SG", "alt": 193},
  "DeAndre Jordan":      {"id": 817205,     "pos": "C",  "alt": 211},
  "Hunter Dickinson":    {"id": 1178440,    "pos": "C",  "alt": 218},
},

# ===== BLOQUE 3 (21–30) =====

"New York Knicks": {
  "Jalen Brunson":        {"id": 940803,  "pos": "PG", "alt": 188},
  "Karl-Anthony Towns":   {"id": 817410,  "pos": "C",  "alt": 213},
  "Mikal Bridges":        {"id": 940807,  "pos": "SF", "alt": 198},
  "OG Anunoby":           {"id": 885250,  "pos": "SF", "alt": 201},
  "Miles McBride":        {"id": 1133771, "pos": "PG", "alt": 188},
  "Josh Hart":            {"id": 885211,  "pos": "SF", "alt": 196},
  "Landry Shamet":        {"id": 940768,  "pos": "SG", "alt": 193},
  "Jordan Clarkson":      {"id": 817034,  "pos": "SG", "alt": 191},
  "Mitchell Robinson":    {"id": 940771,  "pos": "C",  "alt": 213},
  "Tyler Kolek":          {"id": 1179063, "pos": "PG", "alt": 191},
  "Jose Alvarado":       {"id": 1133905,    "pos": "PG", "alt": 182},
  "Kevin McCullar Jr.":   {"id": 1180101, "pos": "SG", "alt": 201},
  "Pacome Dadiet":        {"id": 1189510, "pos": "SF", "alt": 200},
  "Mohamed Diawara":      {"id": 1191718, "pos": "PF", "alt": 204},
  "Ariel Hukporti":       {"id": 1184822, "pos": "C",  "alt": 213},
  "Trey Jemison":         {"id": 1181161, "pos": "C",  "alt": 208},
  "Dillon Jones":         {"id": 1181873, "pos": "SF", "alt": 198},
},

"Orlando Magic": {
  "Paolo Banchero":      {"id": 1181231, "pos": "PF", "alt": 208},
  "Franz Wagner":        {"id": 905191,  "pos": "SF", "alt": 208},
  "Jalen Suggs":         {"id": 1134017, "pos": "SG", "alt": 196},
  "Wendell Carter Jr.":  {"id": 940800,  "pos": "C",  "alt": 208},
  "Anthony Black":       {"id": 1458552, "pos": "PG", "alt": 201},
  "Tristan da Silva":    {"id": 1183279, "pos": "SF", "alt": 206},
  "Jevon Carter":      {"id": 940801,    "pos": "PG",    "alt": 185},
  "Desmond Bane":        {"id": 1092872, "pos": "SG", "alt": 196},
  "Noah Penda":          {"id": 1503503, "pos": "PF", "alt": 199},
  "Jett Howard":         {"id": 1433907, "pos": "SG", "alt": 198},
  "Goga Bitadze":        {"id": 958769,  "pos": "C",  "alt": 208},
  "Jonathan Isaac":      {"id": 885228,  "pos": "PF", "alt": 208},
  "Orlando Robinson":    {"id": 1182314, "pos": "C",  "alt": 208},
  "Jamal Cain":          {"id": 1180021, "pos": "SF", "alt": 198},
  "Moritz Wagner":       {"id": 940759,  "pos": "C",  "alt": 211},
},

"Oklahoma City Thunder": {
  "Shai Gilgeous-Alexander": {"id": 940794,  "pos": "PG", "alt": 198},
  "Chet Holmgren":           {"id": 1181394, "pos": "C",  "alt": 216},
  "Luguentz Dort":           {"id": 989021,  "pos": "SG", "alt": 193},
  "Jaylin Williams":         {"id": 1178575, "pos": "PF", "alt": 206},
  "Jalen Williams":          {"id": 1178085, "pos": "SF", "alt": 196},
  "Alex Caruso":             {"id": 855801,  "pos": "SG", "alt": 196},
  "Isaiah Joe":              {"id": 1092943, "pos": "SG", "alt": 191},
  "Isaiah Hartenstein":      {"id": 861639,  "pos": "C",  "alt": 213},
  "Cason Wallace":           {"id": 1458358, "pos": "PG", "alt": 191},
  "Ajay Mitchell":           {"id": 1182041, "pos": "PG", "alt": 193},
  "Branden Carlson":         {"id": 1183295, "pos": "C",  "alt": 213},
  "Mason Plumlee":           {"id": 817324, "pos": "FC",  "alt": 208},
  "Chris Youngblood":        {"id": 1180507, "pos": "SG", "alt": 193},
  "Aaron Wiggins":           {"id": 1133815, "pos": "SG", "alt": 196},
  "Kenrich Williams":        {"id": 957474,  "pos": "SF", "alt": 198},
  "Brooks Barnhizer":        {"id": 1178284, "pos": "SG", "alt": 198},
  "Nikola Topić":            {"id": 1176526, "pos": "PG", "alt": 198}
},

"Philadelphia 76ers": {
  "Tyrese Maxey":       {"id": 1092926, "pos": "PG", "alt": 188},
  "Paul George":        {"id": 817108,  "pos": "SF", "alt": 203},
  "Kelly Oubre Jr.":    {"id": 817307,  "pos": "SF", "alt": 201},
  "Andre Drummond":     {"id": 817069,  "pos": "C",  "alt": 211},
  "Joel Embiid":        {"id": 817081,  "pos": "C",  "alt": 213},
  "Quentin Grimes":     {"id": 1133792, "pos": "SG", "alt": 196},
  "VJ Edgecombe":       {"id": 1941186, "pos": "SG", "alt": 193},
  "Jared McCain":       {"id": 1600125, "pos": "PG", "alt": 191},
  "Dominick Barlow":    {"id": 1387556, "pos": "PF", "alt": 206},
  "Eric Gordon":        {"id": 817115,  "pos": "SG", "alt": 191},
  "Justin Edwards":     {"id": 1599610, "pos": "SF", "alt": 203},
  "Adem Bona":          {"id": 1441213, "pos": "C",  "alt": 206},
  "Jabari Walker":      {"id": 1183277, "pos": "PF", "alt": 201},
  "Trendon Watford":    {"id": 1133800, "pos": "PF", "alt": 203},
  "Johni Broome":       {"id": 1181677, "pos": "C",  "alt": 208},
  "Kyle Lowry":         {"id": 817239,  "pos": "PG", "alt": 183},
  "Hunter Sallis":      {"id": 1181398, "pos": "SG", "alt": 196},
  "Charles Bassey":     {"id": 1133907, "pos": "C",  "alt": 208},
},

"Phoenix Suns": {
  "Devin Booker":      {"id": 817000,  "pos": "SG", "alt": 196},
  "Grayson Allen":     {"id": 940814,  "pos": "SG", "alt": 193},
  "Royce O'Neale":     {"id": 905919,  "pos": "PF", "alt": 198},
  "Jalen Green":       {"id": 1133838, "pos": "SG", "alt": 193},
  "Dillon Brooks":     {"id": 885254,  "pos": "SF", "alt": 198},
  "Mark Williams":     {"id": 1181229, "pos": "C",  "alt": 218},
  "Collin Gillespie":  {"id": 1178405, "pos": "PG", "alt": 185},
  "Ryan Dunn":         {"id": 1433958,    "pos": "SG", "alt": 203},
  "Jordan Goodwin":    {"id": 1134112,    "pos": "PG", "alt": 193},
  "Oso Ighodaro":      {"id": 1179061,    "pos": "PF", "alt": 213},
  "Nigel Hayes":       {"id": 905897,     "pos": "PF", "alt": 203},
  "Nick Richards":     {"id": 1092907,    "pos": "C",  "alt": 213},
  "Isaiah Livers":     {"id": 1133845,    "pos": "SF", "alt": 198},
  "Khaman Maluach":    {"id": 1915077,    "pos": "C",  "alt": 216},
  "Rasheer Fleming":   {"id": 1431085,    "pos": "PF", "alt": 206},
  "Koby Brea":         {"id": 1179740,    "pos": "SG", "alt": 201},
  "CJ Huntley":        {"id": 1183164,    "pos": "PF", "alt": 211},
  "Cole Anthony":      {"id": 1092865, "pos": "PG", "alt": 188},
  "Amir Coffey":       {"id": 989007,  "pos": "SF", "alt": 201},
},

"Portland Trail Blazers": {
 "Jerami Grant":        {"id": 817119,  "pos": "PF", "alt": 203},
  "Shaedon Sharpe":      {"id": 1178822, "pos": "SG", "alt": 196},
  "Donovan Clingan":     {"id": 1433891, "pos": "C",  "alt": 218},
  "Deni Avdija":         {"id": 913627,  "pos": "SF", "alt": 206},
  "Toumani Camara":      {"id": 1179735, "pos": "SF", "alt": 203},
  "Damian Lillard":      {"id": 817234,  "pos": "PG", "alt": 188},
  "Jrue Holiday":        {"id": 817159,  "pos": "SG", "alt": 193},
  "Matisse Thybulle":    {"id": 989085,  "pos": "SG", "alt": 196},
  "Kris Murray":         {"id": 1180989, "pos": "PF", "alt": 203},
  "Scoot Henderson":     {"id": 1206071, "pos": "PG", "alt": 191},
  "Caleb Love":          {"id": 1178849, "pos": "SG", "alt": 193},
  "Sidy Cissoko":        {"id": 1144809, "pos": "SG", "alt": 198},
  "Robert Williams III": {"id": 940756,  "pos": "C",  "alt": 206},
  "Blake Wesley":        {"id": 1179467, "pos": "PG", "alt": 191},
  "Rayan Rupert":        {"id": 1412271, "pos": "SF", "alt": 198},
  "Javonte Cook":        {"id": 1510810, "pos": "SF", "alt": 191},
  "Yang Hansen":         {"id": 1503509, "pos": "C",  "alt": 218},
},

"Sacramento Kings": {
"Domantas Sabonis":     {"id": 846968,  "pos": "C",  "alt": 211},
  "DeMar DeRozan":        {"id": 817057,  "pos": "SF", "alt": 198},
  "Malik Monk":           {"id": 885258,  "pos": "SG", "alt": 191},
  "Russell Westbrook":    {"id": 817437,  "pos": "PG", "alt": 191},
  "Zach LaVine":          {"id": 817218,  "pos": "SG", "alt": 196},
  "De'Andre Hunter":    {"id": 988300,    "pos": "SF",    "alt": 203},
  "Precious Achiuwa":     {"id": 1092863, "pos": "PF", "alt": 203},
  "Keegan Murray":        {"id": 1180986, "pos": "PF", "alt": 203},
  "Isaac Jones":          {"id": 1436626, "pos": "C",  "alt": 206},
  "Dario Saric":          {"id": 846947,  "pos": "PF", "alt": 208},
  "Maxime Raynaud":       {"id": 1178001, "pos": "C",  "alt": 216},
  "Keon Ellis":           {"id": 1179991, "pos": "SG", "alt": 191},
  "Drew Eubanks":         {"id": 957172,  "pos": "C",  "alt": 208},
  "Nique Clifford":       {"id": 1183280, "pos": "SG", "alt": 198},
  "Devin Carter":         {"id": 1178805, "pos": "SG", "alt": 191},
  "Isaiah Stevens":       {"id": 1178034, "pos": "PG", "alt": 183},
  "Daeqwon Plowden":      {"id": 1178896, "pos": "SF", "alt": 198},
  "Dylan Cardwell":       {"id": 1178595, "pos": "C",  "alt": 211},
},

"San Antonio Spurs": {
 "Victor Wembanyama":   {"id": 998725,  "pos": "C",  "alt": 224},
  "Stephon Castle":      {"id": 1599025, "pos": "SG", "alt": 198},
  "Julian Champagnie":   {"id": 1178394, "pos": "SF", "alt": 201},
  "Keldon Johnson":      {"id": 987552,  "pos": "SF", "alt": 196},
  "Devin Vassell":       {"id": 1092911, "pos": "SG", "alt": 196},
  "De'Aaron Fox":        {"id": 885244,  "pos": "PG", "alt": 191},
  "Luke Kornet":         {"id": 905898,  "pos": "C",  "alt": 218},
  "Harrison Barnes":     {"id": 816977,    "pos": "PF", "alt": 203},
  "Dylan Harper":        {"id": 1503356,    "pos": "SG", "alt": 198},
  "Jordan McLaughlin":   {"id": 951089,    "pos": "PG", "alt": 183},
  "Jeremy Sochan":       {"id": 1179822,    "pos": "PF", "alt": 203},
  "Kelly Olynyk":        {"id": 817306,    "pos": "C",  "alt": 211},
  "Lindy Waters III":    {"id": 1182205,    "pos": "SG", "alt": 198},
  "Carter Bryant":       {"id": 1947115,    "pos": "PF", "alt": 203},
  "Bismack Biyombo":     {"id": 816990,    "pos": "C",  "alt": 203},
  "David Jones":         {"id": 1180244,    "pos": "SF", "alt": 198},
  "Harrison Ingram":     {"id": 1178002,    "pos": "PF", "alt": 201},
  "Riley Minix":         {"id": 1599804,    "pos": "PF", "alt": 201}

},

"Toronto Raptors": {
 "Scottie Barnes":             {"id": 1133784, "pos": "PF", "alt": 206},
  "RJ Barrett":                {"id": 988995,  "pos": "SG", "alt": 198},
  "Jakob Poeltl":              {"id": 846973,  "pos": "C",  "alt": 213},
  "Gradey Dick":               {"id": 1433917, "pos": "SG", "alt": 201},
  "Immanuel Quickley":         {"id": 1092912, "pos": "PG", "alt": 191},
  "Sandro Mamukelashvili":     {"id": 1134004, "pos": "PF", "alt": 211},
  "Brandon Ingram":            {"id": 846926,  "pos": "SF", "alt": 203},
  "Collin Murray-Boyles":      {"id": 1599104, "pos": "PF", "alt": 201},
  "Ja'Kobe Walter":            {"id": 1602175, "pos": "SG", "alt": 193},
  "Jamal Shead":               {"id": 1179509, "pos": "PG", "alt": 183},
  "Jamison Battle":            {"id": 1180204, "pos": "SF", "alt": 201},
  "Ochai Agbaji":              {"id": 1182725, "pos": "SG", "alt": 196},
  "Chucky Hepburn":            {"id": 1178672, "pos": "PG", "alt": 188},
  "Garrett Temple":            {"id": 817397,  "pos": "SF", "alt": 196},
  "Jonathan Mogbo":            {"id": 1457921, "pos": "PF", "alt": 203},
  "A.J. Lawson":               {"id": 1392408, "pos": "SG", "alt": 198},
  "Trayce Jackson-Davis":    {"id": 1178292, "pos": "PF/C",  "alt": 206},
},

"Utah Jazz": {
  "Lauri Markkanen":        {"id": 885207,  "pos": "PF", "alt": 213},
  "Collin Sexton":          {"id": 940769,  "pos": "SG", "alt": 185},
  "Keyonte George":         {"id": 1430963, "pos": "PG", "alt": 193},
  "Walker Kessler":         {"id": 1178659, "pos": "C",  "alt": 216},
  "Kyle Filipowski":        {"id": 1430922, "pos": "PF", "alt": 211},
  "Jusuf Nurkic":           {"id": 817300,  "pos": "C",  "alt": 211},
  "Ace Bailey":             {"id": 1894884,   "pos": "SF", "alt": 208},
  "Isaiah Collier":         {"id": 1600527,   "pos": "PG", "alt": 196},
  "Svi Mykhailiuk":         {"id": 940776,    "pos": "SG", "alt": 201},
  "Brice Sensabaugh":       {"id": 1436499,   "pos": "SG", "alt": 196},
  "Kevin Love":             {"id": 817238,    "pos": "PF", "alt": 203},
  "Cody Williams":          {"id": 1599362,   "pos": "SF", "alt": 203},
  "EJ Harkless":            {"id": 1180086,   "pos": "SG", "alt": 191},
  "Lonzo Ball":             {"id": 885217,    "pos": "PG",    "alt": 198},
  "Jaren Jackson Jr.":      {"id": 940785,  "pos": "PF/C",  "alt": 208},
  "Vince Williams Jr.":     {"id": 1178858, "pos": "SG/SF", "alt": 193},
  "Jock Landale":           {"id": 954950,  "pos": "C",     "alt": 211},
  "John Konchar":           {"id": 988875,  "pos": "SG",    "alt": 196},
  "Oscar Tshiebwe":         {"id": 1178811,   "pos": "C",  "alt": 203},
  "Georges Niang":          {"id": 846946,    "pos": "PF", "alt": 201},
},

"Washington Wizards": {
  "Alexandre Sarr":        {"id": 1540633, "pos": "C",  "alt": 216},
  "Carlton Carrington":    {"id": 1599182, "pos": "PG", "alt": 196},
  "Bilal Coulibaly":       {"id": 1410380, "pos": "SF", "alt": 201},
  "Trae Young":            {"id": 940755,  "pos": "PG", "alt": 185},
  "Tre Johnson":           {"id": 1503358, "pos": "SG", "alt": 198},
  "Kyshawn George":        {"id": 1542321, "pos": "SF", "alt": 201},
  "Justin Champagnie":     {"id": 1133778, "pos": "PF", "alt": 198},
  "Tristan Vukcevic":      {"id": 1085856, "pos": "C",  "alt": 213},
  "Cam Whitmore":          {"id": 1433908, "pos": "SF", "alt": 201},
  "Jamir Watkins":         {"id": 1178866, "pos": "SF", "alt": 201},
  "Will Riley":            {"id": 1941973, "pos": "SF", "alt": 203},
  "D'Angelo Russell":    {"id": 817357,  "pos": "PG",    "alt": 193},
  "Anthony Davis":       {"id": 817054,  "pos": "PF/C",  "alt": 208},
  "Jaden Hardy":         {"id": 1206072, "pos": "SG",    "alt": 191},
  "Anthony Gill":          {"id": 905681,  "pos": "PF", "alt": 203},
  "Sharife Cooper":        {"id": 1133794, "pos": "PG", "alt": 185}, 

},
}
# ============================================================================
# ESTILOS CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    
    :root {
        --primary: #00D9FF;
        --secondary: #0A4D68;
        --accent: #00FFA3;
        --warning: #FFD93D;
        --danger: #FF6B6B; 
        --dark-blue: #05161A;
        --card-blue: #0B2027;
        --mid-blue: #112B35;
        --text-primary: #E0F4FF;
        --text-secondary: #B8D4E0;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--dark-blue) 0%, #0A1F2E 100%);
    }
    
    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        color: var(--text-primary) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        color: var(--accent) !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--card-blue), var(--dark-blue));
        border-right: 1px solid #1A3A4A;
    }
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, var(--card-blue), var(--mid-blue));
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 8px;
    }
    
    /* SELECTBOX VISIBLE */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--mid-blue) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div > div,
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] p {
        color: #E0F4FF !important;
        font-weight: 500 !important;
    }
    
    div[data-baseweb="popover"] > div,
    ul[data-baseweb="menu"] {
        background-color: var(--card-blue) !important;
        border: 2px solid var(--primary) !important;
        border-radius: 8px !important;
    }
    
    li[role="option"], li[data-baseweb="option"] {
        background-color: var(--card-blue) !important;
        color: #E0F4FF !important;
        padding: 10px 15px !important;
    }
    
    li[role="option"] *, li[data-baseweb="option"] * {
        color: #E0F4FF !important;
    }
    
    li[role="option"]:hover, li[data-baseweb="option"]:hover {
        background-color: var(--mid-blue) !important;
        color: var(--accent) !important;
    }
    
    li[role="option"]:hover *, li[data-baseweb="option"]:hover * {
        color: var(--accent) !important;
    }
    
    li[aria-selected="true"] {
        background-color: rgba(0, 217, 255, 0.2) !important;
        color: var(--primary) !important;
    }
    
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        color: var(--dark-blue) !important;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 255, 163, 0.4);
    }

    .stat-card {
        background: linear-gradient(135deg, var(--card-blue), var(--mid-blue));
        border-left: 4px solid var(--primary);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }

    [data-testid="stMetricValue"] {
        color: var(--accent) !important;
    }
    
    .alert-info {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.15), rgba(0, 217, 255, 0.05));
        border-left: 4px solid var(--primary);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.15), rgba(255, 107, 107, 0.05));
        border-left: 4px solid var(--danger);
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        color: var(--text-primary) !important;
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(0, 255, 163, 0.15), rgba(0, 255, 163, 0.05));
        border-left: 4px solid var(--accent);
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        color: var(--text-primary) !important;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(255, 217, 61, 0.15), rgba(255, 217, 61, 0.05));
        border-left: 4px solid var(--warning);
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        color: var(--text-primary) !important;
    }
    
    p, span, div {
        color: var(--text-primary) !important;
    }
    
    .stMarkdown p, .stMarkdown span {
        color: var(--text-primary) !important;
    }
</style>
""", unsafe_allow_html=True)


def mostrar_progreso_detallado(jugador_actual, equipo_actual, progreso_pct, total_jugadores, procesados):
    """Muestra barra de progreso con detalles del jugador actual"""
    
    st.markdown(f"""
    <style>
        .progress-card {{
            background: linear-gradient(135deg, var(--card-blue), var(--mid-blue));
            border-left: 4px solid var(--primary);
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 8px 25px rgba(0, 217, 255, 0.2);
        }}
        
        .progress-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .current-player {{
            font-size: 24px;
            font-weight: 700;
            color: #00FFA3;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .player-badge {{
            background: linear-gradient(135deg, #00D9FF, #00FFA3);
            color: #05161A;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }}
        
        .progress-stats {{
            display: flex;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .stat-item {{
            display: flex;
            flex-direction: column;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #B8D4E0;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 20px;
            font-weight: 700;
            color: #00D9FF;
        }}
        
        @keyframes slideIn {{
            from {{
                transform: translateX(-20px);
                opacity: 0;
            }}
            to {{
                transform: translateX(0);
                opacity: 1;
            }}
        }}
        
        .animated-entry {{
            animation: slideIn 0.3s ease-out;
        }}
    </style>
    
    <div class="progress-card animated-entry">
        <div class="progress-header">
            <div class="current-player">
                🏀 {jugador_actual}
                <span class="player-badge">{equipo_actual}</span>
            </div>
        </div>
        
        <div class="progress-stats">
            <div class="stat-item">
                <div class="stat-label">Progreso</div>
                <div class="stat-value">{progreso_pct:.1f}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Procesados</div>
                <div class="stat-value">{procesados}/{total_jugadores}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Restantes</div>
                <div class="stat-value">{total_jugadores - procesados}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# FUNCIONES AUXILIARES BÁSICAS
# ============================================================================

def calcular_indice_consistencia(serie):
    """Calcula el índice de consistencia basado en coeficiente de variación"""
    if len(serie) < 3:
        return "Sin Datos", 0
    cv = serie.std() / serie.mean() if serie.mean() > 0 else 0
    if cv < 0.15:
        return "🔥 Muy Estable", cv
    if cv < 0.30:
        return "✅ Estable", cv
    if cv < 0.45:
        return "⚠️ Inestable", cv
    return "🎲 Volátil", cv


def calcular_tendencia_reciente(df_jug, metrica, num_partidos_recientes=3):
    """Calcula si el jugador está mejorando en los últimos N partidos"""
    if len(df_jug) < num_partidos_recientes + 3:
        return "Sin Datos", 0
    
    ultimos = df_jug.tail(num_partidos_recientes)[metrica].mean()
    previos = df_jug.head(len(df_jug) - num_partidos_recientes).tail(num_partidos_recientes)[metrica].mean()
    
    diferencia = ultimos - previos
    porcentaje = (diferencia / previos * 100) if previos > 0 else 0
    
    if porcentaje > 10:
        return "📈 Mejorando", porcentaje
    elif porcentaje < -10:
        return "📉 Decayendo", porcentaje
    else:
        return "➡️ Estable", porcentaje

# ============================================================================
# FUNCIONES DE GRÁFICAS MEJORADAS
# ============================================================================

def crear_grafica_apilada_mejorada(df_equipo, metrica, titulo, colores_jugadores):
    """Gráfica apilada con fechas corregidas"""
    df_work = df_equipo.copy().sort_values('Fecha')
    df_work['Fecha_Label'] = df_work['Fecha'].dt.strftime('%d/%m')
    
    df_pivot = df_work.pivot_table(
        index=['Fecha', 'Fecha_Label'],
        columns='Jugador',
        values=metrica,
        aggfunc='sum',
        fill_value=0
    ).reset_index().set_index('Fecha_Label')
    
    df_pivot = df_pivot.sort_values('Fecha')
    if 'Fecha' in df_pivot.columns: 
        del df_pivot['Fecha']

    fig = go.Figure()

    for jugador in df_pivot.columns:
        valores = df_pivot[jugador]
        if valores.sum() > 0:
            fig.add_trace(go.Bar(
                name=jugador,
                x=df_pivot.index,
                y=valores,
                marker=dict(
                    color=colores_jugadores.get(jugador, '#6B9EB0'),
                    line=dict(color='rgba(0, 0, 0, 0.3)', width=1)
                ),
                hovertemplate=f'<b>{jugador}</b><br>{metrica}: %{{y:.1f}}<br>Fecha: %{{x}}<extra></extra>'
            ))

    fig.update_layout(
        title={'text': titulo, 'font': {'size': 20, 'family': 'Space Mono, monospace', 'color': '#00FFA3'}, 'x': 0.02},
        barmode='stack',
        paper_bgcolor='rgba(11, 32, 39, 0.5)',
        plot_bgcolor='rgba(11, 32, 39, 0.8)',
        height=500,
        xaxis=dict(
            type='category',
            title=dict(text='Últimos Partidos NBA', font=dict(color='#E0F4FF', size=14)),
            gridcolor='rgba(26, 58, 74, 0.3)',
            color='#E0F4FF',
            tickangle=-45
        ),
        yaxis=dict(
            title=dict(text=f'Total {metrica}', font=dict(color='#E0F4FF', size=14)),
            gridcolor='rgba(26, 58, 74, 0.3)',
            color='#E0F4FF'
        ),
        legend=dict(
            orientation='v', x=1.02, y=1,
            bgcolor='rgba(11, 32, 39, 0.9)',
            font=dict(color="#E0F4FF", size=10)
        ),
        hovermode='x unified',
        bargap=0.15,
        margin=dict(l=60, r=150, t=60, b=80)
    )
    return fig


def crear_grafica_individual_mejorada(df_jug, metrica, linea, nombre_jugador):
    """Gráfica individual con tema oscuro"""
    df_plot = df_jug.copy().sort_values('Fecha')
    df_plot['Fecha_Str'] = df_plot['Fecha'].dt.strftime('%d/%m')
    
    fig = go.Figure()
    
    fig.add_hline(
        y=linea,
        line_dash="dash",
        line_color="#FFD93D",
        line_width=2,
        annotation_text=f"Línea {linea}",
        annotation_position="top right",
        annotation_font_color="#FFD93D"
    )
    
    colores_barras = ['#00FFA3' if v >= linea else '#FF6B6B' for v in df_plot[metrica]]
    
    fig.add_trace(go.Bar(
        x=df_plot['Fecha_Str'],
        y=df_plot[metrica],
        marker=dict(
            color=colores_barras,
            line=dict(color='rgba(0, 0, 0, 0.4)', width=1.5)
        ),
        text=df_plot[metrica].round(1),
        textposition="outside",
        textfont=dict(size=11, color='#E0F4FF'),
        hovertemplate="<b>%{x}</b><br>Valor: %{y:.1f}<extra></extra>",
        name=metrica
    ))
    
    if len(df_plot) >= 3:
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha_Str'],
            y=df_plot[metrica].rolling(window=3, min_periods=1).mean(),
            mode='lines',
            line=dict(color='#00D9FF', width=2, dash='dot'),
            name='Tendencia',
            hovertemplate='Promedio: %{y:.1f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'📊 Evolución de {metrica} - {nombre_jugador}',
            font=dict(size=18, color='#00FFA3'),
            x=0.02
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(11, 32, 39, 0.5)',
        plot_bgcolor='rgba(11, 32, 39, 0.8)',
        height=450,
        xaxis=dict(type='category', title='Fecha', color='#E0F4FF'),
        yaxis=dict(title=metrica, color='#E0F4FF', gridcolor='rgba(26, 58, 74, 0.3)'),
        legend=dict(bgcolor='rgba(11, 32, 39, 0.8)', font=dict(color='#E0F4FF')),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    return fig

# ============================================================================
# FUNCIONES DE DATOS Y SCRAPING
# ============================================================================

def obtener_datos_partido(nombre_equipo):
    """Detecta el próximo juego del equipo"""
    try:
        nba_team = teams.find_teams_by_full_name(nombre_equipo)
        if not nba_team:
            return None
        team_id = nba_team[0]['id']
        
        contexto = {"hay_juego": False, "rival": None, "localia": None, "fecha": None}
        
        for dias_adelante in range(8):
            try:
                fecha_busqueda = obtener_fecha_panama() + timedelta(days=dias_adelante)
                fecha_str = fecha_busqueda.strftime('%m/%d/%Y')
                
                sb = scoreboardv2.ScoreboardV2(game_date=fecha_str)
                juegos = sb.get_data_frames()[0]
                
                partido = juegos[(juegos['HOME_TEAM_ID'] == team_id) | (juegos['VISITOR_TEAM_ID'] == team_id)]
                
                if not partido.empty:
                    row = partido.iloc[0]
                    contexto["hay_juego"] = True
                    es_home = row['HOME_TEAM_ID'] == team_id
                    contexto["localia"] = "Local" if es_home else "Visitante"
                    
                    rival_id = row['VISITOR_TEAM_ID'] if es_home else row['HOME_TEAM_ID']
                    rival_info = teams.find_team_name_by_id(rival_id)
                    contexto["rival"] = rival_info['full_name'] if rival_info else None
                    
                    if dias_adelante == 0:
                        contexto["fecha"] = "Hoy"
                    elif dias_adelante == 1:
                        contexto["fecha"] = "Mañana"
                    else:
                        contexto["fecha"] = f"En {dias_adelante} días"
                    
                    return contexto
            except:
                continue
        
        return contexto
    except:
        return None


def obtener_jugadores_lesionados(team_name):
    """Obtiene jugadores lesionados"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    
    team_id = TEAM_IDS.get(team_name)
    if not team_id:
        return pd.DataFrame()
    
    lesionados = []
    
    try:
        url_next = f"https://api.sofascore.com/api/v1/team/{team_id}/events/next/0"
        response = requests.get(url_next, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('events'):
                event_id = data['events'][0]['id']
                
                url_lineup = f"https://api.sofascore.com/api/v1/event/{event_id}/lineups"
                time.sleep(0.3)
                response_lineup = requests.get(url_lineup, headers=headers, timeout=10)
                
                if response_lineup.status_code == 200:
                    lineup_data = response_lineup.json()
                    
                    is_home = data['events'][0]['homeTeam']['id'] == team_id
                    team_key = 'home' if is_home else 'away'
                    
                    missing = lineup_data.get(team_key, {}).get('missingPlayers', [])
                    
                    for player in missing:
                        player_info = player.get('player', {})
                        lesionados.append({
                            'Jugador': player_info.get('name', 'N/A'),
                            'ID': player_info.get('id', 0),
                            'Razon': player.get('reason', 'Unknown'),
                            'Tipo': player.get('type', 'Unknown'),
                            'Confirmado': True
                        })
    
    except Exception:
        pass
    
    return pd.DataFrame(lesionados)


@st.cache_data(show_spinner=False, ttl=3600)
def scrapear_jugador(player_id, nombre_jugador, equipo_sel, cantidad=7):
    """Extrae estadísticas del jugador"""
    lista_stats = []
    scraper = requests.Session() 
    
    info_jugador = JUGADORES_DB.get(equipo_sel, {}).get(nombre_jugador, {})
    altura = info_jugador.get("alt", 0)
    posicion = info_jugador.get("pos", "N/A")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'es-ES,es;q=0.9',
        'Referer': 'https://www.sofascore.com/',
        'Origin': 'https://www.sofascore.com',
    }
    
    try:
        url_lista = f"https://api.sofascore.com/api/v1/player/{player_id}/events/last/0"
        response = scraper.get(url_lista, headers=headers, impersonate="chrome120", timeout=CONFIG['TIMEOUT_SCRAPING'])
        response.raise_for_status()
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
            response_stats.raise_for_status()
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
        st.error(f"❌ Error en {nombre_jugador}: {str(e)}")
        
    return pd.DataFrame(lista_stats)

# ============================================================================
# MACHINE LEARNING MEJORADO V2
# ============================================================================

def crear_features_ml_v2(df_jugador, target_col='Puntos'):
    """15 features mejoradas"""
    df = df_jugador.copy().sort_values('Fecha')
    features = []
    targets = []
    
    for i in range(3, len(df)):
        window = df.iloc[max(0, i-7):i]
        actual_len = len(window)
        
        avg_3 = window.tail(3)[target_col].mean()
        avg_7 = window[target_col].mean()
        
        feature_dict = {
            'avg_3_games': avg_3,
            'avg_7_games': avg_7,
            'trend_5': np.polyfit(range(min(5, len(window))), window.tail(5)[target_col].values, 1)[0] if len(window) >= 2 else 0,
            'std_recent': window[target_col].std() if actual_len > 1 else 0,
            'coef_var': (window[target_col].std() / window[target_col].mean()) if window[target_col].mean() > 0 else 0,
            'dias_descanso': df.iloc[i]['Dias_Descanso'] if pd.notna(df.iloc[i]['Dias_Descanso']) else 2,
            'es_local': 1 if df.iloc[i]['Localia'] == 'Local' else 0,
            'minutos_avg': window['Minutos'].mean(),
            'eficiencia_avg': window['Eficiencia'].mean(),
            'max_recent_3': window.tail(3)[target_col].max(),
            'min_recent_3': window.tail(3)[target_col].min(),
            'rango_recent': window.tail(3)[target_col].max() - window.tail(3)[target_col].min(),
            'racha_positiva': sum(1 for x in window.tail(3)[target_col] if x > window[target_col].mean()),
            'rendimiento_b2b': window[window['Dias_Descanso'] <= 1][target_col].mean() if len(window[window['Dias_Descanso'] <= 1]) > 0 else window[target_col].mean(),
            'diff_local_visita': (window[window['Localia']=='Local'][target_col].mean() - window[window['Localia']=='Visitante'][target_col].mean()) if len(window[window['Localia']=='Local']) > 0 and len(window[window['Localia']=='Visitante']) > 0 else 0,
        }
        
        features.append(feature_dict)
        targets.append(df.iloc[i][target_col])
    
    return pd.DataFrame(features), np.array(targets)


def entrenar_modelo_v2(df_equipo, jugador_nombre, target_col='Puntos'):
    """XGBoost optimizado"""
    df_jug = df_equipo[df_equipo['Jugador'] == jugador_nombre].copy()
    if len(df_jug) < 5:
        return None, None, None
    
    X, y = crear_features_ml_v2(df_jug, target_col)
    if len(X) < 2:
        return None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    
    modelo.fit(X_scaled, y)
    
    y_pred = modelo.predict(X_scaled)
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    
    importances = dict(zip(X.columns, modelo.feature_importances_))
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    return modelo, scaler, {
        'mae': mae,
        'rmse': rmse,
        'importance': sorted_imp,
        'n_samples': len(X)
    }


def predecir_proximo_partido_v2(modelo, scaler, df_jugador, target_col='Puntos', es_local=True, dias_descanso=2):
    """Predicción con intervalo dinámico"""
    if modelo is None:
        return None, None
    
    df = df_jugador.copy().sort_values('Fecha')
    window = df.tail(7)
    
    avg_3 = window.tail(3)[target_col].mean()
    avg_7 = window[target_col].mean()
    
    features_prox = {
        'avg_3_games': avg_3,
        'avg_7_games': avg_7,
        'trend_5': np.polyfit(range(min(5, len(window))), window.tail(5)[target_col].values, 1)[0] if len(window) >= 2 else 0,
        'std_recent': window[target_col].std() if len(window) > 1 else 0,
        'coef_var': (window[target_col].std() / window[target_col].mean()) if window[target_col].mean() > 0 else 0,
        'dias_descanso': dias_descanso,
        'es_local': 1 if es_local else 0,
        'minutos_avg': window['Minutos'].mean(),
        'eficiencia_avg': window['Eficiencia'].mean(),
        'max_recent_3': window.tail(3)[target_col].max(),
        'min_recent_3': window.tail(3)[target_col].min(),
        'rango_recent': window.tail(3)[target_col].max() - window.tail(3)[target_col].min(),
        'racha_positiva': sum(1 for x in window.tail(3)[target_col] if x > window[target_col].mean()),
        'rendimiento_b2b': window[window['Dias_Descanso'] <= 1][target_col].mean() if len(window[window['Dias_Descanso'] <= 1]) > 0 else window[target_col].mean(),
        'diff_local_visita': (window[window['Localia']=='Local'][target_col].mean() - window[window['Localia']=='Visitante'][target_col].mean()) if len(window[window['Localia']=='Local']) > 0 and len(window[window['Localia']=='Visitante']) > 0 else 0,
    }
    
    X_prox_scaled = scaler.transform(pd.DataFrame([features_prox]))
    prediccion = modelo.predict(X_prox_scaled)[0]
    
    coef_var = features_prox['coef_var']
    
    if coef_var < 0.2:
        margen = prediccion * 0.15
    elif coef_var < 0.35:
        margen = prediccion * 0.25
    else:
        margen = prediccion * 0.35
    
    intervalo = (max(0, prediccion - margen), prediccion + margen)
    
    return prediccion, intervalo

# ============================================================================
# SISTEMA DE ANÁLISIS DE IMPACTO POR LESIONES
# ============================================================================

def calcular_impacto_ausencias(df_equipo, jugadores_ausentes, metrica='Puntos'):
    """
    Calcula el impacto de jugadores ausentes en el equipo
    
    Returns:
        dict con impacto por métrica y redistribución esperada
    """
    if not jugadores_ausentes:
        return {'impacto_total': 0, 'redistribucion': {}}
    
    impacto = {
        'jugadores_out': [],
        'produccion_perdida': {},
        'redistribucion_estimada': {},
        'beneficiarios_principales': []
    }
    
    # Calcular producción perdida
    for jugador_out in jugadores_ausentes:
        df_jug = df_equipo[df_equipo['Jugador'] == jugador_out]
        
        if not df_jug.empty:
            stats_perdidas = {
                'Puntos': df_jug['Puntos'].mean(),
                'Rebotes': df_jug['Rebotes'].mean(),
                'Asistencias': df_jug['Asistencias'].mean(),
                'Minutos': df_jug['Minutos'].mean()
            }
            
            impacto['jugadores_out'].append({
                'nombre': jugador_out,
                'stats': stats_perdidas
            })
            
            for stat, val in stats_perdidas.items():
                impacto['produccion_perdida'][stat] = impacto['produccion_perdida'].get(stat, 0) + val
    
    # Identificar beneficiarios (jugadores que históricamente mejoran cuando otros faltan)
    if impacto['jugadores_out']:
        minutos_disponibles = impacto['produccion_perdida'].get('Minutos', 0)
        
        # Obtener jugadores activos del equipo
        equipo_nombre = df_equipo['Equipo'].iloc[0]
        jugadores_activos = [j for j in df_equipo['Jugador'].unique() if j not in jugadores_ausentes]
        
        # Calcular quién se beneficia más (top 3 en minutos que no están lesionados)
        df_activos = df_equipo[df_equipo['Jugador'].isin(jugadores_activos)]
        top_beneficiarios = df_activos.groupby('Jugador')['Minutos'].mean().sort_values(ascending=False).head(3)
        
        # Redistribuir proporcionalmente
        total_minutos_top3 = top_beneficiarios.sum()
        
        for jugador, minutos_avg in top_beneficiarios.items():
            proporcion = minutos_avg / total_minutos_top3 if total_minutos_top3 > 0 else 0
            
            # Estimar aumento en cada stat
            impacto['redistribucion_estimada'][jugador] = {
                'Puntos': impacto['produccion_perdida'].get('Puntos', 0) * proporcion * 0.7,  # 70% se redistribuye
                'Rebotes': impacto['produccion_perdida'].get('Rebotes', 0) * proporcion * 0.7,
                'Asistencias': impacto['produccion_perdida'].get('Asistencias', 0) * proporcion * 0.7,
                'Minutos_extra': minutos_disponibles * proporcion
            }
            
            impacto['beneficiarios_principales'].append(jugador)
    
    return impacto


def ajustar_prediccion_por_contexto(prediccion_base, jugador_nombre, df_equipo, 
                                     lesionados_df, metrica='Puntos'):
    """
    Ajusta la predicción considerando lesiones de compañeros y regresos
    
    Returns:
        prediccion_ajustada, dict con explicación del ajuste
    """
    ajuste_info = {
        'prediccion_original': prediccion_base,
        'ajustes_aplicados': [],
        'prediccion_final': prediccion_base,
        'confianza_ajuste': 'media'
    }
    
    if lesionados_df.empty:
        return prediccion_base, ajuste_info
    
    equipo_jugador = df_equipo[df_equipo['Jugador'] == jugador_nombre]['Equipo'].iloc[0]
    
    # Obtener lesionados del mismo equipo
    jugadores_out = lesionados_df['Jugador'].tolist()
    jugadores_out_equipo = [j for j in jugadores_out if j in JUGADORES_DB.get(equipo_jugador, {}).keys()]
    
    # Calcular impacto
    impacto = calcular_impacto_ausencias(df_equipo, jugadores_out_equipo, metrica)
    
    # AJUSTE 1: Si el jugador está entre los beneficiarios
    if jugador_nombre in impacto.get('beneficiarios_principales', []):
        boost = impacto['redistribucion_estimada'][jugador_nombre][metrica]
        prediccion_ajustada = prediccion_base + boost
        
        ajuste_info['ajustes_aplicados'].append({
            'tipo': 'Beneficiario de ausencias',
            'jugadores_out': [j['nombre'] for j in impacto['jugadores_out']],
            'boost': boost,
            'razon': f"Se espera +{boost:.1f} {metrica} por mayor uso"
        })
        ajuste_info['confianza_ajuste'] = 'alta'
    
    # AJUSTE 2: Detectar si el jugador viene de lesión (regreso reciente)
    else:
        df_jug = df_equipo[df_equipo['Jugador'] == jugador_nombre].sort_values('Fecha')
        
        if len(df_jug) >= 2:
            # Revisar si hay un gap grande entre los últimos partidos (indicador de lesión)
            ultima_fecha = df_jug['Fecha'].iloc[-1]
            penultima_fecha = df_jug['Fecha'].iloc[-2]
            dias_gap = (ultima_fecha - penultima_fecha).days
            
            if dias_gap > 10:  # Estuvo fuera más de 10 días
                # Reducir predicción en primer partido de regreso
                penalizacion = prediccion_base * 0.15  # 15% de reducción
                prediccion_ajustada = prediccion_base - penalizacion
                
                ajuste_info['ajustes_aplicados'].append({
                    'tipo': 'Regreso de lesión',
                    'dias_fuera': dias_gap,
                    'penalizacion': penalizacion,
                    'razon': f"Primer partido tras {dias_gap} días fuera. -15% por precaución"
                })
                ajuste_info['confianza_ajuste'] = 'media'
            else:
                prediccion_ajustada = prediccion_base
        else:
            prediccion_ajustada = prediccion_base
    
    ajuste_info['prediccion_final'] = prediccion_ajustada
    
    return prediccion_ajustada, ajuste_info


def mostrar_analisis_lesiones(df_equipo, lesionados_df, equipo_nombre):
    """
    Muestra un análisis visual del impacto de lesiones
    """
    jugadores_out = lesionados_df['Jugador'].tolist()
    jugadores_out_equipo = [j for j in jugadores_out if j in JUGADORES_DB.get(equipo_nombre, {}).keys()]
    
    if not jugadores_out_equipo:
        st.info("✅ No hay jugadores clave ausentes")
        return
    
    impacto = calcular_impacto_ausencias(df_equipo, jugadores_out_equipo)
    
    st.markdown("#### 🚑 IMPACTO DE AUSENCIAS")
    
    # Mostrar jugadores ausentes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Jugadores Ausentes:**")
        for jug_info in impacto['jugadores_out']:
            stats = jug_info['stats']
            st.markdown(f"""
            <div class="alert-danger">
                <b>{jug_info['nombre']}</b><br>
                📊 Producción perdida:<br>
                • Puntos: {stats['Puntos']:.1f}<br>
                • Rebotes: {stats['Rebotes']:.1f}<br>
                • Asistencias: {stats['Asistencias']:.1f}<br>
                • Minutos: {stats['Minutos']:.1f}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if impacto['beneficiarios_principales']:
            st.markdown("**Beneficiarios Esperados:**")
            for beneficiario in impacto['beneficiarios_principales']:
                redistrib = impacto['redistribucion_estimada'][beneficiario]
                st.markdown(f"""
                <div class="alert-success">
                    <b>{beneficiario}</b><br>
                    📈 Aumento esperado:<br>
                    • Puntos: +{redistrib['Puntos']:.1f}<br>
                    • Rebotes: +{redistrib['Rebotes']:.1f}<br>
                    • Asistencias: +{redistrib['Asistencias']:.1f}<br>
                    • Minutos: +{redistrib['Minutos_extra']:.1f}
                </div>
                """, unsafe_allow_html=True)


def detectar_regresos_lesion(df_equipo, dias_umbral=10):
    """
    Detecta jugadores que regresan de lesión (gaps en fechas)
    
    Returns:
        Lista de jugadores con info de su regreso
    """
    regresos = []
    
    for jugador in df_equipo['Jugador'].unique():
        df_jug = df_equipo[df_equipo['Jugador'] == jugador].sort_values('Fecha')
        
        if len(df_jug) >= 2:
            fechas = df_jug['Fecha'].tolist()
            
            # Buscar gaps grandes
            for i in range(1, len(fechas)):
                dias_gap = (fechas[i] - fechas[i-1]).days
                
                if dias_gap > dias_umbral:
                    # Este jugador estuvo fuera
                    ultimo_partido = fechas[i]
                    dias_desde_regreso = (obtener_fecha_panama().replace(tzinfo=None) - ultimo_partido).days
                    
                    if dias_desde_regreso <= 7:  # Regresó hace menos de una semana
                        stats_pre_lesion = df_jug[df_jug['Fecha'] < fechas[i-1]].tail(3)
                        stats_post_lesion = df_jug[df_jug['Fecha'] >= fechas[i]].tail(3)
                        
                        if not stats_pre_lesion.empty and not stats_post_lesion.empty:
                            cambio_puntos = stats_post_lesion['Puntos'].mean() - stats_pre_lesion['Puntos'].mean()
                            cambio_minutos = stats_post_lesion['Minutos'].mean() - stats_pre_lesion['Minutos'].mean()
                            
                            regresos.append({
                                'jugador': jugador,
                                'dias_fuera': dias_gap,
                                'fecha_regreso': ultimo_partido,
                                'dias_desde_regreso': dias_desde_regreso,
                                'cambio_puntos': cambio_puntos,
                                'cambio_minutos': cambio_minutos,
                                'partidos_desde_regreso': len(stats_post_lesion)
                            })
    
    return regresos


def mostrar_regresos_lesion(df_equipo):
    """
    Muestra análisis de jugadores que regresan de lesión
    """
    regresos = detectar_regresos_lesion(df_equipo)
    
    if not regresos:
        return
    
    st.markdown("#### 🩹 JUGADORES EN RECUPERACIÓN")
    
    for reg in regresos:
        color = "warning" if reg['cambio_puntos'] < -3 else "info"
        icono = "⚠️" if reg['cambio_puntos'] < -3 else "ℹ️"
        
        st.markdown(f"""
        <div class="alert-{color}">
            {icono} <b>{reg['jugador']}</b> - Regresó hace {reg['dias_desde_regreso']} días<br>
            • Estuvo fuera: {reg['dias_fuera']} días<br>
            • Partidos jugados desde regreso: {reg['partidos_desde_regreso']}<br>
            • Cambio en puntos: {reg['cambio_puntos']:+.1f}<br>
            • Cambio en minutos: {reg['cambio_minutos']:+.1f}<br>
            {'<b>⚠️ Aún en proceso de readaptación</b>' if reg['cambio_puntos'] < -3 else '<b>✅ Recuperación en progreso</b>'}
        </div>
        """, unsafe_allow_html=True)

def mostrar_importancia_features(metricas_modelo):
    """Visualiza features importantes"""
    if metricas_modelo and 'importance' in metricas_modelo:
        st.markdown("#### 🎯 Features Más Importantes")
        
        importances = metricas_modelo['importance']
        
        df_imp = pd.DataFrame({
            'Feature': list(importances.keys()),
            'Importancia': list(importances.values())
        }).sort_values('Importancia', ascending=False).head(10)
        
        fig = go.Figure(go.Bar(
            x=df_imp['Importancia'],
            y=df_imp['Feature'],
            orientation='h',
            marker=dict(color=df_imp['Importancia'], colorscale='Viridis'),
            text=df_imp['Importancia'].round(3),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Top 10 Features",
            template='plotly_dark',
            height=400,
            paper_bgcolor='rgba(11, 32, 39, 0.5)',
            plot_bgcolor='rgba(11, 32, 39, 0.8)',
            margin=dict(l=150, r=40, t=60, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 🏀 CONTROL PANEL")
    
    # ✅ MOSTRAR METADATA
    metadata = dm.cargar_metadata()
    if metadata:
        ultima = metadata.get('ultima_actualizacion', '')
        if ultima:
            try:
                dt = datetime.fromisoformat(ultima)
                st.info(f"📅 Actualizado: {dt.strftime('%d/%m %H:%M')}")
            except:
                st.info(f"📅 Actualizado recientemente")
    
    equipo_sel = st.selectbox(
        "Selecciona Equipo",
        sorted(list(JUGADORES_DB.keys())),
        key="equipo_sidebar"
    )
    
    st.divider()
    st.markdown("### 🏥 ESTADO DEL EQUIPO")
    
    # ✅ CARGAR LESIONADOS DESDE BD
    lesionados_df = dm.obtener_lesionados_equipo(equipo_sel)
    
    if not lesionados_df.empty:
        st.warning(f"⚠️ {len(lesionados_df)} jugador(es) no disponible(s)")
        for _, row in lesionados_df.iterrows():
            nombre_jugador = row['Jugador']
            razon_original = row.get('Razon', 'No especificado')
            st.markdown(f"🏥 **{nombre_jugador}** - {razon_original}")
    else:
        st.success("✅ Equipo completo disponible")
        
    st.session_state.lesionados_equipo = lesionados_df
    
    # ... (resto del sidebar igual: contexto próximo partido, etc.) ...
    
    st.divider()
    
    # ✅ BOTÓN MODIFICADO
    btn_cargar = st.button("🚀 CARGAR DATOS", use_container_width=True, type="primary")
    
    # ✅ AGREGAR INFO DE STORAGE (DEBUG)
    if st.checkbox("📊 Ver Info de Datos", key="debug_storage"):
        storage = dm.estadisticas_almacenamiento()
        st.json(storage)

# ============================================================================
# LÓGICA DE CARGA 
# ============================================================================

if btn_cargar:
    with st.spinner("⚡ Cargando datos desde base de datos..."):
        # ✅ LECTURA INSTANTÁNEA DE PARQUET
        df_equipo = dm.obtener_stats_equipo(equipo_sel)
        
        if df_equipo.empty:
            st.error("❌ No hay datos disponibles. El scraper automático se ejecuta 3 veces al día.")
            st.info("💡 Tip: Espera a la próxima actualización o ejecuta el scraper manualmente en GitHub Actions.")
        else:
            # Calcular días de descanso
            df_equipo = df_equipo.sort_values(['Jugador', 'Fecha'])
            df_equipo['Dias_Descanso'] = (
                df_equipo.groupby('Jugador')['Fecha']
                .diff()
                .dt.days
            )
            
            # Si hay rival, cargar también
            if 'rival_nombre' in st.session_state and st.session_state.rival_nombre:
                df_rival = dm.obtener_stats_equipo(st.session_state.rival_nombre)
                
                if not df_rival.empty:
                    df_rival = df_rival.sort_values(['Jugador', 'Fecha'])
                    df_rival['Dias_Descanso'] = (
                        df_rival.groupby('Jugador')['Fecha']
                        .diff()
                        .dt.days
                    )
                    df_final = pd.concat([df_equipo, df_rival], ignore_index=True)
                    st.session_state.equipos_cargados = [equipo_sel, st.session_state.rival_nombre]
                else:
                    df_final = df_equipo
                    st.session_state.equipos_cargados = [equipo_sel]
            else:
                df_final = df_equipo
                st.session_state.equipos_cargados = [equipo_sel]
            
            st.session_state.df_equipo = df_final
            
            st.success(f"✅ {len(df_final)} registros cargados instantáneamente")
            st.balloons()
            st.rerun()

# ============================================================================
# VISUALIZACIÓN
# ============================================================================

if "df_equipo" in st.session_state:
    df = st.session_state.df_equipo
    num_viz = st.session_state.get('num_partidos_viz', 7)
    equipos_cargados = st.session_state.get('equipos_cargados', [equipo_sel])
    
    if len(equipos_cargados) > 1:
        titulo_header = f"{equipos_cargados[0]} vs {equipos_cargados[1]}"
    else:
        titulo_header = equipos_cargados[0]
    
    st.markdown(f"""
    <div style='text-align:center; padding:25px; background: linear-gradient(135deg, #00D9FF, #0A4D68); 
                border-radius:15px; margin-bottom:30px;'>
        <h1 style='margin:0; font-size:36px; color:#05161A;'>⚡ {titulo_header}</h1>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 REPARTO DE EQUIPO", "🔍 ANÁLISIS INDIVIDUAL", "🏆 TOP 3"])

    with tab1:
        st.markdown("### 📈 Distribución de Carga por Partido")
        
        fechas_unicas = sorted(df['Fecha'].unique(), reverse=True)[:num_viz]
        df_viz = df[df['Fecha'].isin(fechas_unicas)]
        
        equipo_visualizar = st.selectbox("Selecciona equipo", equipos_cargados, key="equipo_tab1")
        df_equipo_viz = df_viz[df_viz["Equipo"] == equipo_visualizar]
        
        jugadores_unicos = df_equipo_viz['Jugador'].unique()
        colores = ['#00D9FF', '#00FFA3', '#6B9EB0', '#FFD93D', '#FF6B6B', '#9B59B6']
        colores_jugadores = {jug: colores[i % len(colores)] for i, jug in enumerate(jugadores_unicos)}
        
        st.plotly_chart(crear_grafica_apilada_mejorada(df_equipo_viz, "Puntos", f"📊 Puntos: {equipo_visualizar}", colores_jugadores), use_container_width=True)
        st.plotly_chart(crear_grafica_apilada_mejorada(df_equipo_viz, "Rebotes", f"🏀 Rebotes: {equipo_visualizar}", colores_jugadores), use_container_width=True)
        st.plotly_chart(crear_grafica_apilada_mejorada(df_equipo_viz, "Asistencias", f"🎯 Asistencias: {equipo_visualizar}", colores_jugadores), use_container_width=True)

    with tab2:
        st.markdown("### 🔍 Análisis Individual")
        
        col_sel1, col_sel2, col_sel3 = st.columns([2, 2, 1])
        with col_sel1:
            jugador_analisis = st.selectbox("Jugador", sorted(df["Jugador"].unique()), key="jug_t2")
        with col_sel2:
            metrica_focus = st.selectbox("Métrica", ["Puntos", "Rebotes", "Asistencias"], key="met_t2")
        with col_sel3:
            linea_over = st.number_input("Línea O/U", value=15.5, step=0.5, key="linea_t2")

        df_jug_completo = df[df["Jugador"] == jugador_analisis].copy().sort_values("Fecha")
        df_jug = df_jug_completo.tail(num_viz)

        if not df_jug.empty:
            info_jug = JUGADORES_DB.get(df_jug.iloc[0]["Equipo"], {}).get(jugador_analisis, {})
            
            col_info, col_stats = st.columns([1, 2])
            
            with col_info:
                st.markdown(f"""
                <div class="stat-card">
                    <h2 style='margin:0; color:#00FFA3; text-align:center;'>{jugador_analisis}</h2>
                    <hr style='border-color:#00D9FF; margin:10px 0;'>
                    <p style='margin:5px 0; font-size:18px; color:#E0F4FF;'>📍 {info_jug.get('pos', 'N/A')}</p>
                    <p style='margin:5px 0; font-size:18px; color:#E0F4FF;'>📏 {info_jug.get('alt', 0)} cm</p>
                    <p style='margin:5px 0; font-size:18px; color:#E0F4FF;'>🏀 {df_jug.iloc[0]["Equipo"]}</p>
                </div>""", unsafe_allow_html=True)
                
                consistencia, cv_val = calcular_indice_consistencia(df_jug[metrica_focus])
                st.metric("Consistencia", consistencia, f"CV: {cv_val:.3f}")

            with col_stats:
                st.markdown(f"#### 📊 Promedios (Últimos {len(df_jug)} PJ)")
                c_m = st.columns(5)
                m_nombres = ["Puntos", "Rebotes", "Asistencias", "Minutos", "Eficiencia"]
                m_iconos = ["🎯", "🏀", "🤝", "⏱️", "⚡"]
                m_colores = ["#00D9FF", "#00FFA3", "#6B9EB0", "#FFD93D", "#9B59B6"]
                
                for col, m_n, icono, color in zip(c_m, m_nombres, m_iconos, m_colores):
                    with col:
                        val = df_jug[m_n].mean()
                        st.markdown(f"""
                        <div style='background:{color}20; padding:10px; border-radius:8px; text-align:center; border-left:4px solid {color};'>
                            <p style='margin:0; font-size:18px;'>{icono}</p>
                            <h3 style='margin:5px 0; color:{color}; font-size:18px;'>{val:.1f}</h3>
                            <p style='margin:0; font-size:10px; color:#E0F4FF;'>{m_n}</p>
                        </div>""", unsafe_allow_html=True)

            st.divider()
            st.markdown("#### 🤖 PREDICCIÓN IA")
            
            col_ml1, col_ml2 = st.columns(2)
            
            with col_ml1:
                with st.spinner("Entrenando XGBoost..."):
                    modelo, scaler, metricas = entrenar_modelo_v2(df, jugador_analisis, target_col=metrica_focus)
                
                if modelo is not None:
                    st.success("✅ Modelo Optimizado")
                    confianza = max(0.0, min(1.0, 1 - (metricas['mae']/df_jug[metrica_focus].mean())))
                    st.progress(confianza, text=f"Confianza: {confianza*100:.1f}%")
                    
                    with st.expander("📊 Ver Importancia Features"):
                        mostrar_importancia_features(metricas)
                
            with col_ml2:
                if modelo is not None:
                    es_loc_bool = st.session_state.get('localia', 'Local') == 'Local'
                    d_descanso = df_jug_completo['Dias_Descanso'].iloc[-1] if not pd.isna(df_jug_completo['Dias_Descanso'].iloc[-1]) else 2
                    pred_base, _ = predecir_proximo_partido_v2(modelo, scaler, df_jug_completo, target_col=metrica_focus, es_local=es_loc_bool, dias_descanso=d_descanso)
        
                # AJUSTE POR LESIONES Y CONTEXTO
                lesionados_equipo = st.session_state.get('lesionados_equipo', pd.DataFrame())
                pred_ajustada, ajuste_info = ajustar_prediccion_por_contexto(
                    pred_base, 
                    jugador_analisis, 
                    df, 
                    lesionados_equipo, 
                    metrica_focus
                )
        
                # Determinar si hubo ajuste significativo
                diferencia = abs(pred_ajustada - pred_base)
                hay_ajuste = diferencia > 0.5
        
                st.markdown(f"""
                <div style='text-align:center; padding:20px; background:linear-gradient(135deg, #00D9FF, #00FFA3); border-radius:12px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
                      <p style='margin:0; font-size:14px; color:#05161A; font-weight:bold;'>PROYECCIÓN IA {'⚠️ AJUSTADA' if hay_ajuste else ''}</p>
                      <h1 style='margin:5px 0; color:#05161A; font-size:48px;'>{pred_ajustada:.1f}</h1>
                      <p style='margin:0; font-size:12px; color:#05161A;'>Línea: {linea_over}</p>
                      {f'<p style="margin:5px 0; font-size:11px; color:#05161A;">Base: {pred_base:.1f} → Ajuste: {pred_ajustada - pred_base:+.1f}</p>' if hay_ajuste else ''}
                </div>""", unsafe_allow_html=True)
        
                # Mostrar explicación del ajuste
                if hay_ajuste and ajuste_info['ajustes_aplicados']:
                     with st.expander("🔍 Ver Detalles del Ajuste"):
                        for ajuste in ajuste_info['ajustes_aplicados']:
                            st.markdown(f"""
                            **Tipo:** {ajuste['tipo']}  
                            **Razón:** {ajuste['razon']}
                            """)
                    
                            if 'jugadores_out' in ajuste:
                                st.markdown(f"**Jugadores ausentes:** {', '.join(ajuste['jugadores_out'])}")


            st.divider()
            st.markdown(f"#### 📈 Historial de {metrica_focus}")
            st.plotly_chart(crear_grafica_individual_mejorada(df_jug, metrica_focus, linea_over, jugador_analisis), use_container_width=True)

         # ANÁLISIS DE CONTEXTO DEL EQUIPO
            st.divider()
            st.markdown("#### 🏥 ANÁLISIS DE CONTEXTO DEL EQUIPO")
            
            col_ctx1, col_ctx2 = st.columns(2)
            
            with col_ctx1:
                # Mostrar impacto de lesiones
                equipo_jugador = df_jug.iloc[0]["Equipo"]
                lesionados_equipo = st.session_state.get('lesionados_equipo', pd.DataFrame())
                
                if not lesionados_equipo.empty:
                    mostrar_analisis_lesiones(df, lesionados_equipo, equipo_jugador)
                else:
                    st.info("✅ No hay lesiones reportadas")
            
            with col_ctx2:
                # Mostrar jugadores en recuperación
                mostrar_regresos_lesion(df)


    with tab3:
        st.markdown("### 🏆 Top 3 Comparativas")
        
        df_top = df.groupby('Jugador').head(num_viz)
        promedios = df_top.groupby('Jugador').agg({
            'Puntos': 'mean',
            'Rebotes': 'mean',
            'Asistencias': 'mean',
            'Equipo': 'first'
        }).reset_index()
        
        categorias = [
            {"t": "🎯 ANOTADORES", "m": "Puntos", "c": "#00D9FF"},
            {"t": "🏀 REBOTEADORES", "m": "Rebotes", "c": "#00FFA3"},
            {"t": "🤝 ASISTIDORES", "m": "Asistencias", "c": "#9B59B6"}
        ]
        
        cols = st.columns(3)
        
        for i, cat in enumerate(categorias):
            with cols[i]:
                st.markdown(f"#### {cat['t']}")
                top3 = promedios.nlargest(3, cat['m'])
                medallas = ["🥇", "🥈", "🥉"]
                
                for j, (_, row) in enumerate(top3.iterrows()):
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {cat['c']}25, {cat['c']}05); 
                                padding:15px; border-radius:12px; margin-bottom:12px; border-left:5px solid {cat['c']};'>
                        <div style='display:flex; justify-content:space-between;'>
                            <span style='font-size:16px; font-weight:bold; color:#E0F4FF;'>{row['Jugador']}</span>
                            <span style='font-size:20px;'>{medallas[j]}</span>
                        </div>
                        <span style='color:{cat['c']}; font-size:24px; font-weight:bold;'>{row[cat['m']]:.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)


    # streamlit run app_nba.py
    # .\Hoops_Analytics\Scripts\Activate.ps1


    