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

from logic_nba import scrapear_jugador, obtener_jugadores_lesionados, obtener_proximo_partido
from config_nba import JUGADORES_DB, TEAM_IDS

# ── ML: todo viene de ml_predictor ──────────────────────────────────────────
from ml_predictor import predecir

from curl_cffi import requests
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
    st.write("---")
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
        .stat-item {{ display: flex; flex-direction: column; }}
        .stat-label {{ font-size: 12px; color: #B8D4E0; margin-bottom: 5px; }}
        .stat-value {{ font-size: 20px; font-weight: 700; color: #00D9FF; }}
        @keyframes slideIn {{
            from {{ transform: translateX(-20px); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        .animated-entry {{ animation: slideIn 0.3s ease-out; }}
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
# FUNCIONES DE GRÁFICAS
# ============================================================================

def crear_grafica_apilada_mejorada(df_equipo, metrica, titulo, colores_jugadores):
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
    df_plot = df_jug.copy().sort_values('Fecha')
    df_plot['Fecha_Str'] = df_plot['Fecha'].dt.strftime('%d/%m')
    
    fig = go.Figure()
    fig.add_hline(
        y=linea, line_dash="dash", line_color="#FFD93D", line_width=2,
        annotation_text=f"Línea {linea}", annotation_position="top right",
        annotation_font_color="#FFD93D"
    )
    
    colores_barras = ['#00FFA3' if v >= linea else '#FF6B6B' for v in df_plot[metrica]]
    
    fig.add_trace(go.Bar(
        x=df_plot['Fecha_Str'], y=df_plot[metrica],
        marker=dict(color=colores_barras, line=dict(color='rgba(0, 0, 0, 0.4)', width=1.5)),
        text=df_plot[metrica].round(1), textposition="outside",
        textfont=dict(size=11, color='#E0F4FF'),
        hovertemplate="<b>%{x}</b><br>Valor: %{y:.1f}<extra></extra>",
        name=metrica
    ))
    
    if len(df_plot) >= 3:
        fig.add_trace(go.Scatter(
            x=df_plot['Fecha_Str'],
            y=df_plot[metrica].rolling(window=3, min_periods=1).mean(),
            mode='lines', line=dict(color='#00D9FF', width=2, dash='dot'),
            name='Tendencia', hovertemplate='Promedio: %{y:.1f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=f'📊 Evolución de {metrica} - {nombre_jugador}', font=dict(size=18, color='#00FFA3'), x=0.02),
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

# ============================================================================
# SISTEMA DE ANÁLISIS DE IMPACTO POR LESIONES
# ============================================================================

def calcular_impacto_ausencias(df_equipo, jugadores_ausentes, metrica='Puntos'):
    if not jugadores_ausentes:
        return {'impacto_total': 0, 'redistribucion': {}}
    
    impacto = {
        'jugadores_out': [],
        'produccion_perdida': {},
        'redistribucion_estimada': {},
        'beneficiarios_principales': []
    }
    
    for jugador_out in jugadores_ausentes:
        df_jug = df_equipo[df_equipo['Jugador'] == jugador_out]
        if not df_jug.empty:
            stats_perdidas = {
                'Puntos': df_jug['Puntos'].mean(),
                'Rebotes': df_jug['Rebotes'].mean(),
                'Asistencias': df_jug['Asistencias'].mean(),
                'Minutos': df_jug['Minutos'].mean()
            }
            impacto['jugadores_out'].append({'nombre': jugador_out, 'stats': stats_perdidas})
            for stat, val in stats_perdidas.items():
                impacto['produccion_perdida'][stat] = impacto['produccion_perdida'].get(stat, 0) + val
    
    if impacto['jugadores_out']:
        minutos_disponibles = impacto['produccion_perdida'].get('Minutos', 0)
        jugadores_activos = [j for j in df_equipo['Jugador'].unique() if j not in jugadores_ausentes]
        df_activos = df_equipo[df_equipo['Jugador'].isin(jugadores_activos)]
        top_beneficiarios = df_activos.groupby('Jugador')['Minutos'].mean().sort_values(ascending=False).head(3)
        total_minutos_top3 = top_beneficiarios.sum()
        
        for jugador, minutos_avg in top_beneficiarios.items():
            proporcion = minutos_avg / total_minutos_top3 if total_minutos_top3 > 0 else 0
            impacto['redistribucion_estimada'][jugador] = {
                'Puntos': impacto['produccion_perdida'].get('Puntos', 0) * proporcion * 0.7,
                'Rebotes': impacto['produccion_perdida'].get('Rebotes', 0) * proporcion * 0.7,
                'Asistencias': impacto['produccion_perdida'].get('Asistencias', 0) * proporcion * 0.7,
                'Minutos_extra': minutos_disponibles * proporcion
            }
            impacto['beneficiarios_principales'].append(jugador)
    
    return impacto


def ajustar_prediccion_por_contexto(prediccion_base, jugador_nombre, df_equipo,
                                     lesionados_df, metrica='Puntos'):
    ajuste_info = {
        'prediccion_original': prediccion_base,
        'ajustes_aplicados': [],
        'prediccion_final': prediccion_base,
        'confianza_ajuste': 'media'
    }
    
    if lesionados_df.empty:
        return prediccion_base, ajuste_info
    
    equipo_jugador = df_equipo[df_equipo['Jugador'] == jugador_nombre]['Equipo'].iloc[0]
    jugadores_out = lesionados_df['Jugador'].tolist()
    jugadores_out_equipo = [j for j in jugadores_out if j in JUGADORES_DB.get(equipo_jugador, {}).keys()]
    impacto = calcular_impacto_ausencias(df_equipo, jugadores_out_equipo, metrica)
    
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
    else:
        df_jug = df_equipo[df_equipo['Jugador'] == jugador_nombre].sort_values('Fecha')
        if len(df_jug) >= 2:
            ultima_fecha = df_jug['Fecha'].iloc[-1]
            penultima_fecha = df_jug['Fecha'].iloc[-2]
            dias_gap = (ultima_fecha - penultima_fecha).days
            if dias_gap > 10:
                penalizacion = prediccion_base * 0.15
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
    jugadores_out = lesionados_df['Jugador'].tolist()
    jugadores_out_equipo = [j for j in jugadores_out if j in JUGADORES_DB.get(equipo_nombre, {}).keys()]
    
    if not jugadores_out_equipo:
        st.info("✅ No hay jugadores clave ausentes")
        return
    
    impacto = calcular_impacto_ausencias(df_equipo, jugadores_out_equipo)
    st.markdown("#### 🚑 IMPACTO DE AUSENCIAS")
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
    regresos = []
    for jugador in df_equipo['Jugador'].unique():
        df_jug = df_equipo[df_equipo['Jugador'] == jugador].sort_values('Fecha')
        if len(df_jug) >= 2:
            fechas = df_jug['Fecha'].tolist()
            for i in range(1, len(fechas)):
                dias_gap = (fechas[i] - fechas[i-1]).days
                if dias_gap > dias_umbral:
                    ultimo_partido = fechas[i]
                    dias_desde_regreso = (obtener_fecha_panama().replace(tzinfo=None) - ultimo_partido).days
                    if dias_desde_regreso <= 7:
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
    if metricas_modelo and 'importance' in metricas_modelo:
        st.markdown("#### 🎯 Features Más Importantes")
        importances = metricas_modelo['importance']
        df_imp = pd.DataFrame({
            'Feature': list(importances.keys()),
            'Importancia': list(importances.values())
        }).sort_values('Importancia', ascending=False).head(10)
        
        fig = go.Figure(go.Bar(
            x=df_imp['Importancia'], y=df_imp['Feature'],
            orientation='h',
            marker=dict(color=df_imp['Importancia'], colorscale='Viridis'),
            text=df_imp['Importancia'].round(3), textposition='auto'
        ))
        fig.update_layout(
            title="Top 10 Features", template='plotly_dark', height=400,
            paper_bgcolor='rgba(11, 32, 39, 0.5)', plot_bgcolor='rgba(11, 32, 39, 0.8)',
            margin=dict(l=150, r=40, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SIDEBAR
# ============================================================================
import socket
socket.setdefaulttimeout(5)

@st.cache_data(ttl=1800, show_spinner=False)
def obtener_datos_partido_cached(nombre_equipo):
    try:
        return obtener_proximo_partido(nombre_equipo)
    except Exception as e:
        return {
            "hay_juego": False, "rival": None, "rival_display": None,
            "localia": None, "fecha": None, "_error": str(e)
        }


with st.sidebar:
    st.markdown("### 🏀 CONTROL PANEL")
    
    try:
        metadata = dm.cargar_metadata()
        if metadata:
            ultima = metadata.get('ultima_actualizacion', '')
            if ultima:
                dt = datetime.fromisoformat(ultima)
                hace_horas = (datetime.now() - dt).total_seconds() / 3600
                color_estado = "success" if hace_horas < 2 else "info" if hace_horas < 6 else "warning"
                icono_estado = "✅" if hace_horas < 2 else "ℹ️" if hace_horas < 6 else "⏰"
                st.markdown(f"""
                <div class="alert-{color_estado}">
                    {icono_estado} <b>Última actualización:</b><br>
                    <small>{dt.strftime('%d/%m %H:%M')} (hace {hace_horas:.1f}h)</small>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.caption(f"⚠️ Metadata: {e}")

    equipo_sel = st.selectbox(
        "Selecciona Equipo",
        sorted(list(JUGADORES_DB.keys())),
        key="equipo_sidebar"
    )

    st.divider()

    st.markdown("### 🏥 ESTADO DEL EQUIPO")
    try:
        lesionados_df = dm.obtener_lesionados_equipo(equipo_sel)
        if not lesionados_df.empty:
            st.warning(f"⚠️ {len(lesionados_df)} no disponible(s)")
            with st.expander("Ver detalles"):
                for _, row in lesionados_df.iterrows():
                    st.markdown(f"🏥 **{row['Jugador']}** - {row.get('Razon', 'N/A')}")
        else:
            st.success("✅ Equipo completo")
        st.session_state.lesionados_equipo = lesionados_df
    except Exception as e:
        st.caption(f"⚠️ Error lesionados: {e}")
        st.session_state.lesionados_equipo = pd.DataFrame()

    st.divider()

    st.markdown("### 🆚 PRÓXIMO PARTIDO")
    contexto = {"hay_juego": False, "rival": None, "localia": None, "fecha": None}
    try:
        with st.spinner("Buscando partido..."):
            contexto = obtener_datos_partido_cached(equipo_sel)
    except Exception as e:
        st.caption(f"⚠️ Error: {type(e).__name__}")

    if contexto.get("_error"):
        st.caption(f"⚠️ {contexto['_error'][:60]}")

    if contexto.get("hay_juego") and contexto.get("rival"):
        nombre_mostrar = contexto.get('rival_display', contexto['rival'])
        st.markdown(f'''
        <div class="alert-info">
            <h4 style='margin:0 0 10px 0; color:#00D9FF;'>🏟️ Próximo Juego</h4>
            <p style='font-size:16px; margin:5px 0; color:#E0F4FF;'><b>{nombre_mostrar}</b></p>
            <p style='margin:5px 0; color:#E0F4FF;'>📍 {contexto["localia"]}</p>
            <p style='margin:0; color:#00FFA3;'>📅 {contexto.get("fecha", "Próximamente")}</p>
        </div>
        ''', unsafe_allow_html=True)

        st.session_state.rival_nombre = contexto['rival']
        st.session_state.rival_display = nombre_mostrar
        st.session_state.localia = contexto['localia']
        st.session_state.event_id = contexto.get('event_id')

        if st.checkbox("📊 Comparar con rival", key="cargar_rival"):
            st.session_state.incluir_rival = True
        else:
            st.session_state.incluir_rival = False
    else:
        st.warning("📅 Sin partidos próximos detectados")
        st.session_state.rival_nombre = None
        st.session_state.incluir_rival = False

    st.markdown("#### 📊 Partidos a Analizar")
    num_partidos_visualizar = st.slider("Visualizar últimos", min_value=3, max_value=10, value=7)
    st.session_state.num_partidos_viz = num_partidos_visualizar

    st.divider()
    btn_cargar = st.button("🚀 CARGAR DATOS", use_container_width=True, type="primary")


# ============================================================================
# LÓGICA DE CARGA
# ============================================================================

if btn_cargar:
    with st.spinner("⚡ Cargando datos desde base de datos..."):
        df_equipo = dm.obtener_stats_equipo(equipo_sel)
        
        if df_equipo.empty:
            st.error(f"❌ No hay datos del equipo **{equipo_sel}**")
            st.info("💡 **Ejecuta el scraper inicial:**\n```bash\npython scraper_automatico.py\n```")
            st.warning("⏰ O espera a que GitHub Actions ejecute automáticamente (3 veces al día)")
        else:
            df_equipo = df_equipo.sort_values(['Jugador', 'Fecha'])
            df_equipo['Dias_Descanso'] = (
                df_equipo.groupby('Jugador')['Fecha'].diff().dt.days
            )
            
            equipos_cargados = [equipo_sel]
            
            if st.session_state.get('incluir_rival', False) and st.session_state.get('rival_nombre'):
                rival = st.session_state.rival_nombre
                df_rival = dm.obtener_stats_equipo(rival)
                
                if not df_rival.empty:
                    df_rival = df_rival.sort_values(['Jugador', 'Fecha'])
                    df_rival['Dias_Descanso'] = (
                        df_rival.groupby('Jugador')['Fecha'].diff().dt.days
                    )
                    df_final = pd.concat([df_equipo, df_rival], ignore_index=True)
                    equipos_cargados.append(rival)
                    st.success(f"✅ Cargados: **{equipo_sel}** + **{rival}**")
                else:
                    df_final = df_equipo
                    st.warning(f"⚠️ No hay datos del rival (**{rival}**). Solo se cargó {equipo_sel}")
            else:
                df_final = df_equipo
            
            st.session_state.df_equipo = df_final
            st.session_state.equipos_cargados = equipos_cargados
            
            st.success(f"✅ {len(df_final)} registros | {len(df_final['Jugador'].unique())} jugadores")
            st.balloons()
            st.rerun()

# ============================================================================
# VISUALIZACIÓN
# ============================================================================

if "df_equipo" in st.session_state:
    df = st.session_state.df_equipo
    num_viz = st.session_state.get('num_partidos_viz', 7)
    equipos_cargados = st.session_state.get('equipos_cargados', [equipo_sel])
    
    titulo_header = f"{equipos_cargados[0]} vs {equipos_cargados[1]}" if len(equipos_cargados) > 1 else equipos_cargados[0]
    
    st.markdown(f"""
<style>
    @keyframes fadeInScale {{
        from {{ opacity: 0; transform: scale(0.95); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}
    .main-header {{
        text-align: center; padding: clamp(15px, 3vw, 30px);
        background: linear-gradient(135deg, #00D9FF, #0A4D68);
        border-radius: 15px; margin-bottom: 20px;
        animation: fadeInScale 0.5s ease-out;
        box-shadow: 0 8px 25px rgba(0, 217, 255, 0.3);
    }}
    .brand-logo {{ font-size: clamp(40px, 8vw, 80px); margin-bottom: 5px; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3)); }}
    .brand-name {{ margin: 0; font-size: clamp(24px, 5vw, 42px); color: #FFD700; font-family: 'Arial Black', sans-serif; text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5); letter-spacing: 2px; }}
    .brand-subtitle {{ margin: 5px 0 0 0; font-size: clamp(14px, 2.5vw, 18px); color: #E0F4FF; font-weight: 600; letter-spacing: 1px; }}
    .team-matchup {{ margin-top: 15px; padding: 12px 20px; background: rgba(5, 22, 26, 0.4); border-radius: 25px; display: inline-block; }}
    .team-name {{ font-size: clamp(18px, 4vw, 28px); color: #00FFA3; font-weight: bold; }}
</style>
<div class="main-header">
    <div class="brand-logo">🏀</div>
    <h1 class="brand-name">CANAL HOOPS ANALYTICS</h1>
    <p class="brand-subtitle">by Rodolfo Cisco</p>
    <div class="team-matchup">
        <span class="team-name">{titulo_header}</span>
    </div>
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
                m_iconos  = ["🎯", "🏀", "🤝", "⏱️", "⚡"]
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

            # ── PREDICCIÓN IA ────────────────────────────────────────────────
            st.divider()
            st.markdown("#### 🤖 PREDICCIÓN IA")
            
            col_ml1, col_ml2 = st.columns(2)
            
            es_loc_bool  = st.session_state.get('localia', 'Local') == 'Local'
            d_descanso   = df_jug_completo['Dias_Descanso'].iloc[-1]
            d_descanso   = d_descanso if pd.notna(d_descanso) else 2

            with st.spinner("Calculando predicción..."):
                resultado_ml = predecir(
                    df, jugador_analisis,
                    target_col=metrica_focus,
                    es_local=es_loc_bool,
                    dias_descanso=d_descanso
                )

            with col_ml1:
                if resultado_ml:
                    nivel_label = "✅ XGBoost (25 features)" if resultado_ml['nivel'] == 'xgboost' else "⚡ Ensemble (fallback)"
                    st.success(nivel_label)
                    confianza_pct = resultado_ml['confianza']
                    st.progress(confianza_pct / 100, text=f"Confianza: {confianza_pct:.1f}%")
                    
                    metricas_mod = resultado_ml.get('metricas_modelo', {})
                    if resultado_ml['nivel'] == 'xgboost' and 'importance' in metricas_mod:
                        with st.expander("📊 Ver Importancia Features"):
                            mostrar_importancia_features(metricas_mod)
                    elif resultado_ml['nivel'] == 'fallback':
                        ajustes = metricas_mod.get('ajustes', {})
                        if ajustes.get('items'):
                            with st.expander("📊 Ajustes aplicados"):
                                for item in ajustes['items']:
                                    st.caption(f"• {item}")

            with col_ml2:
                if resultado_ml:
                    pred_base = resultado_ml['prediccion']
                    lesionados_equipo = st.session_state.get('lesionados_equipo', pd.DataFrame())
                    pred_ajustada, ajuste_info = ajustar_prediccion_por_contexto(
                        pred_base, jugador_analisis, df, lesionados_equipo, metrica_focus
                    )
                    
                    diferencia = abs(pred_ajustada - pred_base)
                    hay_ajuste = diferencia > 0.5

                    st.markdown(f"""
                    <div style='text-align:center; padding:20px; background:linear-gradient(135deg, #00D9FF, #00FFA3); border-radius:12px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
                          <p style='margin:0; font-size:14px; color:#05161A; font-weight:bold;'>PROYECCIÓN IA {'⚠️ AJUSTADA' if hay_ajuste else ''}</p>
                          <h1 style='margin:5px 0; color:#05161A; font-size:48px;'>{pred_ajustada:.1f}</h1>
                          <p style='margin:0; font-size:12px; color:#05161A;'>Línea: {linea_over}</p>
                          {f'<p style="margin:5px 0; font-size:11px; color:#05161A;">Base: {pred_base:.1f} → Ajuste: {pred_ajustada - pred_base:+.1f}</p>' if hay_ajuste else ''}
                    </div>""", unsafe_allow_html=True)
                    
                    if hay_ajuste and ajuste_info['ajustes_aplicados']:
                        with st.expander("🔍 Ver Detalles del Ajuste"):
                            for ajuste in ajuste_info['ajustes_aplicados']:
                                st.markdown(f"**Tipo:** {ajuste['tipo']}  \n**Razón:** {ajuste['razon']}")
                                if 'jugadores_out' in ajuste:
                                    st.markdown(f"**Jugadores ausentes:** {', '.join(ajuste['jugadores_out'])}")
                else:
                    st.warning("⚠️ No hay suficientes datos para predecir (mínimo 3 partidos)")

            # ── GRÁFICA HISTORIAL ────────────────────────────────────────────
            st.divider()
            st.markdown(f"#### 📈 Historial de {metrica_focus}")
            st.plotly_chart(
                crear_grafica_individual_mejorada(df_jug, metrica_focus, linea_over, jugador_analisis),
                use_container_width=True
            )

            # ── ANÁLISIS DEFENSIVO RIVAL ─────────────────────────────────────
            rival_nombre = st.session_state.get('rival_nombre')
            if rival_nombre and rival_nombre in df['Equipo'].unique():
                st.divider()
                st.markdown("#### 🆚 ANÁLISIS DEL RIVAL")
                from logic_nba import calcular_stats_defensivas_rival
                stats_rival = calcular_stats_defensivas_rival(df, rival_nombre, metrica_focus)

                if stats_rival['disponible']:
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.markdown(f'''
                        <div class="stat-card">
                            <p style="color:#B8D4E0; font-size:12px; margin:0;">PRODUCCIÓN TOTAL RIVAL</p>
                            <h2 style="color:#FF6B6B; margin:5px 0;">{stats_rival["produccion_total_rival"]}</h2>
                            <p style="color:#E0F4FF; font-size:12px;">{metrica_focus} combinados</p>
                        </div>''', unsafe_allow_html=True)
                    with col_r2:
                        st.markdown(f'''
                        <div class="stat-card">
                            <p style="color:#B8D4E0; font-size:12px; margin:0;">PROMEDIO POR JUGADOR</p>
                            <h2 style="color:#FFD93D; margin:5px 0;">{stats_rival["promedio_por_jugador"]}</h2>
                            <p style="color:#E0F4FF; font-size:12px;">Jugadores con datos: {stats_rival["total_jugadores_data"]}</p>
                        </div>''', unsafe_allow_html=True)
                    with col_r3:
                        st.markdown(f'''
                        <div class="stat-card">
                            <p style="color:#B8D4E0; font-size:12px; margin:0;">NIVEL RIVAL</p>
                            <h2 style="color:#00FFA3; margin:5px 0; font-size:20px;">{stats_rival["nivel_rival"]}</h2>
                            <p style="color:#E0F4FF; font-size:12px;">Pace ~{stats_rival["pace_semanal"]} partidos/semana</p>
                        </div>''', unsafe_allow_html=True)

                    if stats_rival['top_jugadores']:
                        st.markdown(f"**Top amenazas del rival en {metrica_focus}:**")
                        tops = stats_rival['top_jugadores']
                        cols_top = st.columns(len(tops))
                        medallas = ["🥇", "🥈", "🥉"]
                        for idx, (jug, val) in enumerate(tops.items()):
                            with cols_top[idx]:
                                st.markdown(f'''
                                <div style="background:rgba(255,107,107,0.1); padding:10px; border-radius:8px; border-left:3px solid #FF6B6B; text-align:center;">
                                    <p style="margin:0; font-size:11px; color:#B8D4E0;">{medallas[idx]} {jug}</p>
                                    <h3 style="margin:5px 0; color:#FF6B6B;">{val}</h3>
                                </div>''', unsafe_allow_html=True)
                else:
                    st.info(f"💡 {stats_rival['mensaje']} — Carga el rival con '📊 Comparar con rival' en el sidebar")

            # ── ANÁLISIS DE CONTEXTO ─────────────────────────────────────────
            st.divider()
            st.markdown("#### 🏥 ANÁLISIS DE CONTEXTO DEL EQUIPO")
            col_ctx1, col_ctx2 = st.columns(2)
            with col_ctx1:
                equipo_jugador = df_jug.iloc[0]["Equipo"]
                lesionados_equipo = st.session_state.get('lesionados_equipo', pd.DataFrame())
                if not lesionados_equipo.empty:
                    mostrar_analisis_lesiones(df, lesionados_equipo, equipo_jugador)
                else:
                    st.info("✅ No hay lesiones reportadas")
            with col_ctx2:
                mostrar_regresos_lesion(df)

    with tab3:
        st.markdown("### 🏆 Top 3 Comparativas")
        df_top = df.groupby('Jugador').head(num_viz)
        promedios = df_top.groupby('Jugador').agg({
            'Puntos': 'mean', 'Rebotes': 'mean',
            'Asistencias': 'mean', 'Equipo': 'first'
        }).reset_index()

        rival_nombre = st.session_state.get('rival_nombre')
        hay_rival_en_datos = rival_nombre and rival_nombre in df['Equipo'].unique()

        if hay_rival_en_datos:
            st.markdown(f"#### ⚔️ {equipo_sel} vs {rival_nombre}")
            for metrica_tab3 in ['Puntos', 'Rebotes', 'Asistencias']:
                col_eq, col_vs, col_riv = st.columns([5, 1, 5])
                top_eq  = promedios[promedios['Equipo'] == equipo_sel].nlargest(3, metrica_tab3)
                top_riv = promedios[promedios['Equipo'] == rival_nombre].nlargest(3, metrica_tab3)
                iconos  = {'Puntos': '🎯', 'Rebotes': '🏀', 'Asistencias': '🤝'}
                medallas = ["🥇", "🥈", "🥉"]

                with col_eq:
                    st.markdown(f"**{iconos[metrica_tab3]} {equipo_sel}**")
                    for j, (_, row) in enumerate(top_eq.iterrows()):
                        st.markdown(f'''
                        <div style="background:rgba(0,217,255,0.1); padding:10px; border-radius:8px; margin-bottom:8px; border-left:4px solid #00D9FF;">
                            <span style="color:#E0F4FF;">{medallas[j]} {row["Jugador"]}</span>
                            <span style="float:right; color:#00D9FF; font-weight:bold;">{row[metrica_tab3]:.1f}</span>
                        </div>''', unsafe_allow_html=True)

                with col_vs:
                    st.markdown("<div style='text-align:center; padding-top:30px; color:#FFD93D; font-weight:bold; font-size:20px;'>VS</div>", unsafe_allow_html=True)

                with col_riv:
                    st.markdown(f"**{iconos[metrica_tab3]} {rival_nombre}**")
                    for j, (_, row) in enumerate(top_riv.iterrows()):
                        st.markdown(f'''
                        <div style="background:rgba(255,107,107,0.1); padding:10px; border-radius:8px; margin-bottom:8px; border-left:4px solid #FF6B6B;">
                            <span style="color:#E0F4FF;">{medallas[j]} {row["Jugador"]}</span>
                            <span style="float:right; color:#FF6B6B; font-weight:bold;">{row[metrica_tab3]:.1f}</span>
                        </div>''', unsafe_allow_html=True)
                st.divider()
        else:
            categorias = [
                {"t": "🎯 ANOTADORES",   "m": "Puntos",      "c": "#00D9FF"},
                {"t": "🏀 REBOTEADORES", "m": "Rebotes",     "c": "#00FFA3"},
                {"t": "🤝 ASISTIDORES",  "m": "Asistencias", "c": "#9B59B6"}
            ]
            cols = st.columns(3)
            for i, cat in enumerate(categorias):
                with cols[i]:
                    st.markdown(f"#### {cat['t']}")
                    top3 = promedios.nlargest(3, cat['m'])
                    medallas = ["🥇", "🥈", "🥉"]
                    for j, (_, row) in enumerate(top3.iterrows()):
                        st.markdown(f'''
                        <div style="background: linear-gradient(135deg, {cat["c"]}25, {cat["c"]}05);
                                    padding:15px; border-radius:12px; margin-bottom:12px; border-left:5px solid {cat["c"]};">
                            <div style="display:flex; justify-content:space-between;">
                                <span style="font-size:16px; font-weight:bold; color:#E0F4FF;">{row["Jugador"]}</span>
                                <span style="font-size:20px;">{medallas[j]}</span>
                            </div>
                            <span style="color:{cat["c"]}; font-size:24px; font-weight:bold;">{row[cat["m"]]:.1f}</span>
                        </div>''', unsafe_allow_html=True)

            if rival_nombre:
                st.info(f"💡 Marca '📊 Comparar con rival' en el sidebar y recarga para ver **{rival_nombre}** aquí")
   
    # streamlit run app_nba.py
    # .\Hoops_Analytics\Scripts\Activate.ps1


    