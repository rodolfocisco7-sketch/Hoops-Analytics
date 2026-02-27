# ml_predictor.py
# Sistema híbrido de predicción NBA
# Nivel 1: XGBoost  — modelo principal
# Nivel 2: Ensemble RF + Regresión + WMA — fallback automático

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ============================================================================
# NIVEL 2 — ENSEMBLE HÍBRIDO 
# ============================================================================

def prediccion_wma(df, metrica, ventana=5):
    """Promedio ponderado — más peso a partidos recientes"""
    valores = df.tail(ventana)[metrica].values
    pesos = np.arange(1, len(valores) + 1)
    return np.average(valores, weights=pesos)


def prediccion_regression(df, metrica):
    """Tendencia lineal básica"""
    if len(df) < 7:
        return None
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[metrica].values
    modelo = LinearRegression()
    modelo.fit(X, y)
    return modelo.predict([[len(df)]])[0]


def _features_simplificadas(df, metrica):
    """6 features críticas para RF (fallback)"""
    features, targets = [], []
    for i in range(3, len(df)):
        ventana = df.iloc[max(0, i-5):i]
        features.append([
            ventana[metrica].mean(),
            ventana[metrica].std() if len(ventana) > 1 else 0,
            ventana['Minutos'].mean(),
            1 if df.iloc[i]['Localia'] == 'Local' else 0,
            df.iloc[i].get('Dias_Descanso', 2) if pd.notna(df.iloc[i].get('Dias_Descanso', np.nan)) else 2,
            ventana[metrica].iloc[-1]
        ])
        targets.append(df.iloc[i][metrica])
    return np.array(features), np.array(targets)


def prediccion_rf(df, metrica):
    """Random Forest simplificado"""
    if len(df) < 10:
        return None
    X, y = _features_simplificadas(df, metrica)
    modelo = RandomForestRegressor(
        n_estimators=50, max_depth=3,
        min_samples_split=3, random_state=42
    )
    modelo.fit(X, y)
    ventana = df.tail(5)
    X_next = [[
        ventana[metrica].mean(),
        ventana[metrica].std() if len(ventana) > 1 else 0,
        ventana['Minutos'].mean(),
        1, 2,
        df[metrica].iloc[-1]
    ]]
    return modelo.predict(X_next)[0]


def _ajustes_contextuales(df, metrica):
    """Ajustes por tendencia, racha y volatilidad"""
    ajustes = {'items': [], 'total': 0}

    if len(df) >= 6:
        ultimos_3 = df.tail(3)[metrica].mean()
        previos_3 = df.iloc[-6:-3][metrica].mean()
        diferencia = ultimos_3 - previos_3
        if abs(diferencia) > 2:
            ajustes['items'].append(f"Tendencia: {diferencia:+.1f}")
            ajustes['total'] += diferencia * 0.3

    if len(df) >= 3:
        promedio_total = df[metrica].mean()
        ultimos = df.tail(3)[metrica]
        if all(ultimos > promedio_total):
            ajustes['items'].append("Racha caliente: +1.5")
            ajustes['total'] += 1.5
        elif all(ultimos < promedio_total):
            ajustes['items'].append("Racha fría: -1.5")
            ajustes['total'] -= 1.5

    cv = df[metrica].std() / df[metrica].mean() if df[metrica].mean() > 0 else 0
    if cv > 0.4:
        ajustes['items'].append("Alta volatilidad: -1.0")
        ajustes['total'] -= 1.0

    return ajustes


def calcular_confianza(num_partidos, std):
    """Score de confianza 0–100%"""
    factor_datos = min(num_partidos / 10, 1.0) * 50
    factor_estabilidad = max(0, (1 - std / 10)) * 50
    return min(100, factor_datos + factor_estabilidad)


def predecir_ensemble(df_jugador, metrica='Puntos'):
    """
    Ensemble híbrido RF + Regresión + WMA con ajustes contextuales.
    Usado como fallback cuando XGBoost no tiene suficientes datos.
    Requiere mínimo 3 partidos.
    """
    df = df_jugador.copy().sort_values('Fecha')
    predicciones = []

    pred_rf = prediccion_rf(df, metrica)
    if pred_rf is not None:
        predicciones.append(('RF', pred_rf, 0.4))

    pred_reg = prediccion_regression(df, metrica)
    if pred_reg is not None:
        predicciones.append(('REG', pred_reg, 0.35))

    pred_wma = prediccion_wma(df, metrica)
    predicciones.append(('WMA', pred_wma, 0.25))

    total_peso = sum(p[2] for p in predicciones)
    prediccion_final = sum(p[1] * p[2] for p in predicciones) / total_peso

    ajustes = _ajustes_contextuales(df, metrica)
    prediccion_ajustada = prediccion_final + ajustes['total']

    std_recent = df.tail(5)[metrica].std()
    margen = std_recent * 1.5

    return {
        'prediccion': round(max(0, prediccion_ajustada), 1),
        'intervalo': (max(0, prediccion_ajustada - margen), prediccion_ajustada + margen),
        'modelos_usados': [p[0] for p in predicciones],
        'confianza': calcular_confianza(len(df), std_recent),
        'ajustes': ajustes,
        'nivel': 'fallback'
    }


# ============================================================================
# NIVEL 1 — XGBOOST
# ============================================================================

def _safe_col(df, col, default=0):
    """Devuelve la columna si existe, o una serie de default"""
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def crear_features_xgb(df_jugador, target_col='Puntos'):
    """
    25 features usando todos los campos disponibles del JSON de SofaScore.

    Campos base (ya extraídos):
        Puntos, Rebotes, Asistencias, Minutos, Eficiencia, Localia, Dias_Descanso

    Campos nuevos del mismo JSON (sin llamadas extra):
        FG_Pct, 3P_Pct, Triples, Robos, Tapones, Perdidas, PlusMinus,
        Reb_Off, Reb_Def, FT_Pct
    """
    df = df_jugador.copy().sort_values('Fecha')
    features, targets = [], []

    for i in range(3, len(df)):
        window = df.iloc[max(0, i-7):i]

        # ── Features base 
        avg_3       = window.tail(3)[target_col].mean()
        avg_7       = window[target_col].mean()
        std_recent  = window[target_col].std() if len(window) > 1 else 0
        coef_var    = std_recent / avg_7 if avg_7 > 0 else 0

        trend_vals  = window.tail(5)[target_col].values
        trend_5     = np.polyfit(range(len(trend_vals)), trend_vals, 1)[0] if len(trend_vals) >= 2 else 0

        dias_desc   = df.iloc[i].get('Dias_Descanso', 2)
        dias_desc   = dias_desc if pd.notna(dias_desc) else 2
        es_local    = 1 if df.iloc[i]['Localia'] == 'Local' else 0

        min_avg     = window['Minutos'].mean()
        efic_avg    = window['Eficiencia'].mean() if 'Eficiencia' in window.columns else 0

        max_3       = window.tail(3)[target_col].max()
        min_3       = window.tail(3)[target_col].min()
        rango_3     = max_3 - min_3
        racha_pos   = sum(1 for x in window.tail(3)[target_col] if x > avg_7)

        b2b_data    = window[window['Dias_Descanso'] <= 1][target_col]
        rend_b2b    = b2b_data.mean() if len(b2b_data) > 0 else avg_7

        local_data  = window[window['Localia'] == 'Local'][target_col]
        visit_data  = window[window['Localia'] == 'Visitante'][target_col]
        diff_loc    = (local_data.mean() - visit_data.mean()) if len(local_data) > 0 and len(visit_data) > 0 else 0

        # ── Features nuevas 
        fg_pct_avg  = _safe_col(window, 'FG_Pct').mean()
        tp_pct_avg  = _safe_col(window, '3P_Pct').mean()
        triples_avg = _safe_col(window, 'Triples').mean()
        robos_avg   = _safe_col(window, 'Robos').mean()
        tapones_avg = _safe_col(window, 'Tapones').mean()
        perdidas_avg= _safe_col(window, 'Perdidas').mean()
        pm_avg      = _safe_col(window, 'PlusMinus').mean()       # +/-
        reb_off_avg = _safe_col(window, 'Reb_Off').mean()
        reb_def_avg = _safe_col(window, 'Reb_Def').mean()
        ft_pct_avg  = _safe_col(window, 'FT_Pct').mean()

        features.append({
            # Base
            'avg_3_games':      avg_3,
            'avg_7_games':      avg_7,
            'trend_5':          trend_5,
            'std_recent':       std_recent,
            'coef_var':         coef_var,
            'dias_descanso':    dias_desc,
            'es_local':         es_local,
            'minutos_avg':      min_avg,
            'eficiencia_avg':   efic_avg,
            'max_recent_3':     max_3,
            'min_recent_3':     min_3,
            'rango_recent':     rango_3,
            'racha_positiva':   racha_pos,
            'rendimiento_b2b':  rend_b2b,
            'diff_local_visita':diff_loc,
            
            'fg_pct_avg':       fg_pct_avg,
            'tp_pct_avg':       tp_pct_avg,
            'triples_avg':      triples_avg,
            'robos_avg':        robos_avg,
            'tapones_avg':      tapones_avg,
            'perdidas_avg':     perdidas_avg,
            'plus_minus_avg':   pm_avg,
            'reb_off_avg':      reb_off_avg,
            'reb_def_avg':      reb_def_avg,
            'ft_pct_avg':       ft_pct_avg,
        })
        targets.append(df.iloc[i][target_col])

    return pd.DataFrame(features), np.array(targets)


def entrenar_xgboost(df_equipo, jugador_nombre, target_col='Puntos'):
    """
    Entrena XGBoost con 25 features.
    Requiere mínimo 5 partidos.
    Retorna (modelo, scaler, metricas) o (None, None, None) si no hay datos.
    """
    df_jug = df_equipo[df_equipo['Jugador'] == jugador_nombre].copy()
    if len(df_jug) < 5:
        return None, None, None

    X, y = crear_features_xgb(df_jug, target_col)
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
        random_state=42,
        verbosity=0
    )
    modelo.fit(X_scaled, y)

    y_pred = modelo.predict(X_scaled)
    mae    = np.mean(np.abs(y - y_pred))
    rmse   = np.sqrt(np.mean((y - y_pred) ** 2))

    importances = dict(zip(X.columns, modelo.feature_importances_))
    sorted_imp  = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    return modelo, scaler, {
        'mae': mae,
        'rmse': rmse,
        'importance': sorted_imp,
        'n_samples': len(X)
    }


def predecir_xgboost(modelo, scaler, df_jugador, target_col='Puntos',
                     es_local=True, dias_descanso=2):
    """
    Genera predicción + intervalo dinámico con XGBoost.
    El intervalo se amplía proporcionalmente a la volatilidad del jugador.
    """
    if modelo is None:
        return None, None

    df = df_jugador.copy().sort_values('Fecha')
    window = df.tail(7)

    avg_3       = window.tail(3)[target_col].mean()
    avg_7       = window[target_col].mean()
    std_recent  = window[target_col].std() if len(window) > 1 else 0
    coef_var    = std_recent / avg_7 if avg_7 > 0 else 0

    trend_vals  = window.tail(5)[target_col].values
    trend_5     = np.polyfit(range(len(trend_vals)), trend_vals, 1)[0] if len(trend_vals) >= 2 else 0

    local_data  = window[window['Localia'] == 'Local'][target_col]
    visit_data  = window[window['Localia'] == 'Visitante'][target_col]
    diff_loc    = (local_data.mean() - visit_data.mean()) if len(local_data) > 0 and len(visit_data) > 0 else 0

    b2b_data    = window[window['Dias_Descanso'] <= 1][target_col]
    rend_b2b    = b2b_data.mean() if len(b2b_data) > 0 else avg_7

    features_prox = {
        'avg_3_games':      avg_3,
        'avg_7_games':      avg_7,
        'trend_5':          trend_5,
        'std_recent':       std_recent,
        'coef_var':         coef_var,
        'dias_descanso':    dias_descanso,
        'es_local':         1 if es_local else 0,
        'minutos_avg':      window['Minutos'].mean(),
        'eficiencia_avg':   window['Eficiencia'].mean() if 'Eficiencia' in window.columns else 0,
        'max_recent_3':     window.tail(3)[target_col].max(),
        'min_recent_3':     window.tail(3)[target_col].min(),
        'rango_recent':     window.tail(3)[target_col].max() - window.tail(3)[target_col].min(),
        'racha_positiva':   sum(1 for x in window.tail(3)[target_col] if x > avg_7),
        'rendimiento_b2b':  rend_b2b,
        'diff_local_visita':diff_loc,
        
        'fg_pct_avg':       _safe_col(window, 'FG_Pct').mean(),
        'tp_pct_avg':       _safe_col(window, '3P_Pct').mean(),
        'triples_avg':      _safe_col(window, 'Triples').mean(),
        'robos_avg':        _safe_col(window, 'Robos').mean(),
        'tapones_avg':      _safe_col(window, 'Tapones').mean(),
        'perdidas_avg':     _safe_col(window, 'Perdidas').mean(),
        'plus_minus_avg':   _safe_col(window, 'PlusMinus').mean(),
        'reb_off_avg':      _safe_col(window, 'Reb_Off').mean(),
        'reb_def_avg':      _safe_col(window, 'Reb_Def').mean(),
        'ft_pct_avg':       _safe_col(window, 'FT_Pct').mean(),
    }

    X_scaled = scaler.transform(pd.DataFrame([features_prox]))
    prediccion = modelo.predict(X_scaled)[0]

    # Intervalo dinámico según volatilidad
    if coef_var < 0.2:
        margen = prediccion * 0.15
    elif coef_var < 0.35:
        margen = prediccion * 0.25
    else:
        margen = prediccion * 0.35

    return prediccion, (max(0, prediccion - margen), prediccion + margen)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def predecir(df_equipo, jugador_nombre, target_col='Puntos',
             es_local=True, dias_descanso=2):
    """
    Sistema híbrido con fallback automático de 2 niveles.

    Nivel 1: XGBoost con 25 features (requiere ≥5 partidos)
    Nivel 2: Ensemble RF+Regresión+WMA (requiere ≥3 partidos)

    Retorna:
        dict con prediccion, intervalo, confianza, nivel, metricas_modelo
    """
    df_jug = df_equipo[df_equipo['Jugador'] == jugador_nombre].copy().sort_values('Fecha')

    if df_jug.empty:
        return None

    # ── Intentar Nivel 1: XGBoost 
    modelo, scaler, metricas = entrenar_xgboost(df_equipo, jugador_nombre, target_col)

    if modelo is not None:
        prediccion, intervalo = predecir_xgboost(
            modelo, scaler, df_jug, target_col, es_local, dias_descanso
        )
        if prediccion is not None:
            confianza = max(0.0, min(1.0, 1 - (metricas['mae'] / df_jug[target_col].mean())))
            return {
                'prediccion':      round(max(0, prediccion), 1),
                'intervalo':       intervalo,
                'confianza':       round(confianza * 100, 1),
                'nivel':           'xgboost',
                'metricas_modelo': metricas,
                'modelo':          modelo,
                'scaler':          scaler,
            }

    # ── Fallback Nivel 2: Ensemble 
    resultado = predecir_ensemble(df_jug, target_col)
    return {
        'prediccion':      resultado['prediccion'],
        'intervalo':       resultado['intervalo'],
        'confianza':       resultado['confianza'],
        'nivel':           'fallback',
        'metricas_modelo': {'ajustes': resultado['ajustes']},
        'modelo':          None,
        'scaler':          None,
    }