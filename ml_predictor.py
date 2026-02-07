# ml_predictor.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def prediccion_wma(df, metrica, ventana=5):
    """Promedio ponderado - más peso a partidos recientes"""
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
    
    # Predecir siguiente partido
    return modelo.predict([[len(df)]])[0]

def crear_features_simplificadas(df, metrica):
    """Solo 6 features críticas - más robusto"""
    features = []
    targets = []
    
    for i in range(3, len(df)):
        ventana = df.iloc[max(0, i-5):i]
        
        features.append([
            ventana[metrica].mean(),           # Promedio reciente
            ventana[metrica].std() if len(ventana) > 1 else 0,  # Volatilidad
            ventana['Minutos'].mean(),         # Uso
            1 if df.iloc[i]['Localia'] == 'Local' else 0,  # Casa/Visitante
            df.iloc[i].get('Dias_Descanso', 2) if pd.notna(df.iloc[i].get('Dias_Descanso')) else 2,
            ventana[metrica].iloc[-1]          # Último valor (momentum)
        ])
        
        targets.append(df.iloc[i][metrica])
    
    return np.array(features), np.array(targets)

def prediccion_rf(df, metrica):
    """Random Forest simplificado"""
    if len(df) < 10:
        return None
    
    X, y = crear_features_simplificadas(df, metrica)
    
    modelo = RandomForestRegressor(
        n_estimators=50,
        max_depth=3,
        min_samples_split=3,
        random_state=42
    )
    
    modelo.fit(X, y)
    
    # Feature del próximo partido
    ventana = df.tail(5)
    X_next = [[
        ventana[metrica].mean(),
        ventana[metrica].std() if len(ventana) > 1 else 0,
        ventana['Minutos'].mean(),
        1,  # Asume local
        2,  # Asume 2 días descanso
        df[metrica].iloc[-1]
    ]]
    
    return modelo.predict(X_next)[0]

def aplicar_ajustes_contextuales(df, metrica):
    """Ajustes basados en tendencias y contexto"""
    ajustes = {'items': [], 'total': 0}
    
    # AJUSTE 1: Tendencia reciente
    if len(df) >= 6:
        ultimos_3 = df.tail(3)[metrica].mean()
        previos_3 = df.iloc[-6:-3][metrica].mean()
        diferencia = ultimos_3 - previos_3
        
        if abs(diferencia) > 2:
            ajustes['items'].append(f"Tendencia: {diferencia:+.1f}")
            ajustes['total'] += diferencia * 0.3
    
    # AJUSTE 2: Racha
    if len(df) >= 3:
        promedio_total = df[metrica].mean()
        ultimos = df.tail(3)[metrica]
        
        if all(ultimos > promedio_total):
            ajustes['items'].append("Racha caliente: +1.5")
            ajustes['total'] += 1.5
        elif all(ultimos < promedio_total):
            ajustes['items'].append("Racha fría: -1.5")
            ajustes['total'] -= 1.5
    
    # AJUSTE 3: Volatilidad
    cv = df[metrica].std() / df[metrica].mean() if df[metrica].mean() > 0 else 0
    if cv > 0.4:
        ajustes['items'].append("Alta volatilidad: -1.0")
        ajustes['total'] -= 1.0
    
    return ajustes

def calcular_confianza(num_partidos, std):
    """Score de confianza 0-100%"""
    factor_datos = min(num_partidos / 10, 1.0) * 50
    factor_estabilidad = max(0, (1 - std / 10)) * 50
    
    return min(100, factor_datos + factor_estabilidad)

def predecir_inteligente_v3(df_jugador, metrica='Puntos'):
    """
    Sistema híbrido de predicción con 3 niveles de fallback
    """
    df = df_jugador.copy().sort_values('Fecha')
    
    predicciones = []
    
    # Intentar RF
    pred_rf = prediccion_rf(df, metrica)
    if pred_rf is not None:
        predicciones.append(('RF', pred_rf, 0.4))
    
    # Intentar Regresión
    pred_reg = prediccion_regression(df, metrica)
    if pred_reg is not None:
        predicciones.append(('REG', pred_reg, 0.35))
    
    # Siempre WMA
    pred_wma = prediccion_wma(df, metrica)
    predicciones.append(('WMA', pred_wma, 0.25))
    
    # ENSEMBLE
    total_peso = sum(p[2] for p in predicciones)
    prediccion_final = sum(p[1] * p[2] for p in predicciones) / total_peso
    
    # AJUSTES
    ajustes = aplicar_ajustes_contextuales(df, metrica)
    prediccion_ajustada = prediccion_final + ajustes['total']
    
    # INTERVALO
    std_recent = df.tail(5)[metrica].std()
    margen = std_recent * 1.5
    
    return {
        'prediccion': round(max(0, prediccion_ajustada), 1),
        'intervalo': (
            max(0, prediccion_ajustada - margen),
            prediccion_ajustada + margen
        ),
        'modelos_usados': [p[0] for p in predicciones],
        'confianza': calcular_confianza(len(df), std_recent),
        'ajustes': ajustes
    }