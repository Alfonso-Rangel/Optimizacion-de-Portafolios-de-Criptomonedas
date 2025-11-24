import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

# ============================================================
#  RESUMEN Y MÉTRICAS
# ============================================================

def mostrar_resumen_metrica(metricas_list):
    print("\n=== RESUMEN DE MÉTRICAS ===")
    for s in metricas_list:
        print(f"\n-- {s['name']} --")
        print(f"  Horas analizadas: {s['hours']}")
        print(f"  Retorno horario prom: {s['mean_hour']:.6f}")
        print(f"  Desviación est. horaria: {s['std_hour']:.6f}")
        print(f"  Sharpe Ratio (Horario): {s['sharpe']:.6f}")

        if "weekly_return_%" in s:
            print(f"  Retorno Total (Periodo): {s['weekly_return_%']:.2f}%")
        elif "annual_return_%" in s:
            print(f"  Retorno Anualizado (Est.): {s['annual_return_%']:.2f}%")
        
        print(f"  Curtosis (Fisher): {s['kurtosis']:.4f}")


# ============================================================
#  GRÁFICAS DE DESEMPEÑO
# ============================================================

def graficar_retornos_acumulados(retornos_dict):
    plt.figure(figsize=(12, 6))

    # Colores y estilos profesionales
    styles = {
        "naive":  {"c": "gray", "ls": "--", "label": "Naive 1/N"},
        "sharpe": {"c": "blue", "ls": "-.", "label": "Max Sharpe (PSO)"},
        "kurt":   {"c": "green", "ls": ":",  "label": "Min Curtosis (PSO)"},
        "comp":   {"c": "red",   "ls": "-",  "label": "Compuesto (MOPSO Knee Point)"}
    }

    for key, style in styles.items():
        if key in retornos_dict and not retornos_dict[key].empty:
            cum_ret = (1 + retornos_dict[key]).cumprod() - 1
            plt.plot(cum_ret.index, cum_ret.values, 
                     color=style["c"], linestyle=style["ls"], label=style["label"], linewidth=1.5)

    plt.legend(fontsize=10)
    plt.title("Crecimiento del Portafolio (Retornos Acumulados)", fontsize=14)
    plt.ylabel("Retorno Acumulado (Fracción)", fontsize=12)
    plt.xlabel("Fecha", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def graficar_distribucion_kde(retornos_dict):
    plt.figure(figsize=(10, 5))
    
    labels_map = {
        "naive": "Naive 1/N",
        "sharpe": "Sharpe-PSO",
        "kurt": "Curtosis-PSO",
        "comp": "Compuesto (MOPSO)"
    }
    
    for key, label in labels_map.items():
        if key in retornos_dict:
            retornos_dict[key].dropna().plot.kde(label=label, linewidth=2)

    plt.title("Densidad de Probabilidad de Retornos Horarios", fontsize=14)
    plt.xlabel("Retorno Horario (Fracción)", fontsize=12)
    plt.ylabel("Densidad", fontsize=12)
    plt.xlim(-0.05, 0.05) 
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def graficar_pesos_promedio(pesos_dict, n_activos, asset_names):
    # Calcular promedios
    matrix_rows = []
    indices = []
    
    # Orden específico
    keys_order = ["naive", "sharpe", "kurt", "comp"]
    
    for key in keys_order:
        if key in pesos_dict and pesos_dict[key]:
            # Extraer valores del diccionario de pesos
            arr_pesos = np.vstack(list(pesos_dict[key].values()))
            media = np.mean(arr_pesos, axis=0)
            matrix_rows.append(media)
            indices.append(key.capitalize())
    
    if not matrix_rows:
        return

    df_w = pd.DataFrame(matrix_rows, index=indices, columns=asset_names)

    plt.figure(figsize=(10, 5))
    plt.imshow(df_w, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    cbar = plt.colorbar(label="Peso Promedio (0-1)")
    
    # Etiquetas
    plt.yticks(range(len(df_w)), df_w.index, fontsize=11)
    plt.xticks(range(n_activos), df_w.columns, rotation=45, ha="right", fontsize=10)
    
    # Anotar valores dentro de las celdas
    for i in range(len(df_w)):
        for j in range(n_activos):
            text = plt.text(j, i, f"{df_w.iloc[i, j]:.2f}",
                           ha="center", va="center", color="black" if df_w.iloc[i, j] < 0.5 else "white", fontsize=8)

    plt.title("Distribución Promedio de Activos por Estrategia", fontsize=14)
    plt.tight_layout()
    plt.show()

# Funciones placeholder para compatibilidad
def graficar_distribucion_retornos(x): pass
def graficar_box_plot(x): pass


# ============================================================
#   FRENTE PARETO GLOBAL (VERSIÓN PROFESIONAL)
# ============================================================

def graficar_frente_pareto_global(frentes):
    if len(frentes) == 0:
        print("Sin frentes para graficar.")
        return

    # Consolidar todos los frentes de todas las ventanas
    todos = np.vstack(frentes)

    # --- 1. Procesamiento de Datos ---
    curtosis_vals = todos[:, 0]
    minus_sharpe_vals = todos[:, 1]
    sharpe_real = -minus_sharpe_vals # Invertir para ver Sharpe positivo

    # Filtrado de Outliers (Penalizaciones del algoritmo)
    mask_validos = (sharpe_real > -5) & (curtosis_vals < 1e6)
    
    x_clean = curtosis_vals[mask_validos]
    y_clean = sharpe_real[mask_validos]

    if len(x_clean) == 0:
        print("No hay soluciones válidas.")
        return

    # --- 2. Cálculo de Dominancia (Frente Visual) ---
    puntos_calc = np.column_stack((x_clean, -y_clean))
    
    is_pareto = np.ones(len(puntos_calc), dtype=bool)
    for i, c in enumerate(puntos_calc):
        if is_pareto[i]:
            dominado = np.any(np.all(puntos_calc <= c, axis=1) & np.any(puntos_calc < c, axis=1))
            if dominado:
                is_pareto[i] = False

    x_front = x_clean[is_pareto]
    y_front = y_clean[is_pareto]
    
    idx_order = np.argsort(x_front)
    x_front = x_front[idx_order]
    y_front = y_front[idx_order]

    # --- 3. Identificación del "Knee Point" (Codo) ---
    if len(x_front) > 2:
        x_norm = (x_front - x_front.min()) / (x_front.max() - x_front.min() + 1e-9)
        y_norm = (y_front - y_front.min()) / (y_front.max() - y_front.min() + 1e-9)
        
        p1 = np.array([x_norm[0], y_norm[0]])
        p2 = np.array([x_norm[-1], y_norm[-1]])
        vec_linea = p2 - p1
        vec_unit = vec_linea / np.linalg.norm(vec_linea)
        
        distancias = []
        for i in range(len(x_norm)):
            p = np.array([x_norm[i], y_norm[i]])
            proj = p1 + np.dot(p - p1, vec_unit) * vec_unit
            dist = np.linalg.norm(p - proj)
            distancias.append(dist)
            
        idx_knee = np.argmax(distancias)
        knee_x, knee_y = x_front[idx_knee], y_front[idx_knee]
    else:
        knee_x, knee_y = None, None

    # --- 4. Graficación ---
    plt.figure(figsize=(12, 7))
    
    plt.scatter(x_clean, y_clean, s=10, c='gray', alpha=0.15, label="Soluciones Exploradas")
    
    plt.plot(x_front, y_front, c='firebrick', alpha=0.6, linewidth=1.5)
    plt.scatter(x_front, y_front, c='firebrick', s=30, label="Frente de Pareto", zorder=3)

    if len(x_front) > 0:
        # Max Return
        plt.scatter(x_front[-1], y_front[-1], c='blue', s=80, marker='^', zorder=4, label="Máx Sharpe (Riesgoso)")
        # Min Risk
        plt.scatter(x_front[0], y_front[0], c='green', s=80, marker='v', zorder=4, label="Min Curtosis (Conservador)")
        
        if knee_x is not None:
            plt.scatter(knee_x, knee_y, c='gold', s=120, marker='*', edgecolors='black', zorder=5, label="Punto de Equilibrio (Codo)")
            plt.annotate(f" Selección\n Sharpe: {knee_y:.3f}\n Kurt: {knee_x:.1f}", 
                         (knee_x, knee_y), xytext=(15, -15), textcoords='offset points', 
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.xscale('log') 
    
    if len(y_front) > 0:
        y_rango = y_front.max() - y_front.min()
        plt.ylim(y_front.min() - y_rango*0.1, y_front.max() + y_rango*0.2)

    plt.xlabel("Riesgo de Cola: Curtosis (Log Scale) $\\rightarrow$ Menor es mejor", fontsize=11)
    plt.ylabel("Eficiencia: Ratio de Sharpe Horario $\\rightarrow$ Mayor es mejor", fontsize=11)
    plt.title("Frontera de Eficiencia Multiobjetivo (Global)", fontsize=14, fontweight='bold')
    
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def summary_metrics(series_excess, name):
    series = series_excess.dropna()
    mean_h = series.mean()
    std_h = series.std()
    sharpe = mean_h / std_h if std_h != 0 else np.nan
    
    total_return = (1 + series).prod() - 1
    total_return_pct = total_return * 100
    
    # Calcular Curtosis de Fisher (normal=0) para que sea consistente con el Java
    # fisher=True es el default en scipy, pero lo explicitamos.
    kurt = kurtosis(series.values, fisher=True, bias=False)
    total_hours = len(series)
    
    return {
        "name": name, 
        "hours": total_hours, 
        "mean_hour": mean_h, 
        "std_hour": std_h,
        "sharpe": sharpe, 
        "weekly_return_%": total_return_pct,
        "annual_return_%": mean_h * 24 * 365 * 100,
        "kurtosis": kurt
    }

def mostrar_tabla_comparativa(metricas_list):
    df = pd.DataFrame(metricas_list)
    df = df.rename(columns={
        'name': 'Estrategia',
        'mean_hour': 'Retorno Horario Prom. (fracción)',
        'std_hour': 'Desv. Est. Horaria',
        'sharpe': 'Sharpe (Horario)',
        'annual_return_%': 'Retorno Anualizado (%)',
        'kurtosis': 'Curtosis (Fisher)',
        'hours': 'Horas'
    })

    selected_cols = []
    for col in [
        'Estrategia',
        'Retorno Anualizado (%)',
        'Sharpe (Horario)',
        'Curtosis (Fisher)',
        'Retorno Horario Prom. (fracción)',
        'Desv. Est. Horaria'
    ]:
        if col in df.columns:
            selected_cols.append(col)

    df = df[selected_cols]

    df_styled = df.style.format({
        'Retorno Anualizado (%)': '{:.2f}',
        'Sharpe (Horario)': '{:.4f}',
        'Curtosis (Fisher)': '{:.4f}',
        'Retorno Horario Prom. (fracción)': '{:.6f}',
        'Desv. Est. Horaria': '{:.6f}',
    })

    print("\n\n=== TABLA COMPARATIVA DE MÉTRICAS CLAVE ===")
    print(df_styled.to_string())