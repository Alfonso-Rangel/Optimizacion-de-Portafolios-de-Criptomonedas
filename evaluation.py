# evaluation.py
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


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
        print(f"  Retorno Anualizado (Compuesto): {s['annual_return_%']:.2f}%")
        print(f"  Retorno Total (Periodo): {s['total_return_%']:.2f}%")
        print(f"  Curtosis (Fisher): {s['kurtosis']:.4f}")


# ============================================================
#  GRÁFICAS DE DESEMPEÑO
# ============================================================

def graficar_retornos_acumulados(retornos_dict, titulo=None):
    plt.figure(figsize=(12, 6))
    styles = {
        "naive":  {"c": "gray", "ls": "--", "label": "Naive (1/N)"},
        "sharpe": {"c": "blue", "ls": "-.", "label": "Max Sharpe (PSO)"},
        "curt":   {"c": "green", "ls": ":",  "label": "Min Curtosis (PSO)"},
        "comp":   {"c": "red",   "ls": "-",  "label": "Multiobjetivo (MOPSO)"}
    }

    for key, style in styles.items():
        if key in retornos_dict and not retornos_dict[key].empty:
            cum_ret = (1 + retornos_dict[key]).cumprod() - 1
            plt.plot(
                cum_ret.index,
                cum_ret.values,
                color=style["c"],
                linestyle=style["ls"],
                label=style["label"],
                linewidth=1.5
            )

    plt.legend(fontsize=10)

    if titulo:
        plt.title(titulo, fontsize=14)
    else:
        plt.title("Crecimiento del Portafolio (Retornos Acumulados)", fontsize=14)

    plt.ylabel("Retorno Acumulado (Fracción)", fontsize=12)
    plt.xlabel("Fecha", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
#  MÉTRICAS
# ============================================================

def summary_metrics(series_excess, name):
    """
    Cálculo de métricas consistente con toda la metodología:
    - retornos simples horarios
    - sharpe horario
    - anualización compuesta correcta
    """
    series = series_excess.dropna()
    total_hours = len(series)

    mean_h = series.mean()
    std_h = series.std()
    sharpe = mean_h / std_h if std_h != 0 else np.nan

    # Retorno total del periodo
    total_return = (1 + series).prod() - 1
    total_return_pct = total_return * 100

    # ANUALIZACIÓN COMPUESTA CORRECTA
    if total_hours > 0:
        annual_factor = 8760 / total_hours
        annual_return = (1 + total_return)**annual_factor - 1
    else:
        annual_return = np.nan

    kurt = kurtosis(series.values, fisher=True, bias=False)

    return {
        "name": name,
        "hours": total_hours,
        "mean_hour": mean_h,
        "std_hour": std_h,
        "sharpe": sharpe,
        "total_return_%": total_return_pct,
        "annual_return_%": annual_return * 100,
        "kurtosis": kurt
    }


# ============================================================
#  TABLA COMPARATIVA
# ============================================================

def mostrar_tabla_comparativa(metricas_list):
    df = pd.DataFrame(metricas_list)
    df = df.rename(columns={
        'name': 'Estrategia',
        'mean_hour': 'Retorno Horario Prom. (fracción)',
        'std_hour': 'Desv. Est. Horaria',
        'sharpe': 'Sharpe (Horario)',
        'annual_return_%': 'Retorno Anualizado (%)',
        'total_return_%': 'Retorno Total (Periodo) (%)',
        'kurtosis': 'Curtosis (Fisher)',
        'hours': 'Horas'
    })

    selected_cols = [
        'Estrategia',
        'Retorno Anualizado (%)',
        'Retorno Total (Periodo) (%)',
        'Sharpe (Horario)',
        'Curtosis (Fisher)',
        'Retorno Horario Prom. (fracción)',
        'Desv. Est. Horaria'
    ]

    df = df[selected_cols]

    df_styled = df.style.format({
        'Retorno Anualizado (%)': '{:.2f}',
        'Retorno Total (Periodo) (%)': '{:.2f}',
        'Sharpe (Horario)': '{:.4f}',
        'Curtosis (Fisher)': '{:.4f}',
        'Retorno Horario Prom. (fracción)': '{:.6f}',
        'Desv. Est. Horaria': '{:.6f}',
    })

    print("\n\n=== TABLA COMPARATIVA DE MÉTRICAS CLAVE ===")
    print(df_styled.to_string())


# ============================================================
#  FRENTE DE PARETO
# ============================================================

def graficar_frente_pareto_global(frentes):
    if len(frentes) == 0:
        print("Sin frentes para graficar.")
        return

    todos = np.vstack(frentes)
    curtosis_vals = todos[:, 0]
    minus_sharpe_vals = todos[:, 1]
    sharpe_real = -minus_sharpe_vals

    x_clean = curtosis_vals
    y_clean = sharpe_real

    if len(x_clean) == 0:
        print("No hay soluciones válidas.")
        return

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

    if len(x_front) > 0:
        objs_original = np.column_stack((x_front, -y_front))
        min_vals = np.min(objs_original, axis=0)
        max_vals = np.max(objs_original, axis=0)
        range_vals = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        front_norm = (objs_original - min_vals) / range_vals
        distancias = np.sum(front_norm**2, axis=1)
        idx_utopia = np.argmin(distancias)
        utopia_x, utopia_y = x_front[idx_utopia], y_front[idx_utopia]
    else:
        utopia_x, utopia_y = None, None

    plt.figure(figsize=(12, 7))

    plt.scatter(x_clean, y_clean, s=10, c='gray', alpha=0.15, label="Soluciones Exploradas")

    plt.plot(x_front, y_front, c='firebrick', alpha=0.6, linewidth=1.5)
    plt.scatter(x_front, y_front, c='firebrick', s=30, label="Frente de Pareto", zorder=3)

    if len(x_front) > 0:
        idx_max_sharpe = np.argmax(y_front)
        plt.scatter(x_front[idx_max_sharpe], y_front[idx_max_sharpe],
                    c='blue', s=80, marker='^', zorder=4, label="Máx Sharpe")

        idx_min_curtosis = np.argmin(x_front)
        plt.scatter(x_front[idx_min_curtosis], y_front[idx_min_curtosis],
                    c='green', s=80, marker='v', zorder=4, label="Min Curtosis")

        if utopia_x is not None:
            plt.scatter(utopia_x, utopia_y, c='gold', s=120, marker='*',
                        edgecolors='black', zorder=5, label="Punto de Utopía")
            plt.annotate(
                f" Utopía\n Sharpe: {utopia_y:.3f}\n Curt: {utopia_x:.1f}",
                (utopia_x, utopia_y),
                xytext=(15, 15),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->')
            )

    plt.xscale('log')

    if len(y_front) > 0:
        rango = y_front.max() - y_front.min()
        plt.ylim(y_front.min() - rango*0.1, y_front.max() + rango*0.2)

    plt.xlabel("Curtosis (log) → menor es mejor", fontsize=11)
    plt.ylabel("Sharpe → mayor es mejor", fontsize=11)
    plt.title("Frontera Eficiente Global", fontsize=14, fontweight='bold')

    plt.legend(loc='upper left')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()

def graficar_barras_pesos(pesos_por_estrategia, nombres_activos, titulo_general=None, max_barras=100):
    colores_base = [
        "#1793d1", "#747c92", "#1a1a2e", "#6fbf73", "#FF69B4",
        "#ff4b3e", "#5d2f8e", "#7c3f58", "#ffe548", "#a80030"
    ]
    
    # Asignar colores fijos a los activos
    n_activos = len(nombres_activos)
    colores_activos = {activo: colores_base[i % len(colores_base)] for i, activo in enumerate(nombres_activos)}

    # Verificar datos
    estrategias_con_datos = [e for e, p in pesos_por_estrategia.items() if p]
    if not estrategias_con_datos:
        print("ADVERTENCIA: No hay datos de pesos.")
        return

    # Bucle por cada estrategia (genera un gráfico por estrategia)
    for estrategia, pesos_dict in pesos_por_estrategia.items():
        if not pesos_dict: continue
        
        # --- PROCESAMIENTO DE DATOS ---
        fechas = sorted(pesos_dict.keys())
        
        # Filtrado si hay demasiadas barras (para mantener legibilidad)
        if len(fechas) > max_barras:
            indices = np.linspace(0, len(fechas)-1, max_barras, dtype=int)
            fechas_seleccionadas = [fechas[i] for i in indices]
        else:
            fechas_seleccionadas = fechas
            
        n_ventanas = len(fechas_seleccionadas)
        datos = np.zeros((n_ventanas, n_activos))
        
        # Llenar matriz de datos
        for i, fecha in enumerate(fechas_seleccionadas):
            pesos = pesos_dict[fecha]
            # Ajustar longitud de pesos si no coincide con activos
            lista_pesos = list(pesos)
            if len(lista_pesos) < n_activos:
                lista_pesos += [0.0] * (n_activos - len(lista_pesos))
            elif len(lista_pesos) > n_activos:
                lista_pesos = lista_pesos[:n_activos]
            
            arr = np.array(lista_pesos, dtype=float)
            suma = arr.sum()
            datos[i, :] = (arr / suma * 100) if suma > 0 else 0

        # Ordenar activos para el apilado (según última ventana)
        if datos.shape[0] > 0:
            ultima_participacion = datos[-1, :]
            indices_ordenados = np.argsort(ultima_participacion) # De menor a mayor para apilar bottom-up
        else:
            indices_ordenados = np.arange(n_activos)
            
        datos_ordenados = datos[:, indices_ordenados]
        nombres_ordenados = [nombres_activos[i] for i in indices_ordenados]

        # --- GRAFICACIÓN ---
        fig, ax = plt.subplots(figsize=(18, 12))
        fig.patch.set_facecolor('white')
        
        fechas_mpl = mdates.date2num(fechas_seleccionadas)
        
        # Calcular ancho para el espacio entre barras:
        if len(fechas_seleccionadas) > 1:
            diffs = [(fechas_seleccionadas[i+1] - fechas_seleccionadas[i]).days for i in range(len(fechas_seleccionadas)-1)]
            ancho_promedio = np.mean(diffs)
            width = ancho_promedio * 0.95 
        else:
            width = 30

        bottom = np.zeros(n_ventanas)
        
        # Dibujar Barras
        for i in range(n_activos):
            valores = datos_ordenados[:, i]
            nombre_activo = nombres_ordenados[i]
            color = colores_activos[nombre_activo]
            
            # Barras sin borde negro
            bars = ax.bar(fechas_mpl, valores, width=width, bottom=bottom, 
                          color=color, label=nombre_activo, edgecolor='none', align='center')
            
            # Etiquetas INTERNAS (Números blancos rotados)
            for j, rect in enumerate(bars):
                height = rect.get_height()
                
                # Mostrar el porcentaje en CADA barra, siempre que la altura sea suficiente.
                if height > 4.0: 
                    # Pequeño ajuste para activos con 10% y evitar saturación en nombres largos
                    if height < 10.5 and n_activos > 10:
                        fontsize_val = 7
                    else:
                        fontsize_val = 8
                        
                    cy = rect.get_y() + rect.get_height() / 2
                    ax.text(rect.get_x() + rect.get_width() / 2, cy, f"{height:.1f}", 
                            ha='center', va='center', color='white', fontsize=fontsize_val, rotation=90, weight='bold')
            
            bottom += valores

        # --- ETIQUETAS A LA DERECHA (Espaciado uniforme y color) ---
        # Coordenada X para las etiquetas
        ultima_fecha_num = fechas_mpl[-1]
        x_pos_labels = ultima_fecha_num + (width * 0.6) 
        
        step_y = 100.0 / n_activos
        current_y = step_y / 2
        
        for nombre in nombres_ordenados:
            color_texto = colores_activos[nombre]
            
            ax.text(x_pos_labels, current_y, nombre, 
                    va='center', ha='left', 
                    fontsize=11, weight='bold', color=color_texto)
            
            current_y += step_y

        # --- FORMATO FINAL ---
        
        # Títulos
        titulo_estr = {
            'naive': 'Naive 1/N', 'sharpe': 'Max Sharpe', 'curt': 'Min Curtosis', 'comp': 'Multiobjetivo'
        }.get(estrategia, estrategia.capitalize())
        
        if titulo_general:
            plt.text(x=0.0, y=1.05, s=titulo_general, fontsize=18, weight='bold', transform=ax.transAxes)
        plt.text(x=0.0, y=1.02, s=f"Estrategia: {titulo_estr}", fontsize=12, color='gray', transform=ax.transAxes)

        # Ejes
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.set_ylabel("Participación (%)", labelpad=10, fontsize=12)
        
        # --- MODIFICACIÓN: ETIQUETAS DEL EJE X CON FECHAS DE REBALANCEO ---
        # Establecer las marcas del eje X en las fechas de las barras
        ax.set_xticks(fechas_mpl)
        
        # Formatear las etiquetas del eje X según el número de ventanas
        if n_ventanas <= 50:
            # Hasta 50 ventanas: mostrar día-mes
            etiquetas_fechas = [fecha.strftime('%d-%m-%y') for fecha in fechas_seleccionadas]
            ax.set_xticklabels(etiquetas_fechas, rotation=45, ha='right', fontsize=9)
        else:
            # Muchas ventanas: mostrar solo cada 2ª, 3ª o 4ª fecha
            intervalo = max(1, n_ventanas // 20) 
            etiquetas_fechas = []
            for i, fecha in enumerate(fechas_seleccionadas):
                if i % intervalo == 0:
                    # Para intervalos largos, mostrar día-mes-año completo
                    if intervalo >= 4:
                        etiquetas_fechas.append(fecha.strftime('%d-%m-%y'))
                    else:
                        etiquetas_fechas.append(fecha.strftime('%d-%m'))
                else:
                    etiquetas_fechas.append('')
            
            ax.set_xticklabels(etiquetas_fechas, rotation=45, ha='right', fontsize=9)
            
        
        # Ajustar límites X
        fecha_inicio_mpl = fechas_mpl[0] - (width * 0.7) 
        espacio_derecha_dias = width * 8 
        fecha_fin_mpl = fechas_mpl[-1] + espacio_derecha_dias
        ax.set_xlim(fecha_inicio_mpl, fecha_fin_mpl)
        
        # Estética limpia
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        # Añadir líneas verticales tenues en cada rebalanceo para mejor orientación
        for fecha_mpl in fechas_mpl:
            ax.axvline(x=fecha_mpl, color='gray', alpha=0.15, linewidth=0.5)
        
        plt.tight_layout()
        plt.show()