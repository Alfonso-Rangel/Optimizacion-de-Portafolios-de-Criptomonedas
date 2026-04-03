import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis


def summary_metrics(series_excess, name):

    series = series_excess.dropna()

    mean_h = series.mean()
    std_h = series.std()

    sharpe = mean_h / std_h if std_h != 0 else np.nan

    total_return = (1 + series).prod() - 1

    total_hours = len(series)

    annual_factor = 8760 / total_hours

    annual_return = (1 + total_return)**annual_factor - 1

    kurt = kurtosis(series.values, fisher=True, bias=False)

    return {
        "name": name,
        "sharpe": sharpe,
        "annual_return_%": annual_return * 100,
        "total_return_%": total_return * 100,
        "kurtosis": kurt
    }


def mostrar_resumen_metrica(metricas_list):

    print("\n=== RESUMEN MÉTRICAS ===")

    for m in metricas_list:

        print("\n", m["name"])
        print("Sharpe:", m["sharpe"])
        print("Retorno anual:", m["annual_return_%"])
        print("Curtosis:", m["kurtosis"])


def mostrar_tabla_comparativa(metricas_list):

    df = pd.DataFrame(metricas_list)

    print("\n=== TABLA COMPARATIVA ===")

    print(df)


def graficar_retornos_acumulados(retornos_dict):

    plt.figure(figsize=(10,6))

    for k,v in retornos_dict.items():

        cum = (1+v).cumprod()-1

        plt.plot(cum.index,cum.values,label=k.upper())

    plt.legend()
    plt.title("Retornos acumulados")

    plt.savefig("figures/retornos_acumulados_algoritmos.png")

    plt.close()


def graficar_drawdown(retornos):

    plt.figure(figsize=(10,6))

    for name, series in retornos.items():

        wealth = (1+series).cumprod()
        peak = wealth.cummax()

        dd = (wealth-peak)/peak

        plt.plot(dd,label=name.upper())

    plt.legend()

    plt.title("Drawdown")

    plt.savefig("figures/drawdown_algoritmos.png")

    plt.close()


def graficar_boxplot_retornos(retornos):

    data = []
    labels = []

    for k,v in retornos.items():
        data.append(v)
        labels.append(k.upper())

    plt.figure(figsize=(8,5))

    plt.boxplot(data,labels=labels)

    plt.title("Distribución de retornos")

    plt.savefig("figures/boxplot_retornos_algoritmos.png")

    plt.close()


def graficar_hipervolumen_comparado(df_mopso, df_nsga):

    plt.figure(figsize=(10,6))

    plt.plot(df_mopso.index, df_mopso["Promedio"], label="MOPSO")
    plt.plot(df_nsga.index, df_nsga["Promedio"], label="NSGA-II")

    plt.legend()

    plt.title("Hipervolumen comparado")

    plt.savefig("figures/hipervolumen_algoritmos.png")

    plt.close()


def graficar_barras_pesos(pesos_por_estrategia, nombres_activos, titulo_general=None, max_barras=100):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MultipleLocator

    os.makedirs("figures", exist_ok=True)

    colores_base = [
        "#1793d1", "#747c92", "#1a1a2e", "#6fbf73", "#FF69B4",
        "#ff4b3e", "#5d2f8e", "#7c3f58", "#ffe548", "#a80030"
    ]

    n_activos = len(nombres_activos)

    colores_activos = {
        activo: colores_base[i % len(colores_base)]
        for i, activo in enumerate(nombres_activos)
    }

    estrategias_con_datos = [e for e, p in pesos_por_estrategia.items() if p]

    if not estrategias_con_datos:
        print("ADVERTENCIA: No hay datos de pesos.")
        return

    for estrategia, pesos_dict in pesos_por_estrategia.items():

        if not pesos_dict:
            continue

        fechas = sorted(pesos_dict.keys())

        if len(fechas) > max_barras:
            indices = np.linspace(0, len(fechas)-1, max_barras, dtype=int)
            fechas_seleccionadas = [fechas[i] for i in indices]
        else:
            fechas_seleccionadas = fechas

        n_ventanas = len(fechas_seleccionadas)

        datos = np.zeros((n_ventanas, n_activos))

        for i, fecha in enumerate(fechas_seleccionadas):

            pesos = pesos_dict[fecha]

            lista_pesos = list(pesos)

            if len(lista_pesos) < n_activos:
                lista_pesos += [0.0] * (n_activos - len(lista_pesos))
            elif len(lista_pesos) > n_activos:
                lista_pesos = lista_pesos[:n_activos]

            arr = np.array(lista_pesos, dtype=float)

            suma = arr.sum()

            datos[i, :] = (arr / suma * 100) if suma > 0 else 0

        if datos.shape[0] > 0:
            ultima_participacion = datos[-1, :]
            indices_ordenados = np.argsort(ultima_participacion)
        else:
            indices_ordenados = np.arange(n_activos)

        datos_ordenados = datos[:, indices_ordenados]
        nombres_ordenados = [nombres_activos[i] for i in indices_ordenados]

        fig, ax = plt.subplots(figsize=(18, 12))
        fig.patch.set_facecolor('white')

        fechas_mpl = mdates.date2num(fechas_seleccionadas)

        if len(fechas_seleccionadas) > 1:
            diffs = [
                (fechas_seleccionadas[i+1] - fechas_seleccionadas[i]).days
                for i in range(len(fechas_seleccionadas)-1)
            ]
            ancho_promedio = np.mean(diffs)
            width = ancho_promedio * 0.95
        else:
            width = 30

        bottom = np.zeros(n_ventanas)

        for i in range(n_activos):

            valores = datos_ordenados[:, i]
            nombre_activo = nombres_ordenados[i]
            color = colores_activos[nombre_activo]

            bars = ax.bar(
                fechas_mpl,
                valores,
                width=width,
                bottom=bottom,
                color=color,
                label=nombre_activo,
                edgecolor='none',
                align='center'
            )

            for j, rect in enumerate(bars):

                height = rect.get_height()

                if height > 4.0:

                    if height < 10.5 and n_activos > 10:
                        fontsize_val = 7
                    else:
                        fontsize_val = 8

                    cy = rect.get_y() + rect.get_height() / 2

                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        cy,
                        f"{height:.1f}",
                        ha='center',
                        va='center',
                        color='white',
                        fontsize=fontsize_val,
                        rotation=90,
                        weight='bold'
                    )

            bottom += valores

        ultima_fecha_num = fechas_mpl[-1]
        x_pos_labels = ultima_fecha_num + (width * 0.6)

        step_y = 100.0 / n_activos
        current_y = step_y / 2

        for nombre in nombres_ordenados:

            color_texto = colores_activos[nombre]

            ax.text(
                x_pos_labels,
                current_y,
                nombre,
                va='center',
                ha='left',
                fontsize=11,
                weight='bold',
                color=color_texto
            )

            current_y += step_y

        titulo_estr = {
            'mopso': 'MOPSO',
            'nsga2': 'NSGA-II'
        }.get(estrategia, estrategia.capitalize())

        if titulo_general:

            plt.text(
                x=0.0,
                y=1.05,
                s=titulo_general,
                fontsize=18,
                weight='bold',
                transform=ax.transAxes
            )

        plt.text(
            x=0.0,
            y=1.02,
            s=f"Estrategia: {titulo_estr}",
            fontsize=12,
            color='gray',
            transform=ax.transAxes
        )

        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.yaxis.set_minor_locator(MultipleLocator(10))

        ax.set_ylabel("Participación (%)", labelpad=10, fontsize=12)

        ax.set_xticks(fechas_mpl)

        if n_ventanas <= 50:

            etiquetas_fechas = [
                fecha.strftime('%d-%m-%y')
                for fecha in fechas_seleccionadas
            ]

            ax.set_xticklabels(etiquetas_fechas, rotation=45, ha='right', fontsize=9)

        else:

            intervalo = max(1, n_ventanas // 20)

            etiquetas_fechas = []

            for i, fecha in enumerate(fechas_seleccionadas):

                if i % intervalo == 0:

                    if intervalo >= 4:
                        etiquetas_fechas.append(fecha.strftime('%d-%m-%y'))
                    else:
                        etiquetas_fechas.append(fecha.strftime('%d-%m'))

                else:
                    etiquetas_fechas.append('')

            ax.set_xticklabels(etiquetas_fechas, rotation=45, ha='right', fontsize=9)

        fecha_inicio_mpl = fechas_mpl[0] - (width * 0.7)
        espacio_derecha_dias = width * 8
        fecha_fin_mpl = fechas_mpl[-1] + espacio_derecha_dias

        ax.set_xlim(fecha_inicio_mpl, fecha_fin_mpl)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for fecha_mpl in fechas_mpl:
            ax.axvline(x=fecha_mpl, color='gray', alpha=0.15, linewidth=0.5)

        plt.tight_layout()

        # GUARDAR FIGURA
        nombre_archivo = f"figures/pesos_{estrategia}.png"
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')

        plt.close()

def graficar_frentes_pareto(frente_mopso, frente_nsga, path="figures/frentes_pareto.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    if len(frente_mopso) == 0 or len(frente_nsga) == 0:
        return

    # convertir -Sharpe → Sharpe
    mopso = frente_mopso.copy()
    nsga = frente_nsga.copy()

    mopso[:,0] = -mopso[:,0]
    nsga[:,0] = -nsga[:,0]

    def utopia(front):

        sharpe = front[:,0]
        kurt = front[:,1]

        # normalización: max Sharpe, min kurtosis
        s_norm = (sharpe - sharpe.min()) / (sharpe.max() - sharpe.min() + 1e-9)
        k_norm = (kurt.max() - kurt) / (kurt.max() - kurt.min() + 1e-9)

        score = s_norm + k_norm

        idx = np.argmax(score)

        return front[idx]

    u_m = utopia(mopso)
    u_n = utopia(nsga)

    # ordenar por curtosis
    mopso_sorted = mopso[np.argsort(mopso[:,1])]
    nsga_sorted = nsga[np.argsort(nsga[:,1])]

    plt.figure(figsize=(8,6))

    # scatter
    plt.scatter(
        mopso_sorted[:,1],
        mopso_sorted[:,0],
        alpha=0.7,
        s=40,
        label="MOPSO"
    )

    plt.scatter(
        nsga_sorted[:,1],
        nsga_sorted[:,0],
        alpha=0.7,
        s=40,
        label="NSGA-II"
    )

    # línea del frente
    plt.plot(mopso_sorted[:,1], mopso_sorted[:,0], linewidth=2)
    plt.plot(nsga_sorted[:,1], nsga_sorted[:,0], linewidth=2)

    # puntos utopía
    plt.scatter(
        u_m[1],
        u_m[0],
        marker="*",
        s=350,
        edgecolor="black",
        linewidth=1.2,
        zorder=10,
        label="Utopía MOPSO"
    )

    plt.scatter(
        u_n[1],
        u_n[0],
        marker="*",
        s=350,
        edgecolor="black",
        linewidth=1.2,
        zorder=10,
        label="Utopía NSGA-II"
    )

    plt.xlabel("Curtosis")
    plt.ylabel("Sharpe")

    plt.title("Frentes de Pareto")

    plt.grid(alpha=0.25)

    plt.legend()

    plt.tight_layout()

    plt.savefig(path, dpi=300)

    plt.close()


def graficar_dominancia_frentes(matriz_frentes):

    import matplotlib.pyplot as plt
    import numpy as np

    mopso = matriz_frentes["mopso"]
    nsga = matriz_frentes["nsga2"]

    dom_mopso = 0
    dom_nsga = 0
    empates = 0

    total = 0

    for c in range(len(mopso)):

        for f in range(len(mopso[c])):

            front_m = mopso[c][f]
            front_n = nsga[c][f]

            if front_m.size == 0 or front_n.size == 0:
                continue

            total += 1

            m_dom = 0
            n_dom = 0

            for p in front_m:
                for q in front_n:

                    if np.all(p <= q) and np.any(p < q):
                        m_dom += 1

                    if np.all(q <= p) and np.any(q < p):
                        n_dom += 1

            if m_dom > n_dom:
                dom_mopso += 1
            elif n_dom > m_dom:
                dom_nsga += 1
            else:
                empates += 1

    labels = ["MOPSO domina", "NSGA-II domina", "Empate"]
    values = [dom_mopso, dom_nsga, empates]

    plt.figure(figsize=(7,5))

    bars = plt.bar(labels, values)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height + 0.5,
                 f"{height}",
                 ha='center')

    plt.title("Dominancia entre frentes")

    plt.ylabel("Número de ventanas")

    plt.grid(axis='y', alpha=0.25)

    plt.tight_layout()

    plt.savefig("figures/dominancia_frentes.png", dpi=300)

    plt.close()



def graficar_frente_promedio(matriz_frentes):

    import matplotlib.pyplot as plt
    import numpy as np

    mopso_pts = []
    nsga_pts = []

    for c in range(len(matriz_frentes["mopso"])):

        for f in matriz_frentes["mopso"][c]:
            if f.size > 0:
                mopso_pts.append(f)

        for f in matriz_frentes["nsga2"][c]:
            if f.size > 0:
                nsga_pts.append(f)

    mopso_pts = np.vstack(mopso_pts)
    nsga_pts = np.vstack(nsga_pts)

    plt.figure(figsize=(9,6))

    # hexbin para evitar saturación
    plt.hexbin(mopso_pts[:,1], mopso_pts[:,0],
               gridsize=45,
               bins='log',
               alpha=0.6)

    plt.scatter(nsga_pts[:,1], nsga_pts[:,0],
                alpha=0.15,
                s=8,
                label="NSGA-II")

    plt.yscale("log")

    plt.xlabel("Kurtosis")
    plt.ylabel("-Sharpe (escala log)")

    plt.title("Distribución global de frentes")

    plt.grid(alpha=0.2)

    plt.legend()

    plt.tight_layout()

    plt.savefig("figures/frentes_globales.png", dpi=300)

    plt.close()