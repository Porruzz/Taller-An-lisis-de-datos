"""
Generación de gráficas técnicas de alta resolución para el análisis de
congestión LAN y la evaluación de alertas tempranas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from ..config import PALETA, TASA_LIMITE_MBPS, UMBRAL_ALERTA_ESTANDAR


def _configurar_estilo() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.2,
            "axes.edgecolor": "#CBD5E0",
            "axes.linewidth": 0.8,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#CBD5E0",
            "legend.fontsize": 7.8,
        }
    )


def graficar_serie_temporal_corrida(
    df_ventanas_corrida: pd.DataFrame,
    df_eval_corrida: pd.DataFrame,
    run_id: str,
    tau_base_mbps: float,
    umbral_reg: float,
    ruta_salida: Path,
    col_alerta_reg: str = "alerta_regresion_opt",
    titulo_extra: str = "",
) -> Path:
    """
    Gráfica temporal de tres paneles alineados para una corrida completa:
      1) Tasa capturada r_m (Mbit/s), límite TBF (10 Mbit/s) y umbral tau_base.
      2) Descartes reales d_{k+1} vs pronosticados max(0, y_hat_k), ubicados en
         la ventana k+1 a la que se refiere cada pronóstico, marcando inicios.
      3) Línea de tiempo de congestión observada y alertas (Regresión θ=1,
         Regresión calibrada θ* y Regla Base r_k >= τ*), ubicadas en k+1.
    """
    _configurar_estilo()
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    vent = df_ventanas_corrida[df_ventanas_corrida["valida"]].sort_values("ventana_idx").copy()
    ev = df_eval_corrida.sort_values("ventana_idx_k1").copy()

    t_vent = vent["t_rel_fin_s"].to_numpy(dtype=np.float64)
    r_vent = vent["r_k"].to_numpy(dtype=np.float64)

    # Cada pronóstico emitido al cierre de k se ubica en la ventana k+1 a la que se refiere
    t_k1 = ev["t_rel_fin_k1_s"].to_numpy(dtype=np.float64)
    y_real = ev["y_k"].to_numpy(dtype=np.float64)
    y_pred = ev["y_hat_recortado"].to_numpy(dtype=np.float64)
    cong_real = ev["evento_congestion_k1"].to_numpy(dtype=int)
    alerta_std = ev["alerta_regresion_std"].to_numpy(dtype=int) if "alerta_regresion_std" in ev.columns else ev[col_alerta_reg].to_numpy(dtype=int)
    alerta_opt = ev[col_alerta_reg].to_numpy(dtype=int)
    alerta_base = ev["alerta_base"].to_numpy(dtype=int)
    es_inicio = ev["es_inicio_observado"].to_numpy(dtype=int) == 1

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.4, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.35, 1.05], "hspace": 0.12},
    )

    # Sombreado suave de fases críticas (solo interpretativo)
    for ax in axes:
        for tipo, color_sombra in [("salto_brusco", "#FED7D7"), ("subumbral", "#FEFCBF")]:
            mascara = vent["tipo_transicion"] == tipo
            if mascara.any():
                idx_true = np.where(mascara.to_numpy())[0]
                cortes = np.where(np.diff(idx_true) > 1)[0] + 1
                for bloque in np.split(idx_true, cortes):
                    x0 = float(vent.iloc[bloque[0]]["t_rel_inicio_s"])
                    x1 = float(vent.iloc[bloque[-1]]["t_rel_fin_s"])
                    ax.axvspan(x0, x1, color=color_sombra, alpha=0.28, zorder=0)

    # ---------------- Panel 1: Tasa capturada ----------------
    ax1 = axes[0]
    ax1.step(t_vent, r_vent, where="post", color=PALETA["azul_oscuro"], linewidth=1.6, label="Tasa capturada $r_m$ (PCAP)")
    ax1.axhline(TASA_LIMITE_MBPS, color=PALETA["rojo"], linestyle="--", linewidth=1.1, label="Límite TBF (10 Mbit/s)")
    ax1.axhline(
        tau_base_mbps,
        color=PALETA["morado"],
        linestyle=":",
        linewidth=1.4,
        label=f"Umbral regla base ($\\tau^* = {tau_base_mbps:.2f}$ Mbit/s)",
    )
    ax1.set_ylabel("Tasa (Mbit/s)")
    ax1.set_ylim(0, max(12.2, float(np.nanmax(r_vent)) * 1.18))
    ax1.grid(True, alpha=0.7)
    ax1.legend(loc="upper right", ncol=3)
    sub = f" — {titulo_extra}" if titulo_extra else ""
    ax1.set_title(f"Corrida {run_id.upper()}{sub}: Serie temporal alineada en la ventana pronosticada ($k+1$)")

    # ---------------- Panel 2: Descartes reales vs pronosticados ----------------
    ax2 = axes[1]
    ax2.fill_between(
        t_k1,
        0,
        y_real,
        step="mid",
        color=PALETA["azul_medio"],
        alpha=0.22,
        label="Descartes reales $d_{k+1}$",
    )
    ax2.plot(
        t_k1,
        y_real,
        color=PALETA["azul_medio"],
        linewidth=1.5,
        drawstyle="steps-mid",
    )
    ax2.plot(
        t_k1,
        y_pred,
        color=PALETA["naranja"],
        linewidth=1.65,
        linestyle="-",
        marker="o",
        markersize=2.4,
        label="Pronóstico $\\max(0, \\hat{y}_k)$",
    )
    ax2.axhline(
        umbral_reg,
        color=PALETA["rojo"],
        linestyle="-.",
        linewidth=1.0,
        alpha=0.85,
        label=f"Umbral calibrado ($\\theta^*={umbral_reg:.1f}$)",
    )
    ax2.axhline(
        UMBRAL_ALERTA_ESTANDAR,
        color=PALETA["gris"],
        linestyle=":",
        linewidth=1.0,
        alpha=0.85,
        label=f"Umbral estándar ($\\theta={UMBRAL_ALERTA_ESTANDAR:g}$)",
    )

    # Marcar inicios observados (clasificados según umbral estándar θ=1 para distinguir rampa vs salto)
    inicios_ant_t = t_k1[es_inicio & (alerta_std == 1)]
    inicios_ant_y = y_real[es_inicio & (alerta_std == 1)]
    inicios_omi_t = t_k1[es_inicio & (alerta_std == 0)]
    inicios_omi_y = y_real[es_inicio & (alerta_std == 0)]

    for t_val in inicios_ant_t:
        for ax in axes:
            ax.axvline(t_val, color=PALETA["verde"], linestyle="--", linewidth=1.1, alpha=0.75, zorder=2)
    for t_val in inicios_omi_t:
        for ax in axes:
            ax.axvline(t_val, color=PALETA["rojo"], linestyle="--", linewidth=1.1, alpha=0.75, zorder=2)

    if len(inicios_ant_t) > 0:
        ax2.scatter(
            inicios_ant_t,
            inicios_ant_y,
            color=PALETA["verde"],
            edgecolor="black",
            s=58,
            marker="^",
            zorder=5,
            label="Inicio observado (anticipado con $\\theta=1$)",
        )
    if len(inicios_omi_t) > 0:
        ax2.scatter(
            inicios_omi_t,
            inicios_omi_y,
            color=PALETA["rojo"],
            edgecolor="black",
            s=62,
            marker="X",
            zorder=5,
            label="Inicio observado (omitido)",
        )

    ax2.set_ylabel("Descartes (paquetes/ventana)")
    ymax2 = max(10.0, float(max(np.nanmax(y_real), np.nanmax(y_pred))) * 1.24)
    ax2.set_ylim(-ymax2 * 0.03, ymax2)
    ax2.grid(True, alpha=0.7)
    ax2.legend(loc="upper right", ncol=3)

    # ---------------- Panel 3: Alertas y congestión observada ----------------
    ax3 = axes[2]
    ancho = float(np.nanmedian(ev["delta_k1_s"])) * 0.88 if not ev.empty else 0.9

    mask_real = cong_real == 1
    mask_std_vp = (alerta_std == 1) & (cong_real == 1)
    mask_std_fp = (alerta_std == 1) & (cong_real == 0)
    mask_opt = alerta_opt == 1
    mask_base = alerta_base == 1

    if mask_real.any():
        ax3.bar(t_k1[mask_real], 0.56, bottom=3.10, width=ancho, color=PALETA["azul_medio"], alpha=0.88, label="Congestión real ($d_{k+1}\\geq 1$)")
    if mask_std_vp.any():
        ax3.bar(t_k1[mask_std_vp], 0.56, bottom=2.15, width=ancho, color=PALETA["verde"], alpha=0.82, label="Regresión $\\theta=1$ (acierto)")
    if mask_std_fp.any():
        ax3.bar(t_k1[mask_std_fp], 0.56, bottom=2.15, width=ancho, color=PALETA["rojo"], alpha=0.55, label="Regresión $\\theta=1$ (falsa alarma)")
    if mask_opt.any():
        ax3.bar(t_k1[mask_opt], 0.56, bottom=1.20, width=ancho, color=PALETA["naranja"], alpha=0.90, label=f"Regresión $\\theta^*={umbral_reg:.1f}$")
    if mask_base.any():
        ax3.bar(t_k1[mask_base], 0.56, bottom=0.25, width=ancho, color=PALETA["morado"], alpha=0.82, label=f"Regla base ($r_k\\geq {tau_base_mbps:.2f}$)")

    ax3.set_yticks([0.53, 1.48, 2.43, 3.38])
    ax3.set_yticklabels(["Regla base", "Regresión $\\theta^*$", "Regresión $\\theta=1$", "Real ($d_{k+1}\\geq 1$)"])
    ax3.set_ylim(0.0, 4.35)
    ax3.set_xlabel("Tiempo transcurrido en la corrida (s) — eje referido a la ventana objetivo $k+1$")
    ax3.grid(True, axis="x", alpha=0.7)

    handles, _ = ax3.get_legend_handles_labels()
    handles.extend(
        [
            Patch(facecolor="#FED7D7", alpha=0.50, label="Fase: salto brusco"),
            Patch(facecolor="#FEFCBF", alpha=0.60, label="Fase: estrés sub-umbral"),
        ]
    )
    ax3.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, fontsize=7.6)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.14)
    fig.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


def graficar_calibracion_validacion(
    tabla_reg: pd.DataFrame,
    umbral_reg_std: float,
    umbral_reg_opt: float,
    tabla_base: pd.DataFrame,
    tau_opt: float,
    ruta_salida: Path,
) -> Path:
    """Grafica las curvas de calibración de umbrales en validación (c04)."""
    _configurar_estilo()
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.3))

    ax1.plot(tabla_reg["umbral"], tabla_reg["precision"], color=PALETA["azul_medio"], linewidth=1.8, label="Precisión")
    ax1.plot(tabla_reg["umbral"], tabla_reg["sensibilidad"], color=PALETA["verde"], linewidth=1.8, label="Sensibilidad")
    ax1.plot(tabla_reg["umbral"], tabla_reg["f1"], color=PALETA["naranja"], linewidth=2.1, label="F1-Score")
    ax1.axvline(umbral_reg_std, color=PALETA["gris"], linestyle="--", linewidth=1.2, label=f"Estándar ($\\theta={umbral_reg_std:g}$)")
    ax1.axvline(umbral_reg_opt, color=PALETA["rojo"], linestyle=":", linewidth=1.5, label=f"Óptimo val. ($\\theta^*={umbral_reg_opt:.1f}$)")
    ax1.set_title("Calibración de umbral — Regresión Lineal (c04)")
    ax1.set_xlabel("Umbral sobre $\\hat{y}_k$ (paquetes/ventana)")
    ax1.set_ylabel("Valor de la métrica")
    ax1.set_ylim(-0.04, 1.08)
    ax1.grid(True, alpha=0.7)
    ax1.legend(loc="lower left", fontsize=8.0)

    ax2.plot(tabla_base["tau_mbps"], tabla_base["precision"], color=PALETA["azul_medio"], linewidth=1.8, label="Precisión")
    ax2.plot(tabla_base["tau_mbps"], tabla_base["sensibilidad"], color=PALETA["verde"], linewidth=1.8, label="Sensibilidad")
    ax2.plot(tabla_base["tau_mbps"], tabla_base["f1"], color=PALETA["naranja"], linewidth=2.1, label="F1-Score")
    ax2.axvline(tau_opt, color=PALETA["rojo"], linestyle=":", linewidth=1.6, label=f"Óptimo val. ($\\tau^*={tau_opt:.2f}$ Mbit/s)")
    ax2.set_title("Calibración de umbral — Regla Base $r_k \\geq \\tau$ (c04)")
    ax2.set_xlabel("Umbral de tasa $\\tau$ (Mbit/s)")
    ax2.set_ylabel("Valor de la métrica")
    ax2.set_ylim(-0.04, 1.08)
    ax2.grid(True, alpha=0.7)
    ax2.legend(loc="lower left", fontsize=8.0)

    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


def graficar_matrices_confusion(
    metricas_modelos: List[Tuple[str, Dict[str, float]]],
    ruta_salida: Path,
) -> Path:
    """Dibuja matrices de confusión lado a lado para los modelos evaluados en prueba."""
    _configurar_estilo()
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    n = len(metricas_modelos)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.1))
    if n == 1:
        axes = [axes]

    for ax, (nombre, m) in zip(axes, metricas_modelos):
        mat = np.array([[m["VN"], m["FP"]], [m["FN"], m["VP"]]], dtype=int)
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(1, int(mat.max())))
        etiquetas = [["VN", "FP"], ["FN", "VP"]]
        total = max(1, int(mat.sum()))
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                pct = 100.0 * val / total
                color_txt = "white" if val > mat.max() * 0.55 else "#1A202C"
                ax.text(
                    j,
                    i,
                    f"{etiquetas[i][j]}\n{val}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=color_txt,
                    fontsize=9.8,
                    fontweight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Sin alerta (0)", "Alerta (1)"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Sin descarte (0)", "Congestión (1)"])
        ax.set_xlabel("Predicción emitida en $t_k$")
        ax.set_ylabel("Real en ventana $k+1$")
        ax.set_title(
            f"{nombre}\nPrec={m['precision']:.3f} | Sens={m['sensibilidad']:.3f} | F1={m['f1']:.3f}"
        )

    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


def graficar_inicios_y_persistencia(
    resumen_std: Dict[str, float],
    resumen_opt: Dict[str, float],
    resumen_base: Dict[str, float],
    df_rs_std: pd.DataFrame,
    df_rs_opt: pd.DataFrame,
    df_rs_base: pd.DataFrame,
    ruta_salida: Path,
) -> Path:
    """
    Compara visualmente los tres criterios evaluados en prueba (c05–c06):
      - Izquierda: tasa de detección en inicios observados vs persistencia.
      - Derecha: fracción de inicios anticipados en rampas vs saltos bruscos.
    """
    _configurar_estilo()
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))

    categorias = ["Inicio observado\n($d_k=0, d_{k+1}>0$)", "Persistencia\n($d_k>0, d_{k+1}>0$)"]
    vals_std = [
        100.0 * np.nan_to_num(resumen_std["frac_inicios_anticipados"]),
        100.0 * np.nan_to_num(resumen_std["sensibilidad_persistencia"]),
    ]
    vals_opt = [
        100.0 * np.nan_to_num(resumen_opt["frac_inicios_anticipados"]),
        100.0 * np.nan_to_num(resumen_opt["sensibilidad_persistencia"]),
    ]
    vals_base = [
        100.0 * np.nan_to_num(resumen_base["frac_inicios_anticipados"]),
        100.0 * np.nan_to_num(resumen_base["sensibilidad_persistencia"]),
    ]

    x = np.arange(len(categorias))
    w = 0.25
    b1 = ax1.bar(x - w, vals_std, width=w, color=PALETA["verde"], label="Regresión ($\\theta=1$)")
    b2 = ax1.bar(x, vals_opt, width=w, color=PALETA["naranja"], label="Regresión ($\\theta^*$ val.)")
    b3 = ax1.bar(x + w, vals_base, width=w, color=PALETA["morado"], label="Regla Base ($r_k\\geq \\tau^*$)")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categorias)
    ax1.set_ylabel("Detección / Anticipación (%)")
    ax1.set_ylim(0, 118)
    ax1.set_title("Inicio de episodio vs. persistencia en prueba (c05–c06)")
    ax1.grid(True, axis="y", alpha=0.7)
    ax1.legend(loc="upper left", fontsize=7.8)

    tipos_orden = ["rampa", "salto_brusco"]
    etiquetas_tipo = {"rampa": "Ascenso gradual\n(rampa)", "salto_brusco": "Salto brusco\n(sin rampa previa)"}

    def _val(df_rs: pd.DataFrame, tipo: str) -> float:
        sub = df_rs[df_rs["tipo_transicion_k1"] == tipo]
        return 0.0 if sub.empty else 100.0 * float(sub["frac_anticipados"].iloc[0])

    def _cnt(df_rs: pd.DataFrame, tipo: str) -> str:
        sub = df_rs[df_rs["tipo_transicion_k1"] == tipo]
        return "0/0" if sub.empty else f"{int(sub['inicios_anticipados'].iloc[0])}/{int(sub['inicios_evaluables'].iloc[0])}"

    x2 = np.arange(len(tipos_orden))
    c1 = ax2.bar(x2 - w, [_val(df_rs_std, t) for t in tipos_orden], width=w, color=PALETA["verde"], label="Regresión ($\\theta=1$)")
    c2 = ax2.bar(x2, [_val(df_rs_opt, t) for t in tipos_orden], width=w, color=PALETA["naranja"], label="Regresión ($\\theta^*$ val.)")
    c3 = ax2.bar(x2 + w, [_val(df_rs_base, t) for t in tipos_orden], width=w, color=PALETA["morado"], label="Regla Base ($r_k\\geq \\tau^*$)")

    for bars, df_src in ((c1, df_rs_std), (c2, df_rs_opt), (c3, df_rs_base)):
        for bar, t in zip(bars, tipos_orden):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.0f}%\n({_cnt(df_src, t)})", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels([etiquetas_tipo[t] for t in tipos_orden])
    ax2.set_ylabel("Inicios anticipados (%)")
    ax2.set_ylim(0, 122)
    ax2.set_title("Anticipación de inicios: rampas vs. saltos bruscos")
    ax2.grid(True, axis="y", alpha=0.7)
    ax2.legend(loc="upper right", fontsize=7.8)

    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


def graficar_saturacion_traza(
    df_ventanas: pd.DataFrame,
    ruta_salida: Path,
) -> Path:
    """
    Muestra experimentalmente cuánta información conserva la captura de salida
    PCAP frente a la sobrecarga:
      - Izquierda: tasa capturada r_k vs tasa ofrecida por iperf3 (saturación en ~10 Mbit/s).
      - Derecha: descartes reales d_k vs tasa capturada r_k (no linealidad y techo).
    """
    _configurar_estilo()
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    validas = df_ventanas[df_ventanas["valida"] & (df_ventanas["tasa_ofrecida_mbps"] > 0)].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    sin_drop = validas[validas["d_k"] == 0]
    con_drop = validas[validas["d_k"] > 0]
    ax1.scatter(sin_drop["tasa_ofrecida_mbps"], sin_drop["r_k"], color=PALETA["azul_medio"], alpha=0.55, s=24, label="Ventanas sin descartes ($d_k = 0$)")
    ax1.scatter(con_drop["tasa_ofrecida_mbps"], con_drop["r_k"], color=PALETA["rojo"], alpha=0.65, s=28, marker="^", label="Ventanas con descartes ($d_k > 0$)")
    xs = np.linspace(0, max(17.5, float(validas["tasa_ofrecida_mbps"].max()) + 0.5), 100)
    ax1.plot(xs, np.minimum(TASA_LIMITE_MBPS, xs * (1242.0 / 1200.0)), color=PALETA["gris"], linestyle="--", linewidth=1.2, label="Techo teórico TBF (10 Mbit/s)")
    ax1.set_xlabel("Tasa UDP ofrecida por iperf3 (Mbit/s)")
    ax1.set_ylabel("Tasa capturada $r_k$ en PCAP (Mbit/s)")
    ax1.set_title("Saturación de la observación de salida bajo cola TBF")
    ax1.grid(True, alpha=0.7)
    ax1.legend(loc="lower right", fontsize=8.0)

    ax2.scatter(sin_drop["r_k"], sin_drop["d_k"], color=PALETA["azul_medio"], alpha=0.45, s=22, label="$d_k = 0$")
    ax2.scatter(con_drop["r_k"], con_drop["d_k"], color=PALETA["rojo"], alpha=0.65, s=28, marker="^", label="$d_k > 0$")
    ax2.axvline(TASA_LIMITE_MBPS, color=PALETA["gris"], linestyle="--", linewidth=1.1, label="Límite TBF (10 Mbit/s)")
    ax2.set_xlabel("Tasa capturada $r_k$ (Mbit/s)")
    ax2.set_ylabel("Descartes observados $d_k$ (paquetes/ventana)")
    ax2.set_title("Relación no lineal entre tasa de salida y descartes")
    ax2.grid(True, alpha=0.7)
    ax2.legend(loc="upper left", fontsize=8.0)

    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida
