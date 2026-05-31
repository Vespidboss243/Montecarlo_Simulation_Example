import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure strict adherence to page layout and aesthetic requirements
st.set_page_config(
    page_title="Simulación Montecarlo: Reparaciones Locativas",
    layout="wide",
    initial_sidebar_state="expanded"
)

def validate_triangular_params(optimistic, most_likely, pessimistic):
    """
    Validates and enforces the mathematical constraints for a triangular distribution.
    Returns sorted values to prevent numpy ValueError during simulation.
    """
    params = sorted([optimistic, most_likely, pessimistic])
    return params[0], params[1], params[2]

def run_simulation():
    st.title("Simulación Montecarlo: Reparaciones Locativas")
    st.markdown("### Edificio Federico Drews - Pereira")
    st.markdown("Adaptación de propiedad de 2.300 m² para la Universidad Cooperativa de Colombia (UCC).")
    st.markdown("---")

    # ==========================================
    # SIDEBAR CONTROLS
    # ==========================================
    st.sidebar.title("Parámetros de Simulación")
    st.sidebar.markdown("Defina la distribución (en días) para cada actividad del proyecto.")

    # Actividad 1
    st.sidebar.markdown("#### 1. Normativa, espacios y necesidades")
    p1_opt = st.sidebar.slider("Actividad 1: Optimista", min_value=5, max_value=100, value=20, key="p1_opt")
    p1_ml = st.sidebar.slider("Actividad 1: Más Probable", min_value=5, max_value=100, value=45, key="p1_ml")
    p1_pess = st.sidebar.slider("Actividad 1: Pesimista", min_value=5, max_value=100, value=52, key="p1_pess")

    # Actividad 2
    st.sidebar.markdown("#### 2. Diseños, presupuesto y programación")
    p2_opt = st.sidebar.slider("Actividad 2: Optimista", min_value=30, max_value=200, value=60, key="p2_opt")
    p2_ml = st.sidebar.slider("Actividad 2: Más Probable", min_value=30, max_value=200, value=99, key="p2_ml")
    p2_pess = st.sidebar.slider("Actividad 2: Pesimista", min_value=30, max_value=200, value=114, key="p2_pess")

    # Actividad 3
    st.sidebar.markdown("#### 3. Pliegos, invitaciones y adjudicaciones")
    p3_opt = st.sidebar.slider("Actividad 3: Optimista", min_value=5, max_value=60, value=21, key="p3_opt")
    p3_ml = st.sidebar.slider("Actividad 3: Más Probable", min_value=5, max_value=60, value=25, key="p3_ml")
    p3_pess = st.sidebar.slider("Actividad 3: Pesimista", min_value=5, max_value=60, value=29, key="p3_pess")

    # Actividad 4
    st.sidebar.markdown("#### 4. Ejecución de obra civil y redes")
    p4_opt = st.sidebar.slider("Actividad 4: Optimista", min_value=50, max_value=300, value=90, key="p4_opt")
    p4_ml = st.sidebar.slider("Actividad 4: Más Probable", min_value=50, max_value=300, value=120, key="p4_ml")
    p4_pess = st.sidebar.slider("Actividad 4: Pesimista", min_value=50, max_value=300, value=138, key="p4_pess")

    # Actividad 5
    st.sidebar.markdown("#### 5. Instalación de mobiliario")
    p5_opt = st.sidebar.slider("Actividad 5: Optimista", min_value=1, max_value=40, value=8, key="p5_opt")
    p5_ml = st.sidebar.slider("Actividad 5: Más Probable", min_value=1, max_value=40, value=10, key="p5_ml")
    p5_pess = st.sidebar.slider("Actividad 5: Pesimista", min_value=1, max_value=40, value=12, key="p5_pess")

    # Actividad 6
    st.sidebar.markdown("#### 6. Traslado a las nuevas instalaciones")
    p6_opt = st.sidebar.slider("Actividad 6: Optimista", min_value=1, max_value=40, value=8, key="p6_opt")
    p6_ml = st.sidebar.slider("Actividad 6: Más Probable", min_value=1, max_value=40, value=10, key="p6_ml")
    p6_pess = st.sidebar.slider("Actividad 6: Pesimista", min_value=1, max_value=40, value=12, key="p6_pess")

    # Actividad 7
    st.sidebar.markdown("#### 7. Liquidación de los contratos")
    p7_opt = st.sidebar.slider("Actividad 7: Optimista", min_value=10, max_value=100, value=25, key="p7_opt")
    p7_ml = st.sidebar.slider("Actividad 7: Más Probable", min_value=10, max_value=100, value=40, key="p7_ml")
    p7_pess = st.sidebar.slider("Actividad 7: Pesimista", min_value=10, max_value=100, value=46, key="p7_pess")

    # Actividad 8
    st.sidebar.markdown("#### 8. Cierre documental del proyecto")
    p8_opt = st.sidebar.slider("Actividad 8: Optimista", min_value=1, max_value=50, value=8, key="p8_opt")
    p8_ml = st.sidebar.slider("Actividad 8: Más Probable", min_value=1, max_value=50, value=12, key="p8_ml")
    p8_pess = st.sidebar.slider("Actividad 8: Pesimista", min_value=1, max_value=50, value=14, key="p8_pess")

    st.sidebar.markdown("---")
    
    # Global Simulation Settings
    iterations = st.sidebar.slider(
        "Iteraciones de Simulación", 
        min_value=1000, 
        max_value=50000, 
        value=10000, 
        step=1000
    )

    execute_sim = st.sidebar.button("Ejecutar Simulación", use_container_width=True)

    # ==========================================
    # DEFAULT STATE BEFORE EXECUTION
    # ==========================================
    if not execute_sim:
        st.info("Por favor, ajuste los parámetros de las actividades en el panel lateral y haga clic en 'Ejecutar Simulación' para generar el modelo de riesgo.")
        return

    # ==========================================
    # SIMULATION EXECUTION & DATA PREPARATION
    # ==========================================
    
    # Validate and enforce parameter constraints mathematically
    p1_o, p1_m, p1_p = validate_triangular_params(p1_opt, p1_ml, p1_pess)
    p2_o, p2_m, p2_p = validate_triangular_params(p2_opt, p2_ml, p2_pess)
    p3_o, p3_m, p3_p = validate_triangular_params(p3_opt, p3_ml, p3_pess)
    p4_o, p4_m, p4_p = validate_triangular_params(p4_opt, p4_ml, p4_pess)
    p5_o, p5_m, p5_p = validate_triangular_params(p5_opt, p5_ml, p5_pess)
    p6_o, p6_m, p6_p = validate_triangular_params(p6_opt, p6_ml, p6_pess)
    p7_o, p7_m, p7_p = validate_triangular_params(p7_opt, p7_ml, p7_pess)
    p8_o, p8_m, p8_p = validate_triangular_params(p8_opt, p8_ml, p8_pess)

    # Execute numpy generators for the entire set and round to discrete whole days
    sim_p1 = np.round(np.random.triangular(p1_o, p1_m, p1_p, iterations)).astype(int)
    sim_p2 = np.round(np.random.triangular(p2_o, p2_m, p2_p, iterations)).astype(int)
    sim_p3 = np.round(np.random.triangular(p3_o, p3_m, p3_p, iterations)).astype(int)
    sim_p4 = np.round(np.random.triangular(p4_o, p4_m, p4_p, iterations)).astype(int)
    sim_p5 = np.round(np.random.triangular(p5_o, p5_m, p5_p, iterations)).astype(int)
    sim_p6 = np.round(np.random.triangular(p6_o, p6_m, p6_p, iterations)).astype(int)
    sim_p7 = np.round(np.random.triangular(p7_o, p7_m, p7_p, iterations)).astype(int)
    sim_p8 = np.round(np.random.triangular(p8_o, p8_m, p8_p, iterations)).astype(int)

    # Calculate cumulative durations for the spaghetti chart
    cum_p1 = sim_p1
    cum_p2 = cum_p1 + sim_p2
    cum_p3 = cum_p2 + sim_p3
    cum_p4 = cum_p3 + sim_p4
    cum_p5 = cum_p4 + sim_p5
    cum_p6 = cum_p5 + sim_p6
    cum_p7 = cum_p6 + sim_p7
    cum_p8 = cum_p7 + sim_p8
    
    total_durations = cum_p8

    # Calculate Key Metrics and ensure they are integers
    mean_duration = int(np.round(np.mean(total_durations)))
    median_duration = int(np.round(np.percentile(total_durations, 50)))
    p90_duration = int(np.round(np.percentile(total_durations, 90)))

    # ==========================================
    # METRICS DISPLAY
    # ==========================================
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Finalización Esperada (Promedio)", value=f"{mean_duration} Días")
    with col2:
        st.metric(label="Percentil 50 (Mediana)", value=f"{median_duration} Días")
    with col3:
        st.metric(label="Percentil 90 (Objetivo de Riesgo)", value=f"{p90_duration} Días")

    # ==========================================
    # VISUALIZATIONS
    # ==========================================
    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    # Common styling configurations
    corporate_layout = dict(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    
    # Corporate highlight color (Deep Midnight Blue)
    corporate_highlight = "#003366" 

    # 1. Frequency Histogram
    with chart_col1:
        fig_hist = px.histogram(
            x=total_durations,
            nbins=100,
            color_discrete_sequence=['#4682B4'],
            labels={'x': 'Duración Total del Proyecto (Días)', 'y': 'Frecuencia'},
            title="Distribución de Frecuencia de la Duración del Proyecto"
        )
        fig_hist.update_layout(**corporate_layout)
        fig_hist.add_vline(x=mean_duration, line_dash="dash", line_color=corporate_highlight, annotation_text="Promedio")
        st.plotly_chart(fig_hist, use_container_width=True, theme="streamlit")

    # 2. Cumulative Distribution Function (CDF)
    with chart_col2:
        fig_cdf = px.ecdf(
            x=total_durations,
            color_discrete_sequence=['#4682B4'],
            labels={'x': 'Duración Total del Proyecto (Días)', 'y': 'Probabilidad'},
            title="Probabilidad Acumulada de Finalización"
        )
        fig_cdf.update_layout(**corporate_layout)
        fig_cdf.add_hline(y=0.90, line_dash="dot", line_color=corporate_highlight, annotation_text="90% de Confianza")
        fig_cdf.add_vline(x=p90_duration, line_dash="dot", line_color=corporate_highlight)
        st.plotly_chart(fig_cdf, use_container_width=True, theme="streamlit")

    # 3. Spaghetti Convergence Chart
    st.markdown("---")
    st.markdown("#### Progresión Acumulada por Actividades (Varianza de Rutas de Simulación)")
    
    fig_spag = go.Figure()
    
    # Fully expanded professional labels
    phases_x = [
        'Actividad 1: Revisión de normativa, espacios y necesidades',
        'Actividad 2: Diseños, presupuesto y programación',
        'Actividad 3: Pliegos, invitaciones y adjudicaciones',
        'Actividad 4: Ejecución de obra civil y redes',
        'Actividad 5: Instalación de mobiliario',
        'Actividad 6: Traslado a las nuevas instalaciones',
        'Actividad 7: Liquidación de los contratos',
        'Actividad 8: Cierre documental del proyecto'
    ]
    
    # Background lines (subset of 100 paths with increased opacity for visibility)
    num_paths = min(100, iterations)
    for j in range(num_paths):
        fig_spag.add_trace(go.Scatter(
            x=phases_x, 
            y=[cum_p1[j], cum_p2[j], cum_p3[j], cum_p4[j], cum_p5[j], cum_p6[j], cum_p7[j], cum_p8[j]],
            mode='lines',
            line=dict(color='rgba(112, 128, 144, 0.4)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
    # Foreground Mean Line (rounded to discrete days)
    mean_path_y = [
        int(np.round(np.mean(cum_p1))), 
        int(np.round(np.mean(cum_p2))), 
        int(np.round(np.mean(cum_p3))), 
        int(np.round(np.mean(cum_p4))),
        int(np.round(np.mean(cum_p5))),
        int(np.round(np.mean(cum_p6))),
        int(np.round(np.mean(cum_p7))),
        int(np.round(np.mean(cum_p8)))
    ]
    fig_spag.add_trace(go.Scatter(
        x=phases_x, 
        y=mean_path_y,
        mode='lines+markers',
        line=dict(color='#DC143C', width=4),
        name='Ruta Media Esperada'
    ))
    
    # Dummy trace for legend entry of background paths
    fig_spag.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(112, 128, 144, 0.4)', width=1),
        name='Rutas de Simulación (Varianza)'
    ))

    fig_spag.update_layout(
        **corporate_layout,
        yaxis_title="Duración Acumulada (Días)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_spag, use_container_width=True, theme="streamlit")

    # ==========================================
    # FINAL RAW DATA EXPORT
    # ==========================================
    with st.expander("Ver Resumen de Datos Crudos de la Simulación"):
        summary_df = pd.DataFrame(total_durations, columns=['Duración Total (Días)'])
        # Also round the description outputs for cleaner reading
        st.dataframe(np.round(summary_df.describe()).astype(int).T, use_container_width=True)

if __name__ == "__main__":
    run_simulation()