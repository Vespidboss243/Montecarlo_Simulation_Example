import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Ensure strict adherence to page layout and aesthetic requirements
st.set_page_config(
    page_title="Construction Project Monte Carlo Simulation",
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
    st.title("Construction Project Risk Analysis")
    st.markdown("### Monte Carlo Simulation for Residential Property Development")
    st.markdown("---")

    # ==========================================
    # SIDEBAR CONTROLS
    # ==========================================
    st.sidebar.title("Simulation Parameters")
    st.sidebar.markdown("Define distribution (days) for each project phase.")

    # Phase 1: Architectural Design & Municipal Permitting
    st.sidebar.markdown("#### 1. Design & Permitting")
    p1_opt = st.sidebar.slider("Phase 1: Optimistic", min_value=10, max_value=60, value=30, key="p1_opt")
    p1_ml = st.sidebar.slider("Phase 1: Most Likely", min_value=20, max_value=120, value=60, key="p1_ml")
    p1_pess = st.sidebar.slider("Phase 1: Pessimistic", min_value=60, max_value=200, value=120, key="p1_pess")

    # Phase 2: Site Preparation & Foundation
    st.sidebar.markdown("#### 2. Site Prep & Foundation")
    p2_opt = st.sidebar.slider("Phase 2: Optimistic", min_value=7, max_value=30, value=14, key="p2_opt")
    p2_ml = st.sidebar.slider("Phase 2: Most Likely", min_value=14, max_value=60, value=30, key="p2_ml")
    p2_pess = st.sidebar.slider("Phase 2: Pessimistic", min_value=30, max_value=120, value=60, key="p2_pess")

    # Phase 3: Structural Build & Utilities Routing
    st.sidebar.markdown("#### 3. Structural & Utilities")
    p3_opt = st.sidebar.slider("Phase 3: Optimistic", min_value=30, max_value=90, value=60, key="p3_opt")
    p3_ml = st.sidebar.slider("Phase 3: Most Likely", min_value=60, max_value=150, value=90, key="p3_ml")
    p3_pess = st.sidebar.slider("Phase 3: Pessimistic", min_value=90, max_value=250, value=150, key="p3_pess")

    # Phase 4: Interior Finishes & Final Inspections
    st.sidebar.markdown("#### 4. Interiors & Inspections")
    p4_opt = st.sidebar.slider("Phase 4: Optimistic", min_value=20, max_value=60, value=45, key="p4_opt")
    p4_ml = st.sidebar.slider("Phase 4: Most Likely", min_value=45, max_value=120, value=75, key="p4_ml")
    p4_pess = st.sidebar.slider("Phase 4: Pessimistic", min_value=60, max_value=180, value=120, key="p4_pess")

    st.sidebar.markdown("---")
    
    # Global Simulation Settings
    iterations = st.sidebar.slider(
        "Simulation Iterations", 
        min_value=1000, 
        max_value=50000, 
        value=10000, 
        step=1000
    )

    execute_sim = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

    # ==========================================
    # DEFAULT STATE BEFORE EXECUTION
    # ==========================================
    if not execute_sim:
        st.info("Please adjust the phase parameters in the sidebar and click 'Run Simulation' to generate the risk model.")
        return

    # ==========================================
    # SIMULATION EXECUTION & DATA PREPARATION
    # ==========================================
    
    # Validate and enforce parameter constraints mathematically
    p1_o, p1_m, p1_p = validate_triangular_params(p1_opt, p1_ml, p1_pess)
    p2_o, p2_m, p2_p = validate_triangular_params(p2_opt, p2_ml, p2_pess)
    p3_o, p3_m, p3_p = validate_triangular_params(p3_opt, p3_ml, p3_pess)
    p4_o, p4_m, p4_p = validate_triangular_params(p4_opt, p4_ml, p4_pess)

    # Execute numpy generators for the entire set
    sim_p1 = np.random.triangular(p1_o, p1_m, p1_p, iterations)
    sim_p2 = np.random.triangular(p2_o, p2_m, p2_p, iterations)
    sim_p3 = np.random.triangular(p3_o, p3_m, p3_p, iterations)
    sim_p4 = np.random.triangular(p4_o, p4_m, p4_p, iterations)

    # Calculate cumulative durations for the spaghetti chart
    cum_p1 = sim_p1
    cum_p2 = cum_p1 + sim_p2
    cum_p3 = cum_p2 + sim_p3
    cum_p4 = cum_p3 + sim_p4
    
    total_durations = cum_p4

    # ==========================================
    # PLACEHOLDERS FOR ANIMATION
    # ==========================================
    metrics_placeholder = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    hist_placeholder = chart_col1.empty()
    cdf_placeholder = chart_col2.empty()
    
    st.markdown("---")
    st.markdown("#### Cumulative Phase Progression (Spaghetti Convergence)")
    spaghetti_placeholder = st.empty()

    # Common styling configurations (relying on Streamlit native theme for colors)
    corporate_layout = dict(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    # ==========================================
    # ANIMATION LOOP
    # ==========================================
    # Divide iterations into 10 frames for smooth animation without freezing the browser
    batch_steps = max(iterations // 10, 100) 
    
    for i in range(batch_steps, iterations + batch_steps, batch_steps):
        current_idx = min(i, iterations)
        
        # Sliced data for current animation frame
        current_totals = total_durations[:current_idx]
        
        # Calculate Key Metrics
        mean_duration = np.mean(current_totals)
        median_duration = np.percentile(current_totals, 50)
        p90_duration = np.percentile(current_totals, 90)

        # 1. Update Metrics
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Expected Completion (Mean)", value=f"{mean_duration:,.0f} Days")
            with col2:
                st.metric(label="50th Percentile (Median)", value=f"{median_duration:,.0f} Days")
            with col3:
                st.metric(label="90th Percentile (Risk Target)", value=f"{p90_duration:,.0f} Days")

        # 2. Update Frequency Histogram
        fig_hist = px.histogram(
            x=current_totals,
            nbins=100,
            color_discrete_sequence=['#4682B4'],
            labels={'x': 'Total Project Duration (Days)', 'y': 'Frequency'},
            title="Project Duration Frequency Distribution"
        )
        fig_hist.update_layout(**corporate_layout)
        fig_hist.add_vline(x=mean_duration, line_dash="dash", line_color="#DC143C", annotation_text="Mean")
        hist_placeholder.plotly_chart(fig_hist, use_container_width=True, theme="streamlit")

        # 3. Update CDF
        fig_cdf = px.ecdf(
            x=current_totals,
            color_discrete_sequence=['#4682B4'],
            labels={'x': 'Total Project Duration (Days)', 'y': 'Probability'},
            title="Cumulative Probability of Completion"
        )
        fig_cdf.update_layout(**corporate_layout)
        fig_cdf.add_hline(y=0.90, line_dash="dot", line_color="#DC143C", annotation_text="90% Confidence")
        fig_cdf.add_vline(x=p90_duration, line_dash="dot", line_color="#DC143C")
        cdf_placeholder.plotly_chart(fig_cdf, use_container_width=True, theme="streamlit")

        # 4. Update Spaghetti Convergence Chart
        fig_spag = go.Figure()
        phases_x = ['Phase 1: Design', 'Phase 2: Site Prep', 'Phase 3: Structure', 'Phase 4: Finishes']
        
        # Background lines (random subset to prevent browser crash, max 100)
        num_paths = min(100, current_idx)
        for j in range(num_paths):
            fig_spag.add_trace(go.Scatter(
                x=phases_x, 
                y=[cum_p1[j], cum_p2[j], cum_p3[j], cum_p4[j]],
                mode='lines',
                line=dict(color='rgba(112, 128, 144, 0.15)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
            
        # Foreground Mean Line
        mean_path_y = [
            np.mean(cum_p1[:current_idx]), 
            np.mean(cum_p2[:current_idx]), 
            np.mean(cum_p3[:current_idx]), 
            np.mean(cum_p4[:current_idx])
        ]
        fig_spag.add_trace(go.Scatter(
            x=phases_x, 
            y=mean_path_y,
            mode='lines+markers',
            line=dict(color='#DC143C', width=4), # Crimson for high visibility
            name='Expected Mean Path'
        ))
        
        # Dummy trace for legend entry of background paths
        fig_spag.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='rgba(112, 128, 144, 0.5)', width=1),
            name='Simulation Paths (Variance)'
        ))

        fig_spag.update_layout(
            **corporate_layout,
            yaxis_title="Cumulative Duration (Days)",
            hovermode="x unified"
        )
        spaghetti_placeholder.plotly_chart(fig_spag, use_container_width=True, theme="streamlit")
        
        # Brief pause for animation effect
        time.sleep(0.05)

    # ==========================================
    # FINAL RAW DATA EXPORT
    # ==========================================
    with st.expander("View Raw Simulation Summary"):
        summary_df = pd.DataFrame(total_durations, columns=['Total Duration (Days)'])
        st.dataframe(summary_df.describe().T, use_container_width=True)

if __name__ == "__main__":
    run_simulation()