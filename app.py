import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure strict adherence to page layout and aesthetic requirements
st.set_page_config(
    page_title="Construction Project Monte Carlo Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply minimal, corporate CSS to force clean aesthetics and remove default padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #2F4F4F;
        }
        .stMetric {
            background-color: #F8F9FA;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #E9ECEF;
        }
    </style>
""", unsafe_allow_html=True)

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
    # SIMULATION EXECUTION
    # ==========================================
    
    # Validate and enforce parameter constraints mathematically
    p1_o, p1_m, p1_p = validate_triangular_params(p1_opt, p1_ml, p1_pess)
    p2_o, p2_m, p2_p = validate_triangular_params(p2_opt, p2_ml, p2_pess)
    p3_o, p3_m, p3_p = validate_triangular_params(p3_opt, p3_ml, p3_pess)
    p4_o, p4_m, p4_p = validate_triangular_params(p4_opt, p4_ml, p4_pess)

    # Execute numpy generators
    sim_p1 = np.random.triangular(p1_o, p1_m, p1_p, iterations)
    sim_p2 = np.random.triangular(p2_o, p2_m, p2_p, iterations)
    sim_p3 = np.random.triangular(p3_o, p3_m, p3_p, iterations)
    sim_p4 = np.random.triangular(p4_o, p4_m, p4_p, iterations)

    # Aggregate total timeline
    total_durations = sim_p1 + sim_p2 + sim_p3 + sim_p4

    # Calculate Key Metrics
    mean_duration = np.mean(total_durations)
    median_duration = np.percentile(total_durations, 50)
    p90_duration = np.percentile(total_durations, 90)

    # ==========================================
    # METRICS DISPLAY
    # ==========================================
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Expected Completion (Mean)", value=f"{mean_duration:,.0f} Days")
    with col2:
        st.metric(label="50th Percentile (Median)", value=f"{median_duration:,.0f} Days")
    with col3:
        st.metric(label="90th Percentile (Risk Target)", value=f"{p90_duration:,.0f} Days")

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # VISUALIZATIONS
    # ==========================================
    chart_col1, chart_col2 = st.columns(2)

    # Common styling configurations
    corporate_layout = dict(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    # 1. Frequency Histogram
    with chart_col1:
        st.markdown("#### Project Duration Frequency Distribution")
        fig_hist = px.histogram(
            x=total_durations,
            nbins=100,
            color_discrete_sequence=['#4682B4'], # Steel Blue
            labels={'x': 'Total Project Duration (Days)', 'y': 'Frequency'}
        )
        fig_hist.update_layout(**corporate_layout)
        fig_hist.add_vline(x=mean_duration, line_dash="dash", line_color="#2F4F4F", annotation_text="Mean")
        st.plotly_chart(fig_hist, use_container_width=True)

    # 2. Cumulative Distribution Function (CDF)
    with chart_col2:
        st.markdown("#### Cumulative Probability of Completion")
        fig_cdf = px.ecdf(
            x=total_durations,
            color_discrete_sequence=['#708090'], # Slate Gray
            labels={'x': 'Total Project Duration (Days)', 'y': 'Probability'}
        )
        fig_cdf.update_layout(**corporate_layout)
        fig_cdf.add_hline(y=0.90, line_dash="dot", line_color="#B22222", annotation_text="90% Confidence")
        fig_cdf.add_vline(x=p90_duration, line_dash="dot", line_color="#B22222")
        st.plotly_chart(fig_cdf, use_container_width=True)

    # Raw Data Export Option
    with st.expander("View Raw Simulation Summary"):
        summary_df = pd.DataFrame(total_durations, columns=['Total Duration (Days)'])
        st.dataframe(summary_df.describe().T, use_container_width=True)

if __name__ == "__main__":
    run_simulation()