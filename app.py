import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Construction Project Monte Carlo Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Corporate Styling and UI Injection
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Minimalist styling for metrics and headers */
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #4682B4;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Application Main Header
# -----------------------------------------------------------------------------
st.title("Construction Project Timeline Risk Analysis")
st.markdown(
    "Probabilistic Monte Carlo simulation for a multi-phase custom residential build. "
    "Adjust the triangular distribution parameters in the sidebar to model timeline uncertainties."
)
st.markdown("---")

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("Simulation Parameters")
st.sidebar.markdown("Define duration estimates (in days) for each phase.")

def render_phase_sliders(phase_name, default_opt, default_ml, default_pess, key_prefix):
    """Helper function to generate sliders for triangular distribution inputs."""
    st.sidebar.subheader(phase_name)
    
    # Ensure logical constraints: Optimistic <= Most Likely <= Pessimistic
    opt = st.sidebar.slider(
        "Optimistic (Days)", 
        min_value=5, max_value=300, value=default_opt, key=f"{key_prefix}_opt"
    )
    ml = st.sidebar.slider(
        "Most Likely (Days)", 
        min_value=opt, max_value=max(300, opt+100), value=max(default_ml, opt), key=f"{key_prefix}_ml"
    )
    pess = st.sidebar.slider(
        "Pessimistic (Days)", 
        min_value=ml, max_value=max(500, ml+200), value=max(default_pess, ml), key=f"{key_prefix}_pess"
    )
    st.sidebar.markdown("") # Spacing
    return opt, ml, pess

# Phase 1: Architectural Design & Municipal Permitting
opt_1, ml_1, pess_1 = render_phase_sliders(
    "1. Design & Permitting", 30, 45, 90, "p1"
)

# Phase 2: Site Preparation & Foundation
opt_2, ml_2, pess_2 = render_phase_sliders(
    "2. Site Prep & Foundation", 15, 25, 60, "p2"
)

# Phase 3: Structural Build & Utilities Routing
opt_3, ml_3, pess_3 = render_phase_sliders(
    "3. Structural & Utilities", 45, 75, 120, "p3"
)

# Phase 4: Interior Finishes & Final Inspections
opt_4, ml_4, pess_4 = render_phase_sliders(
    "4. Finishes & Inspections", 30, 50, 90, "p4"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Execution Controls")
iterations = st.sidebar.slider(
    "Simulation Iterations", 
    min_value=1000, 
    max_value=50000, 
    value=10000, 
    step=1000
)

run_button = st.sidebar.button("Run Simulation", use_container_width=True)

# -----------------------------------------------------------------------------
# Simulation Logic and Visualization
# -----------------------------------------------------------------------------
if run_button:
    # Execute Monte Carlo sampling
    # Vectorized operations using numpy for performance over large iterations
    dist_1 = np.random.triangular(opt_1, ml_1, pess_1, iterations)
    dist_2 = np.random.triangular(opt_2, ml_2, pess_2, iterations)
    dist_3 = np.random.triangular(opt_3, ml_3, pess_3, iterations)
    dist_4 = np.random.triangular(opt_4, ml_4, pess_4, iterations)
    
    total_durations = dist_1 + dist_2 + dist_3 + dist_4
    
    # Calculate key statistical metrics
    mean_duration = np.mean(total_durations)
    median_duration = np.percentile(total_durations, 50)
    p90_duration = np.percentile(total_durations, 90)
    
    # Display Top-Level Metrics
    st.subheader("Project Duration Risk Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Expected Completion (Mean)", 
            value=f"{mean_duration:.1f} Days"
        )
    with col2:
        st.metric(
            label="50th Percentile (Median)", 
            value=f"{median_duration:.1f} Days",
            help="50% probability of completing the project on or before this day."
        )
    with col3:
        st.metric(
            label="90th Percentile (Risk Target)", 
            value=f"{p90_duration:.1f} Days",
            help="90% probability of completing the project on or before this day. Commonly used for conservative baseline scheduling."
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    st.subheader("Simulation Distribution Analysis")
    
    chart_col1, chart_col2 = st.columns(2)
    
    # Chart 1: Frequency Histogram
    with chart_col1:
        fig_hist = px.histogram(
            x=total_durations, 
            nbins=50,
            title="Frequency of Total Project Durations",
            labels={'x': 'Total Duration (Days)', 'y': 'Frequency'},
            color_discrete_sequence=['#4682B4'] # Steel Blue
        )
        
        fig_hist.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            title_font=dict(size=16, color="#2c3e50"),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    # Chart 2: Cumulative Distribution Function (CDF)
    with chart_col2:
        # Sort data for accurate manual ECDF plotting
        sorted_data = np.sort(total_durations)
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(
            x=sorted_data, 
            y=y_vals,
            mode='lines',
            name='CDF',
            line=dict(color='#708090', width=3) # Slate Gray
        ))
        
        # Add P90 Target Line
        fig_cdf.add_hline(
            y=0.90, line_dash="dash", line_color="#E74C3C", 
            annotation_text="90% Confidence", 
            annotation_position="bottom right"
        )
        
        fig_cdf.update_layout(
            title="Cumulative Probability of Completion",
            xaxis_title="Total Duration (Days)",
            yaxis_title="Probability",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            title_font=dict(size=16, color="#2c3e50"),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, tickformat=".0%")
        )
        st.plotly_chart(fig_cdf, use_container_width=True)

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to view the risk analysis.")