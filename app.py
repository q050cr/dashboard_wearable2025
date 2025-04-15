# app.py
from weakref import ref
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pdb

st.set_page_config(
    page_title="Wearable Blood Pressure Monitoring",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:christoph.reich@med.uni-heidelberg.de',
        'About': "# This is a header."
    }
)

# reset flag
if "reset_form" not in st.session_state:
    st.session_state["reset_form"] = False

DATA_PATH = "./data/bp_data.csv"

def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(columns=[
        "timestamp", "subject_id", 
        "ref_sys", "ref_dia", 
        "livemetric_sys", "livemetric_dia",
        "watch_sys", "watch_dia"
    ])

def save_data(df):
    df.to_csv(DATA_PATH, index=False)

def calculate_stats(df):
    df = df.dropna()
    stats = {
        "MAE (LiveMetric SYS)": np.mean(np.abs(df["livemetric_sys"] - df["ref_sys"])),
        "MAE (Watch SYS)": np.mean(np.abs(df["watch_sys"] - df["ref_sys"])),
        "MAE (LiveMetric DIA)": np.mean(np.abs(df["livemetric_dia"] - df["ref_dia"])),
        "MAE (Watch DIA)": np.mean(np.abs(df["watch_dia"] - df["ref_dia"])),
    }
    return stats

# ---------- Plot Functions ----------


def plot_scatter_plotly(df, ref_col, device_col, label):
    df_clean = df[[ref_col, device_col]].dropna()
    max_val = int(df_clean.max().max() * 1.1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_clean[ref_col],
        y=df_clean[device_col],
        mode='markers',
        marker=dict(color='royalblue', size=8,
                    line=dict(width=1, color='black')),
        name=label
    ))
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='Perfect Agreement'
    ))
    fig.update_layout(
        xaxis_title="Reference",
        yaxis_title=label,
        margin=dict(l=40, r=40, t=30, b=40),
        height=400,
        template="plotly_white",
        showlegend=False
    )
    return fig


def bland_altman_plotly(df, ref_col, device_col, label):
    df_clean = df[[ref_col, device_col]].dropna()
    mean = (df_clean[ref_col] + df_clean[device_col]) / 2
    diff = df_clean[device_col] - df_clean[ref_col]
    mean_diff = diff.mean()
    std_diff = diff.std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean,
        y=diff,
        mode='markers',
        marker=dict(color='mediumseagreen', size=8,
                    line=dict(width=1, color='black')),
        name='Data'
    ))
    fig.add_hline(y=mean_diff, line=dict(color='red', dash='dash'))
    fig.add_hline(y=mean_diff + 1.96 * std_diff,
                  line=dict(color='gray', dash='dot'))
    fig.add_hline(y=mean_diff - 1.96 * std_diff,
                  line=dict(color='gray', dash='dot'))

    fig.update_layout(
        xaxis_title='Mean of Device and Reference',
        yaxis_title='Difference (Device - Reference)',
        margin=dict(l=40, r=40, t=30, b=40),
        height=400,
        template='plotly_white',
        showlegend=False
    )
    return fig


# --- UI Starts ---
st.title("🩺 Wearable Blood Pressure Monitoring")

df = load_data()

st.sidebar.header("Add New Measurement")

if st.session_state["reset_form"]:
    st.session_state["subject_id_input"] = ""
    st.session_state["ref_sys_input"] = None
    st.session_state["ref_dia_input"] = None
    st.session_state["lm_sys_input"] = None
    st.session_state["lm_dia_input"] = None
    st.session_state["watch_sys_input"] = None
    st.session_state["watch_dia_input"] = None
    st.session_state["reset_form"] = False  # reset the flag after clearing to prevent looping forever
    st.rerun()  # rerun to apply reset

with st.sidebar.form("new_measurement"):
    subject_id = st.text_input(label="Subject ID", value=None, key="subject_id_input")
    ref_sys = st.number_input(label="Reference SYS", min_value=0, max_value=300, value=None, placeholder="Type a number...", key="ref_sys_input")
    ref_dia = st.number_input(label="Reference DIA", min_value=0, max_value=300, value=None, placeholder="Type a number...", key="ref_dia_input")
    lm_sys = st.number_input(label="LiveMetric SYS", min_value=0, max_value=300, value=None, placeholder="Type a number...", key="lm_sys_input")
    lm_dia = st.number_input(label="LiveMetric DIA", min_value=0, max_value=300, value=None, placeholder="Type a number...", key="lm_dia_input")
    watch_sys = st.number_input(label="Watch SYS", min_value=0, max_value=300, value=None, placeholder="Type a number...", key="watch_sys_input")
    watch_dia = st.number_input(label="Watch DIA", min_value=0, max_value=300, value=None, placeholder="Type a number...", key="watch_dia_input")
    submitted = st.form_submit_button(label="Submit")

    if submitted:
        new_row = {
            "timestamp": datetime.now().isoformat(),
            "subject_id": subject_id,
            "ref_sys": ref_sys,
            "ref_dia": ref_dia,
            "livemetric_sys": lm_sys,
            "livemetric_dia": lm_dia,
            "watch_sys": watch_sys,
            "watch_dia": watch_dia
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_data(df)
        st.success("Measurement added!")

        # ✅ Trigger reset
        st.session_state["reset_form"] = True
        st.rerun()


# --- Data Visualization ---
st.subheader("📊 Device Performance Visualization")

devices = {
    "LiveMetric": ("livemetric_sys", "livemetric_dia"),
    "Watch": ("watch_sys", "watch_dia")
}

for device_name, (sys_col, dia_col) in devices.items():
    with st.expander(f"📈 {device_name}"):
        st.markdown("**Systolic**")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_scatter_plotly(
                df, "ref_sys", sys_col, f"{device_name} SYS"), use_container_width=True)
        with col2:
            st.plotly_chart(bland_altman_plotly(
                df, "ref_sys", sys_col, f"{device_name} SYS"), use_container_width=True)

        st.markdown("**Diastolic**")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_scatter_plotly(
                df, "ref_dia", dia_col, f"{device_name} DIA"), use_container_width=True)
        with col2:
            st.plotly_chart(bland_altman_plotly(
                df, "ref_dia", dia_col, f"{device_name} DIA"), use_container_width=True)


# --- Stats & Raw Data ---
with st.expander("📊 Measurement Statistics", expanded=True):
    stats = calculate_stats(df)

    st.markdown("#### Mean Absolute Error (MAE) [mmHg]")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("LiveMetric SYS", f"{stats['MAE (LiveMetric SYS)']:.2f}")
        st.metric("LiveMetric DIA", f"{stats['MAE (LiveMetric DIA)']:.2f}")
    with col2:
        st.metric("Watch SYS", f"{stats['MAE (Watch SYS)']:.2f}")
        st.metric("Watch DIA", f"{stats['MAE (Watch DIA)']:.2f}")


with st.expander("🗃️ Raw Data Table"):
    COLUMN_RENAME = {
        "timestamp": "Timestamp",
        "subject_id": "Subject ID",
        "ref_sys": "Reference SYS",
        "ref_dia": "Reference DIA",
        "livemetric_sys": "LiveMetric SYS",
        "livemetric_dia": "LiveMetric DIA",
        "watch_sys": "Watch SYS",
        "watch_dia": "Watch DIA"
    }
    st.dataframe(df.rename(columns=COLUMN_RENAME), use_container_width=True)
