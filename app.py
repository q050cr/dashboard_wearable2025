# app.py
import plotly.graph_objects as go
from weakref import ref
import streamlit as st
import pandas as pd
import numpy as np
import os
import platform
import shutil
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

# Detect Streamlit Cloud
IS_CLOUD = "streamlit" in platform.platform().lower()

# Define paths
# On Streamlit Cloud, the app reads & writes to /mount/tmp/bp_data.csv
CLOUD_PATH = "/mount/tmp/bp_data.csv"
REPO_PATH = "./data/bp_data.csv"
DATA_PATH = CLOUD_PATH if IS_CLOUD else REPO_PATH

# One-time cloud fallback: only if no saved file exists yet
if IS_CLOUD and not os.path.exists(CLOUD_PATH):
    if os.path.exists(REPO_PATH):
        shutil.copy(REPO_PATH, CLOUD_PATH)
    else:
        pd.DataFrame(columns=[
            "timestamp", "subject_id", "age", "gender", "post_exercise",
            "ref_sys", "ref_dia",
            "livemetric_sys", "livemetric_dia",
            "watch_sys", "watch_dia"
        ]).to_csv(CLOUD_PATH, index=False)

# --- FNS ---


def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        num_cols = ["ref_sys", "ref_dia", "livemetric_sys",
                    "livemetric_dia", "watch_sys", "watch_dia"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
        return df
    return pd.DataFrame(columns=[
        "timestamp", "subject_id", "age", "gender", "post_exercise",
        "ref_sys", "ref_dia",
        "livemetric_sys", "livemetric_dia",
        "watch_sys", "watch_dia"
    ])


def save_data(df):
    df.to_csv(DATA_PATH, index=False)


def calculate_stats(df):
    # Convert relevant columns to numeric, coercing errors to NaN
    cols = ["ref_sys", "ref_dia", "livemetric_sys",
            "livemetric_dia", "watch_sys", "watch_dia"]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    stats = {}

    for device, sys_col, dia_col in [
        ("LiveMetric", "livemetric_sys", "livemetric_dia"),
        ("Watch", "watch_sys", "watch_dia")
    ]:
        # Systolic
        sys_valid = df[["ref_sys", sys_col]].dropna()
        if not sys_valid.empty:
            stats[f"MAE ({device} SYS)"] = np.mean(
                np.abs(sys_valid[sys_col] - sys_valid["ref_sys"]))
        else:
            stats[f"MAE ({device} SYS)"] = np.nan

        # Diastolic
        dia_valid = df[["ref_dia", dia_col]].dropna()
        if not dia_valid.empty:
            stats[f"MAE ({device} DIA)"] = np.mean(
                np.abs(dia_valid[dia_col] - dia_valid["ref_dia"]))
        else:
            stats[f"MAE ({device} DIA)"] = np.nan

    return stats


# ---------- Plot Functions ----------


def plot_scatter_plotly(df, ref_col, device_col, label):
    df_clean = df[[ref_col, device_col]].dropna()
    if df_clean.empty or df_clean[[ref_col, device_col]].dropna().empty:
        return go.Figure().update_layout(
            title="No valid data available",
            template="plotly_white",
            height=300
        )

    max_val = int(df_clean[[ref_col, device_col]].select_dtypes(
        include=[np.number]).max().max() * 1.1)

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
        title=f"Scatter Plot: {label}",
        xaxis_title="Reference",
        yaxis_title=label,
        margin=dict(l=40, r=40, t=30, b=40),
        height=400,
        template="plotly_white",
        font=dict(size=14),  # Increased font size
        showlegend=False
    )
    return fig


def bland_altman_plotly(df, ref_col, device_col, label="Bland-Altman"):
    df_clean = df[[ref_col, device_col]].dropna()
    mean = (df_clean[ref_col] + df_clean[device_col]) / 2
    diff = df_clean[device_col] - df_clean[ref_col]

    mean_diff = diff.mean()
    std_diff = diff.std()

    upper = mean_diff + 1.96 * std_diff
    lower = mean_diff - 1.96 * std_diff

    fig = go.Figure()

    # Data points
    fig.add_trace(go.Scatter(
        x=mean,
        y=diff,
        mode='markers',
        marker=dict(color='mediumseagreen', size=8,
                    line=dict(width=1, color='black')),
        name='Differences'
    ))

    # Shaded area for limits of agreement
    fig.add_shape(
        type="rect",
        x0=mean.min(), x1=mean.max(),
        y0=lower, y1=upper,
        fillcolor="lightgray",
        opacity=0.2,
        line_width=0,
        layer="below"
    )

    # --- Horizontal lines ----
    # perfect agreement:
    fig.add_hline(y=0, line=dict(color='red', dash='dash'),
                  annotation_text="0 (Perfect)", annotation_position="bottom left")
    # mean diff:
    fig.add_hline(y=mean_diff, line=dict(color='gray', dash='dot'),
                  annotation_text=f"Mean: {mean_diff:.2f}", annotation_position="bottom left")
    # mean ± 1.96 SD:
    fig.add_hline(y=upper, line=dict(color='gray', dash='dot'),
                  annotation_text=f"+1.96 SD: {upper:.2f}", annotation_position="bottom left")
    # mean - 1.96 SD:
    fig.add_hline(y=lower, line=dict(color='gray', dash='dot'),
                  annotation_text=f"-1.96 SD: {lower:.2f}", annotation_position="bottom left")

    # Final layout tweaks
    fig.update_layout(
        title=f"Bland-Altman Plot: {label}",
        xaxis_title='Means',
        yaxis_title='Difference',
        height=400,
        template='plotly_white',
        font=dict(size=14),
        showlegend=True
    )

    return fig


def bland_altman_multiple_devices(df, ref_col, device_cols, label="Bland-Altman"):
    import plotly.graph_objects as go
    fig = go.Figure()

    for device_col in device_cols:
        df_clean = df[[ref_col, device_col]].dropna()
        mean = (df_clean[ref_col] + df_clean[device_col]) / 2
        diff = df_clean[device_col] - df_clean[ref_col]

        mean_diff = diff.mean()
        std_diff = diff.std()
        upper = mean_diff + 1.96 * std_diff
        lower = mean_diff - 1.96 * std_diff

        # Data points per device
        fig.add_trace(go.Scatter(
            x=mean,
            y=diff,
            mode='markers',
            name=f"{device_col}",
            marker=dict(size=8, line=dict(width=1, color='black'))
        ))

        # Optional: Add mean + SD lines per device
        # But consider doing this only for 1 device to avoid clutter
        # Or toggle based on which is being highlighted

    # Add the perfect agreement line once
    fig.add_hline(y=0, line=dict(color='red', dash='dash'),
                  annotation_text="0 (Perfect)", annotation_position="bottom left")

    fig.update_layout(
        title=f"{label}",
        xaxis_title='Means',
        yaxis_title='Difference',
        height=400,
        template='plotly_white',
        font=dict(size=14),
        showlegend=True
    )

    return fig

# --- UI Starts ---


# TITLE
st.markdown(
    "<h1 style='text-align: center;'>🩺</h1>",
    unsafe_allow_html=True
)
st.title("Wearable Blood Pressure Monitoring")

# --- img centered ---
col1, col2, col3 = st.columns([1.5, 2, 1])

with col2:
    st.image(
        "img/wearable_bp.png",
        caption="",
        width=300
    )

# --- Sidebar ---
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
    subject_id = st.text_input(label="Subject ID", value=None, placeholder="e.g. DGK_DO_1130Uhr", key="subject_id_input")
    age = st.number_input(label="Age (years)", min_value=18, max_value=120,
                          value=None, placeholder="e.g. 45", key="age_input")
    col_gender, col_exercise = st.columns(2)
    with col_gender:
        gender = st.selectbox(
            "Gender", options=["", "female", "male"], index=0, key="gender_input")
    with col_exercise:
        post_exercise = st.selectbox("Post Exercise?", options=[
                                    "", "yes", "no"], index=0, key="post_exercise_input")

    # measurments
    # Reference SYS/DIA
    col1, col2 = st.columns(2)
    with col1:
        ref_sys = st.number_input(label="Reference SYS", min_value=0, max_value=300,
                                value=None, placeholder="e.g. 120", key="ref_sys_input")
    with col2:
        ref_dia = st.number_input(label="Reference DIA", min_value=0, max_value=300,
                                value=None, placeholder="e.g. 80", key="ref_dia_input")

    # LiveMetric SYS/DIA
    col3, col4 = st.columns(2)
    with col3:
        lm_sys = st.number_input(label="LiveMetric SYS", min_value=0, max_value=300,
                                value=None, placeholder="", key="lm_sys_input")
    with col4:
        lm_dia = st.number_input(label="LiveMetric DIA", min_value=0, max_value=300,
                                value=None, placeholder="", key="lm_dia_input")

    # Watch SYS/DIA
    col5, col6 = st.columns(2)
    with col5:
        watch_sys = st.number_input(label="Watch SYS", min_value=0, max_value=300,
                                    value=None, placeholder="", key="watch_sys_input")
    with col6:
        watch_dia = st.number_input(label="Watch DIA", min_value=0, max_value=300,
                                    value=None, placeholder="", key="watch_dia_input")



    submitted = st.form_submit_button(label="Submit")

    if submitted:
        new_row = {
            "timestamp": datetime.now().isoformat(),
            "subject_id": subject_id,
            "age": age,
            "gender": gender,
            "post_exercise": post_exercise,
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


# --- Stats & Raw Data ---
with st.expander("📈 Measurement Statistics", expanded=True):
    stats = calculate_stats(df)

    st.markdown("#### Mean Absolute Error (MAE) [mmHg]")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("LiveMetric SYS", f"{stats['MAE (LiveMetric SYS)']:.2f}")
        st.metric("LiveMetric DIA", f"{stats['MAE (LiveMetric DIA)']:.2f}")
    with col2:
        st.metric("Watch SYS", f"{stats['MAE (Watch SYS)']:.2f}")
        st.metric("Watch DIA", f"{stats['MAE (Watch DIA)']:.2f}")


# --- Data Visualization ---
st.subheader("📊 Device Performance Visualization")

devices = {
    "LiveMetric": ("livemetric_sys", "livemetric_dia"),
    "Watch": ("watch_sys", "watch_dia")
}

for device_name, (sys_col, dia_col) in devices.items():
    with st.expander(f"{device_name}"):
        st.markdown("**Systolic**")
        col1, col2 = st.columns(2)

        with col1:
            if df[[sys_col, "ref_sys"]].dropna().empty:
                st.info("No systolic data available for scatter plot.")
            else:
                st.plotly_chart(
                    plot_scatter_plotly(df, "ref_sys", sys_col, f"{device_name} SYS"),
                    use_container_width=True,
                    key=f"{device_name}_scatter_sys"
                )

        with col2:
            if df[[sys_col, "ref_sys"]].dropna().empty:
                st.info("No systolic data available for Bland-Altman plot.")
            else:
                st.plotly_chart(
                    bland_altman_plotly(df, "ref_sys", sys_col, f"{device_name} SYS"),
                    use_container_width=True,
                    key=f"{device_name}_ba_sys"
                )
        st.markdown("---")

        st.markdown("**Diastolic**")
        col1, col2 = st.columns(2)
        with col1:
            if df[[dia_col, "ref_dia"]].dropna().empty:
                st.info("No diastolic data available for scatter plot.")
            else:
                st.plotly_chart(
                    plot_scatter_plotly(df, "ref_dia", dia_col, f"{device_name} DIA"),
                    use_container_width=True,
                    key=f"{device_name}_scatter_dia"
                )

        with col2:
            if df[[dia_col, "ref_dia"]].dropna().empty:
                st.info("No diastolic data available for Bland-Altman plot.")
            else:
                st.plotly_chart(
                    bland_altman_plotly(df, "ref_dia", dia_col, f"{device_name} DIA"),
                    use_container_width=True,
                    key=f"{device_name}_ba_dia"
                )
        st.markdown("---")

# --- insert image ---
st.image("img/blood_pressure_derive.png", caption="From Signal to Pressure", use_container_width=True)


# --- Combined Plots ---
st.subheader("Combined Bland-Altman Plots")

# Create 2 columns: SYS on left, DIA on right
col1, col2 = st.columns(2)

# --- Systolic Combined BA ---
with col1:
    st.markdown("### Systolic")
    systolic_ba = bland_altman_multiple_devices(
        df=df,
        ref_col="ref_sys",
        device_cols=[sys for _, (sys, _) in devices.items()],
        label="Systolic (SYS)"
    )
    st.plotly_chart(systolic_ba, use_container_width=True)

# --- Diastolic Combined BA ---
with col2:
    st.markdown("### Diastolic")
    diastolic_ba = bland_altman_multiple_devices(
        df=df,
        ref_col="ref_dia",
        device_cols=[dia for _, (_, dia) in devices.items()],
        label="Diastolic (DIA)"
    )
    st.plotly_chart(diastolic_ba, use_container_width=True)


# --- Raw Data Table ---
with st.expander("🗃️ Raw Data Table"):
    COLUMN_RENAME = {
        "timestamp": "Timestamp",
        "subject_id": "Subject ID",
        "age": "Age",
        "gender": "Gender",
        "post_exercise": "Post Exercise",
        "ref_sys": "Reference SYS",
        "ref_dia": "Reference DIA",
        "livemetric_sys": "LiveMetric SYS",
        "livemetric_dia": "LiveMetric DIA",
        "watch_sys": "Watch SYS",
        "watch_dia": "Watch DIA"
    }
    st.dataframe(df.rename(columns=COLUMN_RENAME), use_container_width=True)


st.download_button(
    label="📥 Download data as CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name="bp_data_export.csv",
    mime="text/csv"
)
