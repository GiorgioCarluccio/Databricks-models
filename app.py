# app.py

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_io.eurostat_sam import (
    load_sam_latest_year,
    extract_model_inputs_from_sam,
)
from io_climate.model import IOClimateModel


# ---------------------------------------------------------------------
# Scenario helper
# ---------------------------------------------------------------------
def make_shock_vectors(
    node_labels,
    country_codes=None,
    sector_codes=None,
    supply_shock_pct: float = 0.0,
    demand_shock_pct: float = 0.0,
):
    sp_frac = supply_shock_pct / 100.0
    sd_frac = demand_shock_pct / 100.0

    if country_codes is None:
        country_codes = []
    if isinstance(country_codes, str):
        country_codes = [country_codes]

    if sector_codes is None:
        sector_codes = []
    if isinstance(sector_codes, str):
        sector_codes = [sector_codes]

    n = len(node_labels)
    sp = np.zeros(n)
    sd = np.zeros(n)

    for i, label in enumerate(node_labels):
        country, sector = label.split("::", 1)

        country_match = (not country_codes) or (country in country_codes)
        sector_match  = (not sector_codes) or (sector in sector_codes)

        if country_match and sector_match:
            sp[i] = sp_frac
            sd[i] = sd_frac

    return sd, sp


# ---------------------------------------------------------------------
# Spark init via Databricks Connect
# ---------------------------------------------------------------------
@st.cache_resource
def init_spark():
    spark = DatabricksSession.builder.getOrCreate()
    return spark


# ---------------------------------------------------------------------
# Load model data from Databricks tables
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_data():
    spark = init_spark()

    # Load latest SAM year from Databricks catalog
    sam_df, latest_year = load_sam_latest_year(spark)

    # Extract IO matrices/vectors
    Z, FD, X, A, globsec_of, node_labels = extract_model_inputs_from_sam(sam_df)

    # Build model
    model = IOClimateModel(
        Z=Z,
        FD=FD,
        X=X,
        globsec_of=globsec_of,
        A=A,
    )

    # UI: unique countries/sectors
    countries = sorted({lbl.split("::")[0] for lbl in node_labels})
    sectors   = sorted({lbl.split("::")[1] for lbl in node_labels})

    return {
        "spark": spark,
        "latest_year": latest_year,
        "model": model,
        "node_labels": node_labels,
        "countries": countries,
        "sectors": sectors,
    }


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="IO Climate Model", layout="wide")
st.title("Physical Risk Propagation – IO Climate Model")

data = load_model_data()
model = data["model"]
node_labels = data["node_labels"]
countries = data["countries"]
sectors = data["sectors"]
latest_year = data["latest_year"]

# Sidebar controls
st.sidebar.header("Scenario definition")

if latest_year is not None:
    st.sidebar.markdown(f"**Base year:** {latest_year}")

selected_countries = st.sidebar.multiselect("Countries to shock", countries, default=[])
selected_sectors = st.sidebar.multiselect("Sectors to shock (P_*)", sectors, default=[])

supply_shock_pct = st.sidebar.number_input(
    "Supply shock (%) – capacity reduction",
    min_value=0.0, max_value=100.0, value=5.0, step=1.0,
)
demand_shock_pct = st.sidebar.number_input(
    "Demand shock (%) – final demand reduction",
    min_value=0.0, max_value=100.0, value=0.0, step=1.0,
)

run_button = st.sidebar.button("Run scenario")

# Baseline info
st.subheader("Model baseline")
st.write(f"Number of country–sector nodes: **{model.n}**")

# ---------------------------------------------------------------------
# Run scenario & display results
# ---------------------------------------------------------------------
if run_button:
    st.subheader("Scenario results")

    # Build shock vectors
    sd, sp = make_shock_vectors(
        node_labels=node_labels,
        country_codes=selected_countries or None,
        sector_codes=selected_sectors or None,
        supply_shock_pct=supply_shock_pct,
        demand_shock_pct=demand_shock_pct,
    )

    with st.spinner("Running IO model..."):
        results = model.run(
            sd=sd, sp=sp,
            gamma=0.5,
            max_iter=200,
            tol=1e-6,
            demand_update_mode="supply_limited",
            return_history=False,
        )

    st.write(
        f"Converged: **{results['converged']}**, "
        f"iterations: **{results['iterations']}**"
    )

    # Build impacts df
    X0 = model.X
    X1 = results["X_final"]
    loss_pct = (X1 - X0) / (X0 + 1e-12) * 100.0

    df = pd.DataFrame({
        "node": node_labels,
        "country": [lbl.split("::")[0] for lbl in node_labels],
        "sector":  [lbl.split("::")[1] for lbl in node_labels],
        "X_baseline": X0,
        "X_shocked": X1,
        "loss_pct": loss_pct,
    })

    # Top losses
    top_losses = df.sort_values("loss_pct").head(30)

    st.markdown("### Top 30 most negatively affected country–sectors")
    st.dataframe(top_losses, use_container_width=True)

    # Bar chart
    st.markdown("### Impact bar chart (% change in output)")
    chart_data = top_losses[["node", "loss_pct"]].set_index("node")
    st.bar_chart(chart_data)

else:
    st.info("Define a scenario in the sidebar and click **Run scenario**.")
