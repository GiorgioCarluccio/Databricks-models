from typing import List, Tuple
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


DEFAULT_SAM_TABLE = "openeconomics.gold.couind_eurostat_sam_y"
FINAL_DEMAND_ACCOUNTS = ["HH", "GOV", "CF", "WRL_REST"]
GDP_ACCOUNTS = ["LAB", "CAP", "TAX"]  # not used in IO matrices, but documented
NACE_DIM_TABLE = "openeconomics.silver.dim_eurostat_nace_21"


def load_sam_latest_year(
    spark: SparkSession,
    table_name: str = DEFAULT_SAM_TABLE,
) -> Tuple[DataFrame, int]:
    """
    Load the EU multi-country SAM in long format from the Databricks catalog,
    and filter it to the most recent year available.

    Parameters
    ----------
    spark : SparkSession
        Active SparkSession (from Databricks / VS Code).
    table_name : str
        Fully-qualified table name in the catalog.

    Returns
    -------
    sam_df_latest : DataFrame
        SAM entries for the most recent year only.
    latest_year : int
        The year used for filtering.
    """
    # Read the table
    sam_df = spark.table(table_name)

    # Find the latest year
    latest_year_row = sam_df.select(
        F.max("time_period").alias("latest_year")
    ).collect()[0]
    latest_year = int(latest_year_row["latest_year"])

    # Filter to that year
    sam_df_latest = sam_df.filter(F.col("time_period") == latest_year)

    return sam_df_latest, latest_year



def extract_model_inputs_from_sam(
    sam_df: DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Extract IO-style model inputs (Z, FD, X, A, globsec_of, node_labels)
    from a SAM in long format, using efficient pivoting in pandas.

    Nodes i are country–sector PRODUCTION accounts:
        - accounts with labels starting with 'P_' (both origin & destination)

    Z  : intermediate-use matrix between production nodes
    FD : final demand from (HH, GOV, CF, WRL_REST)
    X  : gross output = row-sum(Z) + FD
    A  : technical coefficients matrix, from the `share` column
    globsec_of : global sector id (same sector across countries => same id)

    Parameters
    ----------
    sam_df : DataFrame
        SAM in long format for a single year, with columns:
        c_orig, ind_ava, c_dest, ind_use, value, share, ...

    Returns
    -------
    Z : (n, n) np.ndarray
    FD : (n,)   np.ndarray
    X : (n,)    np.ndarray
    A : (n, n)  np.ndarray
    globsec_of : (n,) np.ndarray
    node_labels : list of str  (e.g. "DE::P_C10-12")
    """

    # ------------------------------------------------------------------ #
    # 1) Production nodes = union of P_* on origin and destination sides
    # ------------------------------------------------------------------ #

    prod_orig = (
        sam_df
        .filter(F.col("ind_ava").startswith("P_"))
        .select(
            F.col("c_orig").alias("country"),
            F.col("ind_ava").alias("sector"),
        )
    )

    prod_dest = (
        sam_df
        .filter(F.col("ind_use").startswith("P_"))
        .select(
            F.col("c_dest").alias("country"),
            F.col("ind_use").alias("sector"),
        )
    )

    prod_nodes_df = prod_orig.union(prod_dest).distinct()

    prod_nodes_df = prod_nodes_df.withColumn(
        "node_label",
        F.concat_ws("::", "country", "sector"),
    )

    # Stable order: by country, then sector
    node_pd = prod_nodes_df.orderBy("country", "sector").toPandas()
    node_labels = node_pd["node_label"].tolist()
    n = len(node_labels)

    # ------------------------------------------------------------------ #
    # 2) Intermediate block: P_* -> P_*  (Z and A via pandas pivot)
    # ------------------------------------------------------------------ #

    inter_df = (
        sam_df
        .filter(
            F.col("ind_ava").startswith("P_")
            & F.col("ind_use").startswith("P_")
        )
        .select(
            "c_orig", "ind_ava",
            "c_dest", "ind_use",
            "value", "share",
        )
    )

    inter_pd = inter_df.toPandas()
    inter_pd["row_label"] = inter_pd["c_orig"] + "::" + inter_pd["ind_ava"]
    inter_pd["col_label"] = inter_pd["c_dest"] + "::" + inter_pd["ind_use"]

    # Z: pivot value with sum aggregation, missing pairs → 0
    Z_pd = inter_pd.pivot_table(
        index="row_label",
        columns="col_label",
        values="value",
        aggfunc="sum",
        fill_value=0.0,
    )

    # A: pivot share with sum aggregation, missing pairs → 0
    A_pd = inter_pd.pivot_table(
        index="row_label",
        columns="col_label",
        values="share",
        aggfunc="sum",
        fill_value=0.0,
    )

    # Ensure full n×n grid in correct order
    Z_pd = Z_pd.reindex(index=node_labels, columns=node_labels, fill_value=0.0)
    A_pd = A_pd.reindex(index=node_labels, columns=node_labels, fill_value=0.0)

    Z = Z_pd.to_numpy(dtype=float)
    A = A_pd.to_numpy(dtype=float)

    # ------------------------------------------------------------------ #
    # 3) Final demand: P_* → {HH, GOV, CF, WRL_REST}
    # ------------------------------------------------------------------ #

    fd_df = (
        sam_df
        .filter(
            F.col("ind_ava").startswith("P_")
            & F.col("ind_use").isin(FINAL_DEMAND_ACCOUNTS)
        )
        .select(
            F.concat_ws("::", "c_orig", "ind_ava").alias("row_label"),
            "value",
        )
        .groupBy("row_label")
        .agg(F.sum("value").alias("fd_value"))
    )

    fd_pd = fd_df.toPandas().set_index("row_label")
    fd_pd = fd_pd.reindex(index=node_labels).fillna(0.0)
    FD = fd_pd["fd_value"].to_numpy(dtype=float)

    # ------------------------------------------------------------------ #
    # 4) Gross output X: row-sum(Z) + FD  (IO-style, ignoring GDP block)
    # ------------------------------------------------------------------ #

    X = Z.sum(axis=1) + FD

    # ------------------------------------------------------------------ #
    # 5) Global sector mapping: same P_* across countries ⇒ same id
    # ------------------------------------------------------------------ #

    sectors = node_pd["sector"].tolist()
    unique_sectors = sorted(set(sectors))
    sector_to_global = {sec: k for k, sec in enumerate(unique_sectors)}

    globsec_of = np.array(
        [sector_to_global[sec] for sec in sectors],
        dtype=int,
    )

    return Z, FD, X, A, globsec_of, node_labels

