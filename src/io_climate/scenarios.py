import numpy as np

def make_shock_vectors(
    node_labels,
    country_codes=None,
    sector_codes=None,
    supply_shock_pct=0.0,
    demand_shock_pct=0.0,
):
    """
    Build supply (sp) and demand (sd) shock vectors for IOClimateModel.

    Parameters
    ----------
    node_labels : list of "CC::P_XYZ" strings
    country_codes : str or list of str, optional
        Country codes to shock (e.g. "IT", or ["IT", "DE"]).
        If None → no country filter.
    sector_codes : str or list of str, optional
        Sector codes to shock (e.g. "P_C10-12", or ["P_C10-12","P_F"]).
        If None → no sector filter.
    supply_shock_pct : float
        % reduction in capacity, e.g. 5 → 5% capacity loss.
    demand_shock_pct : float
        % reduction in final demand, e.g. 10 → 10% demand loss.

    Returns
    -------
    sd, sp : np.ndarray (length n)
        Demand shock vector (fraction), supply shock vector (fraction)
    """

    # Convert % inputs to fractions
    sp_frac = supply_shock_pct / 100.0
    sd_frac = demand_shock_pct / 100.0

    # Make lists
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
        country, sector = label.split("::")

        # Check if node matches filters
        country_match = (not country_codes) or (country in country_codes)
        sector_match  = (not sector_codes) or (sector in sector_codes)

        if country_match and sector_match:
            sp[i] = sp_frac
            sd[i] = sd_frac

    return sd, sp

