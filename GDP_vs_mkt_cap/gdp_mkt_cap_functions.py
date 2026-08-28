import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import requests
import re

import yfinance as yf

from bs4 import BeautifulSoup

def build_crp_country_translation_csv(
    crp_path="country_risk_premiums.csv",
    translation_out="crp_country_translation.csv",
):
    """
    Build the CRP-to-World-Bank translation CSV once and save it locally.
    This should be run once to define the mapping, then the CSV can be reviewed
    and edited manually downstream without redoing the country matching logic.
    """
    crp = pd.read_csv(crp_path)
    if "Country" not in crp.columns:
        raise ValueError("CRP file must contain a 'Country' column.")

    def norm(s):
        if pd.isna(s):
            return ""
        s = str(s).lower().strip()
        s = s.replace("&", " and ")
        s = s.replace(".", "")
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    # Manual overrides for names that do not match World Bank labels exactly.
    # This translation layer is the editable definition file for the mapping.
    manual_name_map = {
        "Czech Republic": "Czechia",
        "Korea, Republic of": "Korea, Rep.",
        "Republic of Korea": "Korea, Rep.",
        "Russian Federation": "Russian Federation",
        "Venezuela, RB": "Venezuela, RB",
        "Macedonia, FYR": "North Macedonia",
        "Cote d'Ivoire": "Cote d'Ivoire",
        "Egypt, Arab Rep.": "Egypt, Arab Rep.",
        "Iran, Islamic Rep.": "Iran, Islamic Rep.",
        "Gambia, The": "Gambia, The",
        "Yemen, Rep.": "Yemen, Rep.",
        "Brunei Darussalam": "Brunei Darussalam",
        "Slovak Republic": "Slovakia",
        "United States": "United States",
        "United Kingdom": "United Kingdom"
    }

    meta_url = "https://api.worldbank.org/v2/country/all?format=json&per_page=400"
    r = requests.get(meta_url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    wb_country_df = pd.DataFrame(payload[1])

    if wb_country_df.empty:
        raise RuntimeError("World Bank country metadata returned no rows.")

    wb_country_df = wb_country_df[["id", "iso2Code", "name"]].rename(columns={
        "id": "country_code_iso3",
        "iso2Code": "country_code_iso2",
        "name": "country_name_wb"
    })
    wb_country_df["match_key"] = wb_country_df["country_name_wb"].map(norm)

    crp_lookup = crp[["Country"]].rename(columns={"Country": "country_name_in_crp"}).copy()
    crp_lookup["manual_lookup"] = crp_lookup["country_name_in_crp"].map(manual_name_map)
    crp_lookup["match_key"] = crp_lookup["manual_lookup"].fillna(crp_lookup["country_name_in_crp"]).map(norm)

    translation = crp_lookup.merge(
        wb_country_df[["match_key", "country_name_wb", "country_code_iso3", "country_code_iso2"]],
        on="match_key",
        how="left"
    )

    translation["match_type"] = np.where(
        translation["manual_lookup"].isna(),
        "normalized_exact_or_close",
        "manual_override"
    )

    translation = translation[[
        "country_name_in_crp",
        "country_name_wb",
        "country_code_iso3",
        "country_code_iso2",
        "match_type"
    ]].copy()

    translation["country_code_source"] = "World Bank country metadata API"
    translation["translation_layer_source"] = (
        "World Bank country metadata API; manual overrides for naming differences "
        "between CRP labels and World Bank labels"
    )

    translation.to_csv(translation_out, index=False)
    return translation


def download_worldbank_gdp_from_translation_csv(
    translation_path="crp_country_translation.csv",
    gdp_out="annual_gdp_by_country.csv",
):
    """
    Read the saved translation CSV and download annual GDP data for each matched country.
    This function assumes the translation CSV has already been generated and can be
    edited/reviewed by hand before GDP extraction.
    """
    translation = pd.read_csv(translation_path)
    required_cols = {"country_name_in_crp", "country_name_wb", "country_code_iso3", "country_code_iso2"}
    missing = sorted(required_cols - set(translation.columns))
    if missing:
        raise ValueError(
            "Translation file is missing required columns: " + ", ".join(missing)
        )

    translation = translation.dropna(subset=["country_code_iso3"]).copy()
    rows = []

    for _, row in translation.iterrows():
        iso3 = row["country_code_iso3"]
        gdp_url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/NY.GDP.MKTP.CD?format=json&per_page=200"
        try:
            g = requests.get(gdp_url, timeout=30)
            g.raise_for_status()
            data = g.json()
            obs = data[1] if isinstance(data, list) and len(data) > 1 else []
        except Exception:
            continue

        for entry in obs:
            value = entry.get("value")
            if value is None:
                continue
            rows.append({
                "country_name_in_crp": row["country_name_in_crp"],
                "country_name_wb": row["country_name_wb"],
                "country_code_iso3": iso3,
                "country_code_iso2": row["country_code_iso2"],
                "year": int(entry.get("date", np.nan)),
                "annual_gdp_usd": float(value),
                "source": "World Bank WDI indicator NY.GDP.MKTP.CD"
            })

    gdp_df = pd.DataFrame(rows)
    if not gdp_df.empty:
        gdp_df = gdp_df.sort_values(["country_name_in_crp", "year"]).reset_index(drop=True)

    gdp_df.to_csv(gdp_out, index=False)
    return gdp_df

def calculate_expected_gdp_growth_rate(
    crp_df=None,
    gdp_path="annual_gdp_by_country.csv",
    crp_path="country_risk_premiums.csv",
    growth_years=5,
    out_path="crp_data_with_expected_gdp_growth.csv",
):
    """
    Compute a trailing expected GDP growth rate for each country using annual GDP levels.

    This uses a compound annual growth rate (CAGR) over the most recent `growth_years`
    complete years available for each country, then joins that figure onto the CRP data.

    Returns the CRP table with a new column:
      - expected_gdp_growth_rate_pct
    """
    if crp_df is None:
        crp_df = pd.read_csv(crp_path)

    gdp = pd.read_csv(gdp_path)
    if {"country_name_in_crp", "year", "annual_gdp_usd"} - set(gdp.columns):
        raise ValueError("GDP file must contain country_name_in_crp, year, and annual_gdp_usd columns.")

    gdp = gdp.dropna(subset=["country_name_in_crp", "year", "annual_gdp_usd"]).copy()
    gdp["year"] = gdp["year"].astype(int)
    gdp = gdp.sort_values(["country_name_in_crp", "year"]).reset_index(drop=True)

    growth_rows = []
    for country, group in gdp.groupby("country_name_in_crp"):
        recent = group.sort_values("year").reset_index(drop=True)
        if len(recent) < 2:
            continue

        latest_year = recent["year"].max()
        start_year = latest_year - (growth_years - 1)
        recent_slice = recent[recent["year"] >= start_year].copy()
        if len(recent_slice) < 2:
            continue

        start_value = recent_slice.iloc[0]["annual_gdp_usd"]
        end_value = recent_slice.iloc[-1]["annual_gdp_usd"]
        years_span = recent_slice["year"].iloc[-1] - recent_slice["year"].iloc[0]

        if pd.isna(start_value) or pd.isna(end_value) or start_value <= 0 or years_span <= 0:
            continue

        cagr = ((end_value / start_value) ** (1 / years_span) - 1) 
        growth_rows.append({
            "country_name_in_crp": country,
            "expected_gdp_growth_rate_pct": cagr,
            "expected_gdp_growth_window_years": years_span,
            "expected_gdp_growth_start_year": recent_slice["year"].iloc[0],
            "expected_gdp_growth_end_year": recent_slice["year"].iloc[-1],
        })

    growth_df = pd.DataFrame(growth_rows)
    if growth_df.empty:
        merged = crp_df.copy()
        merged["expected_gdp_growth_rate_pct"] = np.nan
        merged["expected_gdp_growth_window_years"] = np.nan
        merged["expected_gdp_growth_start_year"] = np.nan
        merged["expected_gdp_growth_end_year"] = np.nan
        if out_path is not None:
            merged.to_csv(out_path, index=False)
        return merged

    merged = crp_df.merge(
        growth_df,
        left_on="Country",
        right_on="country_name_in_crp",
        how="left"
    )

    merged = merged.drop(columns=["country_name_in_crp"], errors="ignore")

    if out_path is not None:
        merged.to_csv(out_path, index=False)

    return merged



def fetch_market_cap_data():
    url = "https://companiesmarketcap.com/"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    table = soup.select_one(".marketcap-table")
    if table is None:
        raise ValueError("Could not find the CompaniesMarketCap table")

    for tr in table.select("tbody tr"):
        try:
            name_cell = tr.select_one(".company-name")
            market_cell = tr.select("td")[3]  
            country_cell = tr.select("td")[-1]
            if not name_cell or not market_cell:
                continue
            
            company = name_cell.get_text(" ", strip=True)
            country = country_cell.get_text(" ", strip=True)
  
            market_cap_raw = market_cell.get("data-sort")
            if market_cap_raw is None or market_cap_raw == "":
                continue

            market_cap_usd = float(market_cap_raw)
            country = country_cell.get_text(" ", strip=True)


            rows.append({
            "company": company,
            "market_cap_usd": market_cap_usd,
            "country": country
        })
        except Exception:
            pass

    cmc = pd.DataFrame(rows).sort_values("market_cap_usd", ascending=False).reset_index(drop=True)

    top_10 = cmc.head(10).copy()
    top_10["company"] = top_10["company"].str.replace("  ", " ", regex=False)
    return top_10