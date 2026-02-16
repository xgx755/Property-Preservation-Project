#!/usr/bin/env python3
"""
Data Preparation Script for Wake County Property & AMI Explorer
================================================================
Builds wake_master.geojson by joining census tracts, AMI, and tax data.

Strategy:
  - Tax data has zip codes but no coordinates or tract IDs.
  - We geocode each unique Wake County zip code to a lat/lon centroid,
    then use geopandas point-in-polygon to assign each zip → tract(s).
  - Tax metrics are aggregated by zip, then distributed to tracts
    using housing-unit-weighted proportional allocation.
"""

import json, csv, os, time
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Point

BASE_DIR = Path(__file__).parent

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load Census Tract Boundaries
# ═══════════════════════════════════════════════════════════════════════════
print("▶ [1/7] Loading census tract boundaries...")
tracts = gpd.read_file(BASE_DIR / "Census_Tracts_2020.geojson")
tracts["GEOID"] = tracts["GEOID"].astype(str)
tracts["HOUSING_UNITS"] = pd.to_numeric(tracts["HOUSING_UNITS"], errors="coerce").fillna(0)
print(f"  ✓ {len(tracts)} tracts loaded")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Load AMI Data
# ═══════════════════════════════════════════════════════════════════════════
print("▶ [2/7] Loading AMI data...")
ami_path = BASE_DIR / "Area Median Income (AMI)" / "ACSST5Y2024.S1901-Data.csv"
ami_raw = pd.read_csv(ami_path)
ami_raw = ami_raw.iloc[1:]  # skip label/description row

geo_col = ami_raw.columns[0]
ami_raw["GEOID"] = ami_raw[geo_col].str.split("US").str[-1]
ami_raw["ami"] = pd.to_numeric(ami_raw["S1901_C01_012E"], errors="coerce")

ami = ami_raw[["GEOID", "ami"]].dropna(subset=["ami"]).copy()
print(f"  ✓ AMI for {len(ami)} tracts")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Load Tax Data
# ═══════════════════════════════════════════════════════════════════════════
print("▶ [3/7] Loading tax data (this takes a few minutes)...")
tax_path = BASE_DIR / "Property & Tax Records.xlsx"

tax_df = pd.read_excel(
    tax_path, engine="openpyxl",
    usecols=["Assessed_Building_Value", "Assessed_Land_Value",
             "Year_Built", "PHYSICAL_ZIP_CODE"],
)
print(f"  ✓ {len(tax_df)} tax records loaded")

# Clean
tax_df["zip5"] = tax_df["PHYSICAL_ZIP_CODE"].astype(str).str[:5].str.strip()
tax_df = tax_df[tax_df["zip5"].str.match(r"^\d{5}$", na=False)]
tax_df = tax_df[tax_df["zip5"] != "00000"]

for col in ["Assessed_Building_Value", "Assessed_Land_Value", "Year_Built"]:
    tax_df[col] = pd.to_numeric(tax_df[col], errors="coerce")

tax_df["total_val"] = tax_df["Assessed_Building_Value"].fillna(0) + tax_df["Assessed_Land_Value"].fillna(0)
tax_df["land_ratio_p"] = np.where(
    tax_df["total_val"] > 0,
    tax_df["Assessed_Land_Value"].fillna(0) / tax_df["total_val"],
    np.nan,
)
print(f"  ✓ {len(tax_df)} records after cleaning")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Build ZIP → Tract Crosswalk
# ═══════════════════════════════════════════════════════════════════════════
print("▶ [4/7] Building ZIP-to-Tract crosswalk...")

# Known Wake County ZIP code centroids (lat, lon)
# Source: USPS / Census ZCTA reference data for Wake County area
ZIP_CENTROIDS = {
    "27501": (35.5800, -78.8000), "27502": (35.5920, -78.7820),
    "27511": (35.7580, -78.7810), "27513": (35.7880, -78.7620),
    "27518": (35.7270, -78.7380), "27519": (35.7640, -78.8500),
    "27520": (35.5730, -78.5020), "27522": (36.0580, -78.7830),
    "27523": (35.7350, -78.9380), "27526": (35.5500, -78.6650),
    "27529": (35.5910, -78.5690), "27539": (35.6380, -78.7370),
    "27540": (35.6330, -78.8180), "27545": (35.7560, -78.3870),
    "27560": (35.8420, -78.6380), "27562": (35.6500, -78.9000),
    "27571": (35.8730, -78.4690), "27587": (35.9400, -78.5060),
    "27591": (35.7450, -78.4150), "27592": (35.5900, -78.6380),
    "27596": (36.0200, -78.4790), "27597": (35.9120, -78.3770),
    "27601": (35.7740, -78.6350), "27603": (35.6930, -78.6620),
    "27604": (35.8060, -78.5860), "27605": (35.7880, -78.6600),
    "27606": (35.7490, -78.7090), "27607": (35.8010, -78.7180),
    "27608": (35.8120, -78.6520), "27609": (35.8240, -78.6190),
    "27610": (35.7470, -78.5570), "27612": (35.8430, -78.7020),
    "27613": (35.8880, -78.7360), "27614": (35.9220, -78.6340),
    "27615": (35.8730, -78.6370), "27616": (35.8550, -78.5370),
    "27617": (35.8700, -78.7690), "27628": (35.7800, -78.6400),
    "27697": (35.7800, -78.6400), "27701": (35.9970, -78.9020),
    "27703": (35.9560, -78.8300), "27713": (35.9020, -78.9210),
}

# Create GeoDataFrame of zip centroids
zip_points = []
for z, (lat, lon) in ZIP_CENTROIDS.items():
    zip_points.append({"zip5": z, "geometry": Point(lon, lat)})
zip_gdf = gpd.GeoDataFrame(zip_points, crs="EPSG:4326")

# Spatial join: which tract contains each zip centroid?
zip_to_tracts = gpd.sjoin(zip_gdf, tracts[["GEOID", "HOUSING_UNITS", "geometry"]],
                          how="left", predicate="within")

# Some zips may not land in any tract (edge cases) — use nearest
unmatched = zip_to_tracts[zip_to_tracts["GEOID"].isna()]["zip5"].unique()
if len(unmatched) > 0:
    print(f"  ⚠ {len(unmatched)} zips didn't match a tract, using nearest...")
    for z in unmatched:
        pt = zip_gdf[zip_gdf["zip5"] == z].geometry.iloc[0]
        dists = tracts.geometry.distance(pt)
        nearest_idx = dists.idxmin()
        nearest_geoid = tracts.loc[nearest_idx, "GEOID"]
        nearest_hu = tracts.loc[nearest_idx, "HOUSING_UNITS"]
        zip_to_tracts.loc[zip_to_tracts["zip5"] == z, "GEOID"] = nearest_geoid
        zip_to_tracts.loc[zip_to_tracts["zip5"] == z, "HOUSING_UNITS"] = nearest_hu

# Many zips span multiple tracts. Since we only have centroids,
# each zip maps to exactly one tract. But in reality, a zip spans many tracts.
# To improve coverage, we'll do a spatial buffer approach:
# Buffer each zip point by ~2km and find all intersecting tracts.
print("  Expanding zip coverage with spatial buffers...")

# Project to a meters-based CRS for buffering
tracts_proj = tracts.to_crs(epsg=32617)  # UTM zone 17N (covers NC)
zip_proj = zip_gdf.to_crs(epsg=32617)

crosswalk_rows = []
for _, zrow in zip_proj.iterrows():
    z = zrow["zip5"]
    # Buffer by 3km radius to capture nearby tracts
    buffered = zrow.geometry.buffer(3000)
    # Find tracts that intersect the buffer
    mask = tracts_proj.geometry.intersects(buffered)
    matching = tracts_proj[mask]
    if len(matching) == 0:
        # Fallback: nearest tract
        dists = tracts_proj.geometry.distance(zrow.geometry)
        nearest = dists.idxmin()
        crosswalk_rows.append({
            "zip5": z,
            "GEOID": tracts.loc[nearest, "GEOID"],
            "weight": 1.0,
        })
    else:
        # Weight by housing units in each overlapping tract
        total_hu = matching["HOUSING_UNITS"].sum()
        for idx, trow in matching.iterrows():
            w = trow["HOUSING_UNITS"] / total_hu if total_hu > 0 else 1.0 / len(matching)
            crosswalk_rows.append({
                "zip5": z,
                "GEOID": trow["GEOID"],
                "weight": w,
            })

xwalk = pd.DataFrame(crosswalk_rows)

# Normalize weights per zip
wsum = xwalk.groupby("zip5")["weight"].transform("sum")
xwalk["weight"] = xwalk["weight"] / wsum.replace(0, 1)

print(f"  ✓ {len(xwalk)} zip→tract mappings ({xwalk['zip5'].nunique()} zips → {xwalk['GEOID'].nunique()} tracts)")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Aggregate Tax Data by ZIP, then Distribute to Tracts
# ═══════════════════════════════════════════════════════════════════════════
print("▶ [5/7] Aggregating tax data to tracts...")

zip_agg = tax_df.groupby("zip5").agg(
    avg_val=("total_val", "mean"),
    avg_year=("Year_Built", lambda x: x[x > 0].mean() if (x > 0).any() else np.nan),
    avg_land_ratio=("land_ratio_p", "mean"),
    parcel_count=("total_val", "size"),
).reset_index()

# Join zip aggregates through the crosswalk
tract_via_zip = xwalk.merge(zip_agg, on="zip5", how="inner")

# Weighted aggregation to tract level
tract_via_zip["w_val"] = tract_via_zip["avg_val"] * tract_via_zip["weight"]
tract_via_zip["w_year"] = tract_via_zip["avg_year"] * tract_via_zip["weight"]
tract_via_zip["w_lr"] = tract_via_zip["avg_land_ratio"] * tract_via_zip["weight"]
tract_via_zip["w_count"] = tract_via_zip["parcel_count"] * tract_via_zip["weight"]

tract_agg = tract_via_zip.groupby("GEOID").agg(
    avg_val=("w_val", "sum"),
    avg_year=("w_year", "sum"),
    land_ratio=("w_lr", "sum"),
    parcel_count=("w_count", "sum"),
).reset_index()

# Convert types explicitly
for col in ["avg_val", "avg_year", "land_ratio", "parcel_count"]:
    tract_agg[col] = pd.to_numeric(tract_agg[col], errors="coerce")

print(f"  ✓ Tax data mapped to {len(tract_agg)} tracts")

# ═══════════════════════════════════════════════════════════════════════════
# 6. Merge Everything
# ═══════════════════════════════════════════════════════════════════════════
print("▶ [6/7] Merging all data layers...")

merged = tracts[["GEOID", "NAME", "NAMELSAD", "HOUSING_UNITS", "geometry"]].copy()
merged = merged.merge(ami, on="GEOID", how="left")
merged = merged.merge(tract_agg, on="GEOID", how="left")

# Compute VTI ratio
merged["vti_ratio"] = np.where(
    (merged["ami"].notna()) & (merged["ami"] > 0) &
    (merged["avg_val"].notna()) & (merged["avg_val"] > 0),
    merged["avg_val"] / merged["ami"],
    np.nan,
)

# Round for clean display
merged["avg_val"] = pd.to_numeric(merged["avg_val"], errors="coerce").round(0)
merged["avg_year"] = pd.to_numeric(merged["avg_year"], errors="coerce").round(0)
merged["land_ratio"] = pd.to_numeric(merged["land_ratio"], errors="coerce").round(3)
merged["vti_ratio"] = pd.to_numeric(merged["vti_ratio"], errors="coerce").round(2)
merged["ami"] = pd.to_numeric(merged["ami"], errors="coerce").round(0)

# Summary
print("\n── Summary Statistics ──────────────────────────────")
for col in ["avg_val", "avg_year", "land_ratio", "ami", "vti_ratio"]:
    s = merged[col].dropna()
    if len(s) > 0:
        print(f"  {col:12s}  min={s.min():>12,.1f}  mean={s.mean():>12,.1f}  max={s.max():>12,.1f}  ({len(s)}/{len(merged)} tracts)")

# ═══════════════════════════════════════════════════════════════════════════
# 7. Export GeoJSON
# ═══════════════════════════════════════════════════════════════════════════
print("\n▶ [7/7] Exporting wake_master.geojson...")

# Keep only necessary columns
output_cols = ["GEOID", "NAME", "NAMELSAD", "avg_val", "avg_year",
               "land_ratio", "ami", "vti_ratio", "geometry"]
output = merged[[c for c in output_cols if c in merged.columns]].copy()

output_path = BASE_DIR / "wake_master.geojson"
output.to_file(output_path, driver="GeoJSON")

size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"  ✓ Exported wake_master.geojson ({size_mb:.1f} MB)")
print(f"  ✓ {len(output)} tracts with {output['vti_ratio'].notna().sum()} having VTI data")
print("\n✅ Data preparation complete!")
