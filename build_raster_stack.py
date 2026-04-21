"""
============================================================
NATIONAL CRITICAL MINERAL PROSPECTIVITY MAPPING — CANADA
Data Preparation Pipeline: Provincial Geochemistry + Physics
→ Harmonized Raster Stack (canada_raster_stack.csv)

Inputs:
  - rock_geochem_dataset1-BC.xlsx
  - rock_geochem_dataset2-BC.xlsx
  - MRD_347_Lithogeochemistry_-_Ontario.xlsx
  - Lithogeochemistry-Man.xlsx
  - Lithogeochemistry_AnalysesSask.csv
  - LithogeoYukon.kml
  - acoustic_properties.csv
  - densities_and_magnetic.csv

Output:
  - Canada_geochemistry_merged.csv   (harmonized point samples)
  - canada_raster_stack.csv          (10km x 10km raster grid)

Projection: EPSG:3978 (Canada Atlas Lambert)
Grid resolution: 10 km x 10 km
Interpolation: IDW (k=8 neighbours, 15km spatial thinning)
============================================================
"""

import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET
from pyproj import Transformer
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# TARGET ANALYTE SCHEMA
# All provinces harmonized to these canonical column names
# ============================================================
OXIDE_COLS = [
    'SiO2_pct', 'Al2O3_pct', 'Fe2O3_pct', 'MgO_pct', 'CaO_pct',
    'Na2O_pct', 'K2O_pct', 'TiO2_pct', 'P2O5_pct', 'MnO_pct'
]

ELEMENT_COLS = [
    'Au_ppb', 'Cu_ppm', 'Ni_ppm', 'Co_ppm', 'Li_ppm', 'Cr_ppm',
    'Zn_ppm', 'Pb_ppm', 'Mo_ppm', 'As_ppm', 'Th_ppm', 'U_ppm',
    'La_ppm', 'Ce_ppm', 'Nd_ppm', 'Nb_ppm', 'V_ppm', 'W_ppm'
]

ALL_ANALYTES = OXIDE_COLS + ELEMENT_COLS

META_COLS = ['province', 'wgs84_long', 'wgs84_lat']


# ============================================================
# HELPER: Detection limit / sentinel parsing
# <X  → X/2  (standard DL/2 convention)
# >X  → X    (conservative upper bound)
# negative → abs(val)/2  (below-detection sentinel)
# *, n.a., blank → NaN
# ============================================================
def parse_value(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ('', '*', 'n.a.', 'N/A', 'na', 'NA', '-', 'nd', 'ND', 'bdl'):
        return np.nan
    m = re.match(r'^<\s*([\d.]+)$', s)
    if m:
        return float(m.group(1)) / 2.0
    m = re.match(r'^>\s*([\d.]+)$', s)
    if m:
        return float(m.group(1))
    try:
        v = float(s)
        return abs(v) / 2.0 if v < 0 else v
    except ValueError:
        return np.nan


def parse_cols(df, cols):
    """Apply parse_value to a list of columns in-place."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(parse_value)
    return df


# ============================================================
# BC — method-priority coalescing
# Priority: FA > XRF/LIP > INA > MAS > AIP/MIP > TD > NTD > PD
# ============================================================
def coalesce_bc(row, candidates):
    """Return first non-NaN value across method-priority candidates."""
    for c in candidates:
        v = row.get(c, np.nan)
        if pd.notna(v) and v != 0:
            return v
    return np.nan


def load_bc(path1, path2):
    print(f"  Loading BC: {path1.split('/')[-1]}, {path2.split('/')[-1]}")
    dfs = []
    for path in [path1, path2]:
        df = pd.read_excel(path)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # coordinates already in WGS84
    df = df.dropna(subset=['wgs84_long', 'wgs84_lat'])

    out = pd.DataFrame()
    out['wgs84_long'] = df['wgs84_long']
    out['wgs84_lat'] = df['wgs84_lat']
    out['province'] = 'British Columbia'

    # Method priority lists per analyte
    # FA (fire assay) > XRF/LIP > INA > MAS > AIP/MIP > TD > NTD > PD
    analyte_map = {
        'Au_ppb':    ['Au_FA_ppb', 'Au1_FA_ppb', 'Au_INA_ppb', 'Au_MIP_ppm',
                      'Au_TD_ppb', 'Au_NTD_ppb', 'Au_PD_ppb', 'Au_AAS_ppb', 'Au_AIP_ppb'],
        'Cu_ppm':    ['Cu_LIP_ppm', 'Cu_XRF_ppm', 'Cu_INA_ppm', 'Cu_MAS_ppm',
                      'Cu_AIP_ppm', 'Cu_MIP_ppm', 'Cu_TD_ppm', 'Cu_NTD_ppm',
                      'Cu_PD_ppm', 'Cu_AAS_ppm'],
        'Ni_ppm':    ['Ni_LIP_ppm', 'Ni_XRF_ppm', 'Ni_INA_ppm', 'Ni_MAS_ppm',
                      'Ni_AIP_ppm', 'Ni_MIP_ppm', 'Ni_TD_ppm', 'Ni_NTD_ppm',
                      'Ni_PD_ppm', 'Ni_AAS_ppm'],
        'Co_ppm':    ['Co_LIP_ppm', 'Co_INA_ppm', 'Co_MAS_ppm', 'Co_AIP_ppm',
                      'Co_MIP_ppm', 'Co_TD_ppm', 'Co_NTD_ppm', 'Co_PD_ppm', 'Co_AAS_ppm'],
        'Li_ppm':    ['Li_LIP_ppm', 'Li_INA_ppm', 'Li_MIP_ppm', 'Li_AIP_ppm',
                      'Li_TD_ppm', 'Li_NTD_ppm', 'Li_PD_ppm'],
        'Cr_ppm':    ['Cr_LIP_ppm', 'Cr_XRF_ppm', 'Cr_INA_ppm', 'Cr_MAS_ppm',
                      'Cr_AIP_ppm', 'Cr_MIP_ppm', 'Cr_TD_ppm', 'Cr_NTD_ppm', 'Cr_PD_ppm'],
        'Zn_ppm':    ['Zn_LIP_ppm', 'Zn_INA_ppm', 'Zn_MIP_ppm', 'Zn_AIP_ppm',
                      'Zn_TD_ppm', 'Zn_NTD_ppm', 'Zn_PD_ppm', 'Zn_AAS_ppm'],
        'Pb_ppm':    ['Pb_LIP_ppm', 'Pb_INA_ppm', 'Pb_MIP_ppm', 'Pb_AIP_ppm',
                      'Pb_TD_ppm', 'Pb_NTD_ppm', 'Pb_PD_ppm'],
        'Mo_ppm':    ['Mo_LIP_ppm', 'Mo_INA_ppm', 'Mo_MAS_ppm', 'Mo_MIP_ppm',
                      'Mo_AIP_ppm', 'Mo_TD_ppm', 'Mo_NTD_ppm', 'Mo_PD_ppm', 'Mo_AAS_ppm'],
        'As_ppm':    ['As_LIP_ppm', 'As_INA_ppm', 'As_MAS_ppm', 'As_MIP_ppm',
                      'As_AIP_ppm', 'As_TD_ppm', 'As_NTD_ppm', 'As_PD_ppm', 'As_AAS_ppm'],
        'Th_ppm':    ['Th_LIP_ppm', 'Th_INA_ppm', 'Th_MAS_ppm', 'Th_MIP_ppm',
                      'Th_AIP_ppm', 'Th_TD_ppm', 'Th_NTD_ppm', 'Th_PD_ppm'],
        'U_ppm':     ['U_LIP_ppm', 'U_INA_ppm', 'U_MAS_ppm', 'U_MIP_ppm',
                      'U_AIP_ppm', 'U_TD_ppm', 'U_NTD_ppm', 'U_PD_ppm'],
        'La_ppm':    ['La_LIP_ppm', 'La_INA_ppm', 'La_MIP_ppm', 'La_AIP_ppm',
                      'La_TD_ppm', 'La_NTD_ppm', 'La_PD_ppm'],
        'Ce_ppm':    ['Ce_LIP_ppm', 'Ce_INA_ppm', 'Ce_MIP_ppm', 'Ce_AIP_ppm',
                      'Ce_TD_ppm', 'Ce_NTD_ppm', 'Ce_PD_ppm'],
        'Nd_ppm':    ['Nd_LIP_ppm', 'Nd_INA_ppm', 'Nd_MIP_ppm',
                      'Nd_TD_ppm', 'Nd_NTD_ppm'],
        'Nb_ppm':    ['Nb_LIP_ppm', 'Nb_XRF_ppm', 'Nb_INA_ppm', 'Nb_MIP_ppm',
                      'Nb_AIP_ppm', 'Nb_TD_ppm', 'Nb_NTD_ppm', 'Nb_PD_ppm'],
        'V_ppm':     ['V_LIP_ppm', 'V_INA_ppm', 'V_MAS_ppm', 'V_MIP_ppm',
                      'V_AIP_ppm', 'V_TD_ppm', 'V_NTD_ppm', 'V_PD_ppm'],
        'W_ppm':     ['W_LIP_ppm', 'W_INA_ppm', 'W_MIP_ppm', 'W_AIP_ppm',
                      'W_TD_ppm', 'W_NTD_ppm', 'W_PD_ppm'],
        'SiO2_pct':  ['SiO2_LIP_%', 'SiO2_XRF_%', 'SiO2_MAS_%', 'SiO2_NTD_%', 'SiO2_TD_%'],
        'Al2O3_pct': ['Al2O3_LIP_%', 'Al2O3_XRF_%', 'Al2O3_MAS_%', 'Al2O3_NTD_%', 'Al2O3_TD_%'],
        'Fe2O3_pct': ['Fe2O3(T)_LIP_%', 'Fe2O3(T)_XRF_%', 'Fe2O3(T)_MAS_%',
                      'Fe2O3(T)_NTD_%', 'Fe2O3(T)_TD_%'],
        'MgO_pct':   ['MgO_LIP_%', 'MgO_XRF_%', 'MgO_MAS_%', 'MgO_NTD_%', 'MgO_TD_%'],
        'CaO_pct':   ['CaO_LIP_%', 'CaO_XRF_%', 'CaO_MAS_%', 'CaO_NTD_%', 'CaO_TD_%'],
        'Na2O_pct':  ['Na2O_LIP_%', 'Na2O_XRF_%', 'Na2O_MAS_%', 'Na2O_NTD_%', 'Na2O_TD_%'],
        'K2O_pct':   ['K2O_LIP_%', 'K2O_XRF_%', 'K2O_MAS_%', 'K2O_NTD_%', 'K2O_TD_%'],
        'TiO2_pct':  ['TiO2_LIP_%', 'TiO2_XRF_%', 'TiO2_NTD_%', 'TiO2_TD_%'],
        'P2O5_pct':  ['P2O5_LIP_%', 'P2O5_XRF_%', 'P2O5_TD_%'],
        'MnO_pct':   ['MnO_LIP_%', 'MnO_XRF_%', 'MnO_MAS_%', 'MnO_NTD_%', 'MnO_TD_%'],
    }

    # parse sentinel values in all relevant columns
    all_src_cols = [c for clist in analyte_map.values() for c in clist if c in df.columns]
    df = parse_cols(df, all_src_cols)

    for target, candidates in analyte_map.items():
        existing = [c for c in candidates if c in df.columns]
        if existing:
            out[target] = df[existing].apply(
                lambda row: coalesce_bc(row, existing), axis=1
            )
        else:
            out[target] = np.nan

    print(f"    BC rows: {len(out):,}")
    return out


# ============================================================
# ONTARIO — MRD 347 (Ring of Fire)
# Multi-row header (header at row index 2)
# Coordinates: UTM Zone 17N NAD83 → WGS84
# ============================================================
def load_ontario(path):
    print(f"  Loading Ontario: {path.split('/')[-1]}")
    df = pd.read_excel(path, header=2)

    # rename messy header columns
    df = df.rename(columns={
        'Easting Collar\n(or outcrop)': 'Easting',
        'Northing Collar\n(or outcrop)': 'Northing',
    })

    df = df.dropna(subset=['Easting', 'Northing'])
    df['Easting'] = pd.to_numeric(df['Easting'], errors='coerce')
    df['Northing'] = pd.to_numeric(df['Northing'], errors='coerce')
    df = df.dropna(subset=['Easting', 'Northing'])

    # UTM Zone 17N NAD83 → WGS84
    tr = Transformer.from_crs("EPSG:26917", "EPSG:4326", always_xy=True)
    lons, lats = tr.transform(df['Easting'].values, df['Northing'].values)

    out = pd.DataFrame()
    out['wgs84_long'] = lons
    out['wgs84_lat'] = lats
    out['province'] = 'Ontario'

    # analyte mapping — Ontario uses bare element names + method suffix columns
    # Primary columns are the first occurrence (no suffix)
    analyte_map = {
        'SiO2_pct':  'SiO2',
        'Al2O3_pct': 'Al2O3',
        'Fe2O3_pct': 'Fe2O3T',
        'MgO_pct':   'MgO',
        'CaO_pct':   'CaO',
        'Na2O_pct':  'Na2O',
        'K2O_pct':   'K2O',
        'TiO2_pct':  'TiO2',
        'P2O5_pct':  'P2O5',
        'MnO_pct':   'MnO',
        'Au_ppb':    'Au',
        'Cu_ppm':    'Cu',
        'Ni_ppm':    'Ni',
        'Co_ppm':    'Co',
        'Li_ppm':    'Li',
        'Cr_ppm':    'Cr',
        'Zn_ppm':    'Zn',
        'Pb_ppm':    'Pb',
        'Mo_ppm':    'Mo',
        'As_ppm':    'As',
        'Th_ppm':    'Th',
        'U_ppm':     'U',
        'La_ppm':    'La',
        'Ce_ppm':    'Ce',
        'Nd_ppm':    'Nd',
        'Nb_ppm':    'Nb',
        'V_ppm':     'V',
        'W_ppm':     'W',
    }

    for target, src in analyte_map.items():
        if src in df.columns:
            out[target] = df[src].apply(parse_value)
        else:
            out[target] = np.nan

    # Au in Ontario is in ppm — convert to ppb
    if 'Au_ppb' in out.columns:
        out['Au_ppb'] = out['Au_ppb'] * 1000

    print(f"    Ontario rows: {len(out):,}")
    return out


# ============================================================
# MANITOBA — GeoFile compilation
# Sheet: "Lithogeochemistry data", skip first row (title)
# Coordinates: Latitude_DD / Longitude_DD (WGS84)
# Gold reported in multiple units — coalesce to ppb
# ============================================================
def load_manitoba(path):
    print(f"  Loading Manitoba: {path.split('/')[-1]}")
    df = pd.read_excel(path, sheet_name="Lithogeochemistry data", skiprows=1)

    # coordinates
    df = df.dropna(subset=['Latitude_DD', 'Longitude_DD'])
    df = df[df['Latitude_DD'].apply(lambda x: str(x).replace('.','').replace('-','').isdigit() 
                                     if pd.notna(x) else False)]
    df['Latitude_DD'] = pd.to_numeric(df['Latitude_DD'], errors='coerce')
    df['Longitude_DD'] = pd.to_numeric(df['Longitude_DD'], errors='coerce')
    df = df.dropna(subset=['Latitude_DD', 'Longitude_DD'])

    out = pd.DataFrame()
    out['wgs84_long'] = df['Longitude_DD']
    out['wgs84_lat'] = df['Latitude_DD']
    out['province'] = 'Manitoba'

    # oxides
    oxide_map = {
        'SiO2_pct':  'SiO2_perc',
        'Al2O3_pct': 'Al2O3_perc',
        'Fe2O3_pct': 'Fe2O3(T)_perc',
        'MgO_pct':   'MgO_perc',
        'CaO_pct':   'CaO_perc',
        'Na2O_pct':  'Na2O_perc',
        'K2O_pct':   'K2O_perc',
        'TiO2_pct':  'TiO2_perc',
        'P2O5_pct':  'P2O5_perc',
        'MnO_pct':   'MnO_perc',
    }

    element_map = {
        'Cu_ppm':  'Cu_ppm',
        'Ni_ppm':  'Ni_ppm',
        'Co_ppm':  'Co_ppm',
        'Li_ppm':  'Li_ppm',
        'Cr_ppm':  'Cr_ppm',
        'Zn_ppm':  'Zn_ppm',
        'Pb_ppm':  'Pb_ppm',
        'Mo_ppm':  'Mo_ppm',
        'As_ppm':  'As_ppm',
        'Th_ppm':  'Th_ppm',
        'U_ppm':   'U_ppm',
        'La_ppm':  'La_ppm',
        'Ce_ppm':  'Ce_ppm',
        'Nd_ppm':  'Nd_ppm',
        'Nb_ppm':  'Nb_ppm',
        'V_ppm':   'V_ppm',
        'W_ppm':   'W_ppm',
    }

    for target, src in {**oxide_map, **element_map}.items():
        if src in df.columns:
            out[target] = df[src].apply(parse_value)
        else:
            out[target] = np.nan

    # Gold: coalesce Au_ppb > Au_ppm*1000 > Au_g/tonne*1000
    if 'Au_ppb' in df.columns:
        au = df['Au_ppb'].apply(parse_value)
    else:
        au = pd.Series(np.nan, index=df.index)

    if 'Au_ppm' in df.columns:
        au_ppm = df['Au_ppm'].apply(parse_value) * 1000
        au = au.where(au.notna(), au_ppm)

    if 'Au_g/tonne' in df.columns:
        au_gt = df['Au_g/tonne'].apply(parse_value) * 1000
        au = au.where(au.notna(), au_gt)

    out['Au_ppb'] = au

    print(f"    Manitoba rows: {len(out):,}")
    return out


# ============================================================
# SASKATCHEWAN — CSV with UTM coordinates
# Negative values = below-detection sentinels → abs/2
# Coordinates: UTM_EASTING / UTM_NORTHING (zone embedded in data)
# ============================================================
def load_saskatchewan(path):
    print(f"  Loading Saskatchewan: {path.split('/')[-1]}")
    df = pd.read_csv(path, low_memory=False)

    # Determine UTM zone from X column range
    # Saskatchewan spans zones 12, 13 — use X/Y columns (projected)
    # X and Y appear to be projected easting/northing
    df = df.dropna(subset=['X', 'Y'])
    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    df = df.dropna(subset=['X', 'Y'])

    # Infer UTM zone from easting (zone 12: ~250k-750k, zone 13: ~250k-750k)
    # SK spans UTM 12N and 13N — use COMPANY_ZONE if available, else estimate
    if 'COMPANY_ZONE' in df.columns:
        zones = pd.to_numeric(df['COMPANY_ZONE'], errors='coerce').fillna(13).astype(int)
    else:
        # Estimate from X: eastings > 500000 in zone 12 are more likely zone 13
        zones = np.where(df['X'] < 400000, 12, 13)

    # Reproject UTM → WGS84
    lons, lats = [], []
    for i, row in df.iterrows():
        zone = int(zones[i]) if hasattr(zones, '__getitem__') else int(zones)
        epsg = 26900 + zone  # NAD83 UTM zone
        try:
            tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
            lon, lat = tr.transform(row['X'], row['Y'])
            lons.append(lon)
            lats.append(lat)
        except Exception:
            lons.append(np.nan)
            lats.append(np.nan)

    out = pd.DataFrame()
    out['wgs84_long'] = lons
    out['wgs84_lat'] = lats
    out['province'] = 'Saskatchewan'

    # analyte mapping — SK uses uppercase column names
    analyte_map = {
        'SiO2_pct':  'SIO2_PCT',
        'Al2O3_pct': 'AL2O3_PCT',
        'Fe2O3_pct': 'FE2O3_PCT',
        'MgO_pct':   'MGO_PCT',
        'CaO_pct':   'CAO_PCT',
        'Na2O_pct':  'NA2O_PCT',
        'K2O_pct':   'K2O_PCT',
        'TiO2_pct':  'TIO2_PCT',
        'P2O5_pct':  'P2O5_PCT',
        'MnO_pct':   'MNO_PCT',
        'Au_ppb':    'AU_PPB',
        'Cu_ppm':    'CU_PPM',
        'Ni_ppm':    'NI_PPM',
        'Co_ppm':    'CO_PPM',
        'Li_ppm':    'LI_PPM',
        'Cr_ppm':    'CR_PPM',
        'Zn_ppm':    'ZN_PPM',
        'Pb_ppm':    'PB_PPM',
        'Mo_ppm':    'MO_PPM',
        'As_ppm':    'AS_PPM',
        'Th_ppm':    'TH_PPM',
        'U_ppm':     'U_PPM',
        'La_ppm':    'LA_PPM',
        'Ce_ppm':    'CE_PPM',
        'Nd_ppm':    'ND_PPM',
        'Nb_ppm':    'NB_PPM',
        'V_ppm':     'V_PPM',
        'W_ppm':     'W_PPM',
    }

    for target, src in analyte_map.items():
        if src in df.columns:
            out[target] = df[src].apply(parse_value)
        else:
            out[target] = np.nan

    out = out.dropna(subset=['wgs84_long', 'wgs84_lat'])
    print(f"    Saskatchewan rows: {len(out):,}")
    return out


# ============================================================
# YUKON — KML file
# Parse <description> HTML table from each Placemark
# Coordinates from <Point><coordinates> tag
# ============================================================
def load_yukon(path):
    print(f"  Loading Yukon: {path.split('/')[-1]}")

    tree = ET.parse(path)
    root = tree.getroot()
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    records = []
    for placemark in root.findall('.//kml:Placemark', ns):
        record = {}

        # coordinates
        coords_el = placemark.find('.//kml:coordinates', ns)
        if coords_el is not None and coords_el.text:
            parts = coords_el.text.strip().split(',')
            if len(parts) >= 2:
                try:
                    record['wgs84_long'] = float(parts[0])
                    record['wgs84_lat'] = float(parts[1])
                except ValueError:
                    continue
        else:
            # fallback: look for LATITUDE_DD and LONGITUDE_DD in description
            desc = placemark.find('kml:description', ns)
            if desc is not None and desc.text:
                lat_m = re.search(r'LATITUDE_DD.*?<td>([\d.\-]+)</td>', desc.text)
                lon_m = re.search(r'LONGITUDE_DD.*?<td>([\d.\-]+)</td>', desc.text)
                if lat_m and lon_m:
                    record['wgs84_lat'] = float(lat_m.group(1))
                    record['wgs84_long'] = float(lon_m.group(1))
                else:
                    continue
            else:
                continue

        # parse description HTML table
        desc = placemark.find('kml:description', ns)
        if desc is not None and desc.text:
            for m in re.finditer(r'<td>([^<]+)</td><td>([^<]*)</td>', desc.text):
                key = m.group(1).strip()
                val = m.group(2).strip()
                record[key] = val

        records.append(record)

    df = pd.DataFrame(records)

    out = pd.DataFrame()
    out['wgs84_long'] = pd.to_numeric(df.get('wgs84_long', df.get('LONGITUDE_DD')), errors='coerce')
    out['wgs84_lat'] = pd.to_numeric(df.get('wgs84_lat', df.get('LATITUDE_DD')), errors='coerce')
    out['province'] = 'Yukon'

    # Yukon KML uses all-caps column names
    analyte_map = {
        'SiO2_pct':  'SIO2',
        'Al2O3_pct': 'AL2O3',
        'Fe2O3_pct': 'FE2O3_T',
        'MgO_pct':   'MGO',
        'CaO_pct':   'CAO',
        'Na2O_pct':  'NA2O',
        'K2O_pct':   'K2O',
        'TiO2_pct':  'TIO2',
        'P2O5_pct':  'P2O5',
        'MnO_pct':   'MNO',
        'Au_ppb':    'AU_PPB',
        'Cu_ppm':    'CU',
        'Ni_ppm':    'NI',
        'Co_ppm':    'CO',
        'Li_ppm':    'LI',
        'Cr_ppm':    'CR',
        'Zn_ppm':    'ZN',
        'Pb_ppm':    'PB',
        'Mo_ppm':    'MO',
        'As_ppm':    'ARS',
        'Th_ppm':    'TH',
        'U_ppm':     'U',
        'La_ppm':    'LA',
        'Ce_ppm':    'CE',
        'Nd_ppm':    'ND',
        'Nb_ppm':    'NB',
        'V_ppm':     'V',
        'W_ppm':     'W',
    }

    for target, src in analyte_map.items():
        if src in df.columns:
            out[target] = df[src].apply(parse_value)
        else:
            out[target] = np.nan

    out = out.dropna(subset=['wgs84_long', 'wgs84_lat'])
    print(f"    Yukon rows: {len(out):,}")
    return out


# ============================================================
# PHYSICS LAYERS
# acoustic_properties.csv → Vp_100_kms (P-wave at 100 MPa)
# densities_and_magnetic.csv → Density_gcc, MagSus_SI
# ============================================================
def load_physics(acoustic_path, density_path):
    print("  Loading physics layers...")

    # --- Acoustic properties ---
    ac = pd.read_csv(acoustic_path)
    ac = ac.dropna(subset=['Latitude', 'Longitude'])
    ac['V_100'] = ac['V_100'].apply(parse_value)  # km/s at 100 MPa
    ac = ac.dropna(subset=['V_100'])
    ac_out = pd.DataFrame({
        'wgs84_long': pd.to_numeric(ac['Longitude'], errors='coerce'),
        'wgs84_lat':  pd.to_numeric(ac['Latitude'],  errors='coerce'),
        'Vp_100_kms': ac['V_100']
    }).dropna()

    # --- Density and magnetic ---
    dm = pd.read_csv(density_path)
    # Use high-precision X/Y over truncated LATITUDE/LONGITUDE
    dm = dm.dropna(subset=['X', 'Y'])
    dm['DENSITY'] = dm['DENSITY'].apply(parse_value)
    dm['MAGSUS']  = dm['MAGSUS'].apply(parse_value)
    # Zero MagSus = fill sentinel, not real measurement
    dm.loc[dm['MAGSUS'] == 0, 'MAGSUS'] = np.nan
    dm_out = pd.DataFrame({
        'wgs84_long':  pd.to_numeric(dm['X'], errors='coerce'),
        'wgs84_lat':   pd.to_numeric(dm['Y'], errors='coerce'),
        'Density_gcc': dm['DENSITY'],
        'MagSus_SI':   dm['MAGSUS']
    }).dropna(subset=['wgs84_long', 'wgs84_lat'])

    print(f"    Acoustic: {len(ac_out):,} rows with Vp_100")
    print(f"    Density/Mag: {len(dm_out):,} rows")
    return ac_out, dm_out


# ============================================================
# SPATIAL THINNING
# One representative sample per 10km cell (EPSG:3978)
# Keep row with most non-NaN analyte values per cell
# ============================================================
def spatial_thin(df, res_m=10000, analyte_cols=None):
    if analyte_cols is None:
        analyte_cols = ALL_ANALYTES

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3978", always_xy=True)
    x3978, y3978 = tr.transform(df['wgs84_long'].values, df['wgs84_lat'].values)

    df = df.copy()
    df['_cx'] = (x3978 / res_m).astype(int)
    df['_cy'] = (y3978 / res_m).astype(int)

    existing_analytes = [c for c in analyte_cols if c in df.columns]
    df['_nonnull'] = df[existing_analytes].notna().sum(axis=1)

    # keep row with most non-null analytes per cell
    idx = df.groupby(['_cx', '_cy'])['_nonnull'].idxmax()
    thinned = df.loc[idx].drop(columns=['_cx', '_cy', '_nonnull'])
    return thinned.reset_index(drop=True)


# ============================================================
# IDW INTERPOLATION
# Inverse distance weighting, k nearest neighbours
# log-transform for positive-skewed geochemical data
# ============================================================
def idw_interpolate(x_known, y_known, z_known, x_pred, y_pred, k=8):
    valid = np.isfinite(z_known) & np.isfinite(x_known) & np.isfinite(y_known)
    x_k = x_known[valid]
    y_k = y_known[valid]
    z_k = z_known[valid]

    if len(x_k) < 2:
        return np.full(len(x_pred), np.nan)

    use_log = (z_k > 0).all()
    z = np.log1p(z_k) if use_log else z_k.copy()

    tree = cKDTree(np.stack([x_k, y_k], axis=1))
    k = min(k, len(x_k))
    pred_coords = np.stack([x_pred, y_pred], axis=1)
    dists, idxs = tree.query(pred_coords, k=k)

    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    with np.errstate(divide='ignore'):
        w = 1.0 / np.where(dists == 0, 1e-10, dists)
    w /= w.sum(axis=1, keepdims=True)

    z_pred = (w * z[idxs]).sum(axis=1)
    return np.expm1(z_pred) if use_log else z_pred


# ============================================================
# BUILD RASTER STACK
# ============================================================
def build_raster_stack(geo_merged, ac_pts, dm_pts):
    print("\nBuilding raster stack...")

    tr     = Transformer.from_crs("EPSG:4326", "EPSG:3978", always_xy=True)
    tr_inv = Transformer.from_crs("EPSG:3978", "EPSG:4326", always_xy=True)

    # Grid extent from data coverage (1st-99th percentile + 100km buffer)
    gx, gy = tr.transform(geo_merged['wgs84_long'].values,
                           geo_merged['wgs84_lat'].values)
    XMIN = np.percentile(gx, 1)  - 100000
    XMAX = np.percentile(gx, 99) + 100000
    YMIN = np.percentile(gy, 1)  - 100000
    YMAX = np.percentile(gy, 99) + 100000
    RES  = 10000  # 10 km

    grid_x = np.arange(XMIN, XMAX + RES, RES, dtype=np.float32)
    grid_y = np.arange(YMIN, YMAX + RES, RES, dtype=np.float32)
    NX, NY = len(grid_x), len(grid_y)
    GX, GY = np.meshgrid(grid_x, grid_y)
    gx_f = GX.ravel()
    gy_f = GY.ravel()
    print(f"  Grid: {NX} x {NY} = {NX*NY:,} pixels")

    # Spatially thin geochemistry to 15km cells before interpolation
    THIN_RES = 15000
    geo_thinned = spatial_thin(geo_merged, res_m=THIN_RES)
    print(f"  Thinned geochemistry: {len(geo_merged):,} → {len(geo_thinned):,} samples")

    src_x, src_y = tr.transform(geo_thinned['wgs84_long'].values,
                                  geo_thinned['wgs84_lat'].values)

    # Interpolate geochemistry layers
    raster = {'x_3978': gx_f, 'y_3978': gy_f}

    for col in ALL_ANALYTES:
        if col not in geo_thinned.columns:
            raster[col] = np.full(len(gx_f), np.nan)
            continue
        z = geo_thinned[col].values.astype(float)
        raster[col] = idw_interpolate(src_x, src_y, z, gx_f, gy_f, k=8)
        print(f"    Interpolated {col}")

    # Interpolate physics layers
    # Vp_100_kms from acoustic
    ac_x, ac_y = tr.transform(ac_pts['wgs84_long'].values, ac_pts['wgs84_lat'].values)
    raster['Vp_100_kms'] = idw_interpolate(
        ac_x, ac_y, ac_pts['Vp_100_kms'].values.astype(float), gx_f, gy_f, k=8
    )
    print("    Interpolated Vp_100_kms")

    # Density_gcc from density/magnetic
    dm_x, dm_y = tr.transform(dm_pts['wgs84_long'].values, dm_pts['wgs84_lat'].values)
    raster['Density_gcc'] = idw_interpolate(
        dm_x, dm_y, dm_pts['Density_gcc'].values.astype(float), gx_f, gy_f, k=8
    )
    print("    Interpolated Density_gcc")

    # MagSus_SI — sparse, use only non-NaN
    mag_valid = dm_pts.dropna(subset=['MagSus_SI'])
    if len(mag_valid) > 10:
        mag_x, mag_y = tr.transform(mag_valid['wgs84_long'].values,
                                     mag_valid['wgs84_lat'].values)
        raster['MagSus_SI'] = idw_interpolate(
            mag_x, mag_y, mag_valid['MagSus_SI'].values.astype(float), gx_f, gy_f, k=8
        )
    else:
        raster['MagSus_SI'] = np.zeros(len(gx_f))
    print("    Interpolated MagSus_SI")

    # Convert projected coords back to WGS84
    wgs_lon, wgs_lat = tr_inv.transform(gx_f, gy_f)
    raster['wgs84_long'] = wgs_lon
    raster['wgs84_lat']  = wgs_lat

    df_raster = pd.DataFrame(raster)

    # Drop pixels outside Canada bounding box
    df_raster = df_raster[
        (df_raster['wgs84_long'] >= -141.0) &
        (df_raster['wgs84_long'] <= -52.0)  &
        (df_raster['wgs84_lat']  >= 41.7)   &
        (df_raster['wgs84_lat']  <= 83.0)
    ].reset_index(drop=True)

    # Fill any remaining NaN with column median
    feature_cols = ALL_ANALYTES + ['Vp_100_kms', 'Density_gcc', 'MagSus_SI']
    df_raster[feature_cols] = df_raster[feature_cols].fillna(
        df_raster[feature_cols].median()
    )

    print(f"  Final raster stack: {df_raster.shape[0]:,} pixels x {df_raster.shape[1]} columns")
    return df_raster


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    DATA_DIR = "./data_raw/"

    # --- Load provincial geochemistry ---
    print("\n=== LOADING PROVINCIAL GEOCHEMISTRY ===")

    bc  = load_bc(
        DATA_DIR + "rock_geochem_dataset1-BC.xlsx",
        DATA_DIR + "rock_geochem_dataset2-BC.xlsx"

    )

    """
    Loads Ontario MRD 347 (Ring of Fire / Abitibi lithogeochemistry).
    
    NOTE: The full Ontario dataset used in this study comprised three MRD files:
      - MRD 123 (Abitibi region)
      - MRD 143 (Abitibi region)
      - MRD 347 (Ring of Fire)
    MRD 123 and MRD 143 are not included in this repository but can be obtained
    from the Ontario Geological Survey (OGS) open data portal at:
    https://www.ontario.ca/page/ontario-geological-survey-publications-and-data
    
    For simplicity and to avoid potential licensing issues, only MRD 347 is included here, as it contains a large number of samples from the Ring of Fire area, which is a key focus of the study. MRD 347 provides a representative sample of Ontario's lithogeochemistry
    while keeping the dataset manageable and free of licensing restrictions.
    """
    on  = load_ontario(DATA_DIR + "MRD_347_Lithogeochemistry_-_Ontario.xlsx")
    mb  = load_manitoba(DATA_DIR + "Lithogeochemistry-Man.xlsx")
    sk  = load_saskatchewan(DATA_DIR + "Lithogeochemistry_AnalysesSask.csv")
    yt  = load_yukon(DATA_DIR + "LithogeoYukon.kml")

    # --- Merge all provinces ---
    print("\n=== MERGING PROVINCES ===")
    geo_merged = pd.concat(
        [bc, on, mb, sk, yt],
        ignore_index=True,
        sort=False
    )

    # Drop rows with missing coordinates
    geo_merged = geo_merged.dropna(subset=['wgs84_long', 'wgs84_lat'])

    # Spatial thinning for overrepresented provinces
    # Manitoba and Saskatchewan can be dense in certain areas
    print("Applying spatial thinning to overrepresented provinces...")
    geo_mb = geo_merged[geo_merged['province'] == 'Manitoba']
    geo_sk = geo_merged[geo_merged['province'] == 'Saskatchewan']
    geo_other = geo_merged[~geo_merged['province'].isin(['Manitoba', 'Saskatchewan'])]

    geo_mb_thin = spatial_thin(geo_mb, res_m=10000)
    geo_sk_thin = spatial_thin(geo_sk, res_m=10000)

    geo_merged = pd.concat(
        [geo_other, geo_mb_thin, geo_sk_thin],
        ignore_index=True
    )

    print(f"\nProvince breakdown (post-thinning):")
    print(geo_merged['province'].value_counts().to_string())
    print(f"Total samples: {len(geo_merged):,}")

    # Save merged geochemistry
    geo_merged.to_csv("./data_combined/Canada_geochemistry_merged.csv", index=False)
    print("\nSaved: Canada_geochemistry_merged.csv")

    # --- Load physics layers ---
    print("\n=== LOADING PHYSICS LAYERS ===")
    ac_pts, dm_pts = load_physics(
        DATA_DIR + "acoustic_properties.csv",
        DATA_DIR + "densities_and_magnetic.csv"
    )

    # --- Build raster stack ---
    raster = build_raster_stack(geo_merged, ac_pts, dm_pts)

    raster.to_csv("./data_combined/canada_raster_stack.csv", index=False)
    print("\nSaved: canada_raster_stack.csv")
    print(f"\nDone. Raster stack shape: {raster.shape}")
    print(raster.describe().round(3).to_string())
