"""Generate SNT_metadata_all_layers.json, the all-layers SNT Explorer test file.

Every layer is declared below as (layer_id, dataset_key, filename, column, scale_kind). Text fields
are placeholders; TYPE / SCALE / UNIT_SYMBOL come from SCALES, by the unit traced in the pipeline
code. What is included, what is left out and why: README.md section 8, in this folder.

Run from anywhere; the file is written next to this script (overwriting it):

    python3 docs/schemas/snt_metadata_json/build_snt_metadata_all_layers.py

Then validate it with the snippet in README.md section 5.
"""

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent / "SNT_metadata_all_layers.json"

# scale_kind -> (TYPE, SCALE, UNIT_SYMBOL)
SCALES = {
    "pop_total": ("Threshold", [50000, 100000, 200000, 300000, 400000, 500000], None),  # example POPULATION
    "pop_u5": ("Threshold", [30000, 60000, 90000, 120000], None),  # NER POPULATION_U5
    "pop_pw": ("Threshold", [10000, 20000, 30000, 40000], None),  # NER POPULATION_FE
    "pop_group": ("Threshold", [5000, 10000, 20000, 40000], None),  # generic, smaller sub-populations
    "proportion_rr": ("Threshold", [0.5, 0.8, 0.9, 0.95], None),  # NER REPORTING_RATE (0-1)
    "incidence": ("Threshold", [50, 150, 250, 350, 450, 1000], None),  # example INCIDENCE_* (per 1000)
    "count": ("Threshold", [10, 100, 1000, 10000, 100000], None),  # generic decade bins
    "flag": ("Ordinal", [0, 1], None),
    "block_months": ("Ordinal", [3, 4, 5], None),  # min/max_month_block_size choices
    "month": ("Ordinal", list(range(1, 13)), None),
    "block_share": ("Threshold", [0.5, 0.6, 0.7, 0.8], None),  # 0-1; 0.6 = default seasonality threshold
    "precip_mm": ("Threshold", [25, 50, 100, 200, 300], "mm"),
    "percent": ("Threshold", [25, 50, 75], "%"),
}

# Keys of POPULATION_INDICATOR_DEFINITIONS (identical in all five reference configs)
POP_KIND = {
    "POPULATION": "pop_total",
    "POP_UNDER_5": "pop_u5",
    "POP_PREGNANT_WOMEN": "pop_pw",
    "POP_0_1_Y": "pop_group",
    "POP_1_2_Y": "pop_group",
    "POP_5_10_Y": "pop_group",
    "POP_5_36_M": "pop_group",
    "POP_50_PLUS": "pop_group",
}

# indicator_cols in snt_dhis2_quality_of_care.ipynb (the uppercase count columns)
QOC_COUNTS = ["TEST", "SUSP", "MALTREAT", "CONF", "MALDTH", "MALADM", "ALLADM", "ALLDTH", "ALLOUT", "PRES"]


def build_layers() -> list[tuple[str, str, str, str, str]]:
    """Declare every layer, in pipeline lineage order.

    Returns
    -------
    list[tuple[str, str, str, str, str]]
        One (layer_id, dataset_key, filename, column, scale_kind) tuple per layer.
    """
    layers = []

    # snt_dhis2_formatting -> DHIS2_DATASET_FORMATTED
    for col, kind in POP_KIND.items():
        layers.append((col, "DHIS2_DATASET_FORMATTED", "{COUNTRY_CODE}_population.csv", col, kind))

    # snt_dhis2_population_transformation -> DHIS2_POPULATION_TRANSFORMATION (same filename, other dataset)
    for col, kind in POP_KIND.items():
        layers.append(
            (
                f"{col}_TRANSFORMED",
                "DHIS2_POPULATION_TRANSFORMATION",
                "{COUNTRY_CODE}_population.csv",
                col,
                kind,
            )
        )

    # snt_dhis2_reporting_rate_{dataelement,dataset} -> DHIS2_REPORTING_RATE
    for variant in ["DATAELEMENT", "DATASET"]:
        layers.append(
            (
                f"REPORTING_RATE_{variant}",
                "DHIS2_REPORTING_RATE",
                f"{{COUNTRY_CODE}}_reporting_rate_{variant.lower()}.csv",
                "REPORTING_RATE",
                "proportion_rr",
            )
        )

    # snt_dhis2_incidence -> DHIS2_INCIDENCE
    layers.append(
        ("POPULATION_INCIDENCE", "DHIS2_INCIDENCE", "{COUNTRY_CODE}_incidence.csv", "POPULATION", "pop_total")
    )
    for col in [
        "INCIDENCE_CRUDE",
        "INCIDENCE_ADJ_TESTING",
        "INCIDENCE_ADJ_REPORTING",
        "INCIDENCE_ADJ_CARESEEKING",
    ]:
        layers.append((col, "DHIS2_INCIDENCE", "{COUNTRY_CODE}_incidence.csv", col, "incidence"))

    # snt_dhis2_quality_of_care -> DHIS2_QUALITY_OF_CARE (data_action: imputed | removed)
    for action in ["IMPUTED", "REMOVED"]:
        for col in QOC_COUNTS:
            layers.append(
                (
                    f"{col}_QOC_{action}",
                    "DHIS2_QUALITY_OF_CARE",
                    f"{{COUNTRY_CODE}}_quality_of_care_district_year_{action.lower()}.csv",
                    col,
                    "count",
                )
            )

    # snt_seasonality_{cases,rainfall} -> SNT_SEASONALITY_{CASES,RAINFALL}
    for seasonality, prop_col in [("CASES", "CASES_PROPORTION"), ("RAINFALL", "RAIN_PROPORTION")]:
        dataset = f"SNT_SEASONALITY_{seasonality}"
        filename = f"{{COUNTRY_CODE}}_{seasonality.lower()}_seasonality.csv"
        for col, kind in [
            (f"SEASONALITY_{seasonality}", "flag"),
            (f"SEASONAL_BLOCK_DURATION_{seasonality}", "block_months"),
            (f"SEASONAL_BLOCK_START_MONTH_{seasonality}", "month"),
            (prop_col, "block_share"),
        ]:
            layers.append((col, dataset, filename, col, kind))

    # snt_era5_climate_data -> ERA5_DATASET_CLIMATE (monthly parquet only; no csv twin is published)
    for col in ["MEAN", "MIN", "MAX"]:
        layers.append(
            (
                f"PRECIPITATION_{col}",
                "ERA5_DATASET_CLIMATE",
                "{COUNTRY_CODE}_total_precipitation_monthly.parquet",
                col,
                "precip_mm",
            )
        )

    # snt_worldpop_extract -> WORLDPOP_DATASET_EXTRACT
    layers.append(
        (
            "POPULATION_WORLDPOP",
            "WORLDPOP_DATASET_EXTRACT",
            "{COUNTRY_CODE}_worldpop_population.csv",
            "POPULATION",
            "pop_total",
        )
    )

    # snt_healthcare_access -> SNT_HEALTHCARE_ACCESS
    for col, kind in [
        ("POP_TOTAL", "pop_total"),
        ("POP_COVERED", "pop_total"),
        ("PCT_HEALTH_ACCESS", "percent"),
    ]:
        layers.append(
            (col, "SNT_HEALTHCARE_ACCESS", "{COUNTRY_CODE}_population_covered_health.csv", col, kind)
        )

    return layers


def localized(text: str) -> dict[str, str]:
    """Build a bilingual text field holding the same string in both languages.

    Parameters
    ----------
    text : str
        The text to use for both EN and FR.

    Returns
    -------
    dict[str, str]
        {"EN": text, "FR": text}.
    """
    return {"EN": text, "FR": text}


def build_metadata(layers: list[tuple[str, str, str, str, str]]) -> dict[str, dict]:
    """Turn layer declarations into SNT_metadata.json entries.

    Parameters
    ----------
    layers : list[tuple[str, str, str, str, str]]
        Output of build_layers().

    Returns
    -------
    dict[str, dict]
        The metadata object, keyed by layer id, in declaration order.
    """
    metadata = {}
    for layer_id, dataset, filename, column, kind in layers:
        if layer_id in metadata:
            raise ValueError(f"Duplicate layer id: {layer_id}")
        layer_type, scale, unit_symbol = SCALES[kind]
        placeholder = localized(f"Placeholder text for {layer_id}")
        metadata[layer_id] = {
            "SOURCE_DATA": {
                "DATASET": {"NAME": dataset, "VERSION": "latest"},
                "FILENAME": filename,
                "COLUMN": column,
            },
            "LABEL": placeholder,
            "DESCRIPTION": placeholder,
            "SOURCE": placeholder,
            "UNITS": placeholder,
            "CATEGORY": localized(f"Placeholder text for {dataset}"),
            "TYPE": layer_type,
            "SCALE": scale,
            "UNIT_SYMBOL": unit_symbol,
        }
    return metadata


def main() -> None:
    """Write SNT_metadata_all_layers.json next to this script."""
    metadata = build_metadata(build_layers())
    OUTPUT_PATH.write_text(json.dumps(metadata, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(metadata)} layers -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
