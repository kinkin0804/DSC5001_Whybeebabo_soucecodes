#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from collections import Counter
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Load the preprocessed data
base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
candidates = [
    base_dir / "apartment_rent_preprocessed.csv",
    base_dir / "data" / "apartment_rent_preprocessed.csv",
    Path.cwd() / "apartment_rent_preprocessed.csv",
    Path.home() / "Desktop" / "5001清洗完毕" / "apartment_rent_preprocessed.csv",
]
file_path = next((path for path in candidates if path.is_file()), None)
if file_path is None:
    raise FileNotFoundError(
        "Place apartment_rent_preprocessed.csv in the same folder as this script."
    )

output_dir = base_dir / "Apartment_EDA_results"
output_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(file_path)
columns = ["price", "square_feet", "bedrooms", "bathrooms"]
required = columns + ["cityname", "state", "amenities"]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
for column in columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("Input file:", file_path)
print("Data shape:", df.shape)
print("\nMissing values:")
print(df[required].isna().sum())

# Full-sample descriptive analysis; no predictive models are fitted here.
# Keep extreme observations. Zeros in room counts include preprocessing fills.
plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "savefig.dpi": 300})
number_format = FuncFormatter(lambda value, position: f"{value:,.0f}")


def save_figure(fig, filename):
    fig.savefig(output_dir / filename, bbox_inches="tight", dpi=300)
    plt.close(fig)


def draw_boxplot(ax, groups, labels, colour, horizontal=False):
    options = {
        "patch_artist": True,
        "medianprops": {"color": "red", "linewidth": 1.2},
        "flierprops": {"marker": ".", "markersize": 2, "alpha": 0.25},
    }
    parameters = inspect.signature(ax.boxplot).parameters
    options["tick_labels" if "tick_labels" in parameters else "labels"] = labels
    if "orientation" in parameters:
        options["orientation"] = "horizontal" if horizontal else "vertical"
    else:
        options["vert"] = not horizontal
    boxes = ax.boxplot(groups, **options)
    for box in boxes["boxes"]:
        box.set_facecolor(colour)
    if horizontal:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(number_format)
        ax.grid(axis="x", alpha=0.2)
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(number_format)
        ax.grid(axis="y", alpha=0.2)


# Descriptive statistics
statistics = df[columns].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
statistics.round(2).to_csv(output_dir / "descriptive_statistics.csv")
print("\nDescriptive statistics:")
print(statistics.round(2))


# Variable distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, column, title, label, colour in [
    (axes[0, 0], "price", "Distribution of Monthly Advertised Rent",
     "Monthly advertised rent (USD)", "steelblue"),
    (axes[0, 1], "square_feet", "Distribution of Floor Area",
     "Floor area (square feet)", "seagreen"),
]:
    values = df[column].dropna()
    ax.hist(values, bins=50, color=colour, edgecolor="white")
    ax.set_title(f"{title} (n={len(values):,})")
    ax.set_xlabel(label)
    ax.set_ylabel("Number of listings")
for ax, column, title, colour in [
    (axes[1, 0], "bedrooms", "Distribution of Recorded Bedroom Counts", "tab:orange"),
    (axes[1, 1], "bathrooms", "Distribution of Recorded Bathroom Counts", "tab:purple"),
]:
    counts = df[column].value_counts().sort_index()
    ax.bar(range(len(counts)), counts.values, color=colour)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{value:g}" for value in counts.index],
                       rotation=45 if column == "bathrooms" else 0)
    ax.set_title(title)
    ax.set_xlabel(f"Number of {column}")
    ax.set_ylabel("Number of listings")
fig.text(0.5, 0.01, "All cleaned listings are included. Bedroom and bathroom zeros "
         "include values filled during preprocessing.", ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.05, 1, 1])
save_figure(fig, "01_variable_distributions.png")


# Distributions on logarithmic axes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, column, title, label, colour in [
    (axes[0], "price", "Distribution of Price", "Monthly advertised rent (USD)", "steelblue"),
    (axes[1], "square_feet", "Distribution of Square Feet", "Floor area (square feet)", "seagreen"),
]:
    values = df.loc[df[column] > 0, column].dropna()
    if values.nunique() < 2:
        raise ValueError(f"{column} requires at least two distinct positive values.")
    ax.hist(values, bins=np.geomspace(values.min(), values.max(), 41),
            color=colour, edgecolor="white")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(number_format)
    ax.set_title(f"{title}\n(n={len(values):,})")
    ax.set_xlabel(f"{label} — log scale")
    ax.set_ylabel("Number of listings")
    print(f"\n{column}: non-positive values excluded from log plot =",
          int((df[column] <= 0).sum()))
fig.tight_layout()
save_figure(fig, "02_distributions_log_scale.png")


# Floor area and advertised rent
scatter_data = df[["square_feet", "price"]].dropna()
positive_data = scatter_data.loc[
    (scatter_data["square_feet"] > 0) & (scatter_data["price"] > 0)
]
correlation = scatter_data["square_feet"].corr(scatter_data["price"])
print("\nPearson correlation between original floor area and rent:", round(correlation, 3))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, log_scale, colour in [
    (axes[0], scatter_data, False, "steelblue"),
    (axes[1], positive_data, True, "seagreen"),
]:
    ax.scatter(data["square_feet"], data["price"], s=8, alpha=0.2,
               color=colour, edgecolors="none")
    suffix = " — Log Scales" if log_scale else ""
    ax.set_title(f"Floor Area and Advertised Rent{suffix}\n(n={len(data):,})")
    ax.set_xlabel("Floor area (square feet)" + (" — log scale" if log_scale else ""))
    ax.set_ylabel("Monthly advertised rent (USD)" + (" — log scale" if log_scale else ""))
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(number_format)
        ax.yaxis.set_major_formatter(number_format)
fig.tight_layout()
save_figure(fig, "03_area_rent_scatterplots.png")


# Rent by bedroom and bathroom counts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, column in zip(axes, ["bedrooms", "bathrooms"]):
    data = df.loc[df["price"] > 0, [column, "price"]].dropna()
    summary = data.groupby(column)["price"].agg(
        n="count", median="median", mean="mean", minimum="min", maximum="max"
    ).sort_index()
    summary.round(2).to_csv(output_dir / f"{column}_rent_summary.csv")
    print(f"\nRent by {column}:")
    print(summary.round(2))
    groups = [data.loc[data[column] == value, "price"].to_numpy()
              for value in summary.index]
    labels = [f"{value:g}\n(n={int(summary.loc[value, 'n']):,})"
              for value in summary.index]
    draw_boxplot(ax, groups, labels, "lightsteelblue")
    ax.set_title(f"Advertised Rent by Number of {column}")
    ax.set_xlabel(f"Number of {column}")
    ax.set_ylabel("Monthly advertised rent (USD) — log scale")
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
fig.text(0.5, 0.02, "Zeros include values filled during preprocessing. Groups with few "
         "observations require cautious interpretation.\nBoxplots are calculated from "
         "original rents; the vertical axis uses a log scale.", ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.12, 1, 1])
save_figure(fig, "04_rent_by_room_counts.png")


# Rent in the ten most represented cities
city_data = df[["cityname", "state", "price"]].copy()
for column in ["cityname", "state"]:
    city_data[column] = city_data[column].astype("string").str.strip().replace("", pd.NA)
city_data = city_data.dropna(subset=["cityname", "state", "price"])
city_data = city_data.loc[city_data["price"] > 0].copy()
city_data["city_state"] = city_data["cityname"] + ", " + city_data["state"]
city_counts = city_data.groupby("city_state").size().rename("n").reset_index()
city_counts = city_counts.sort_values(["n", "city_state"], ascending=[False, True])
top_cities = city_counts.head(10)["city_state"].tolist()
city_summary = city_data.groupby("city_state")["price"].agg(
    n="count", median="median", mean="mean",
    q1=lambda values: values.quantile(0.25),
    q3=lambda values: values.quantile(0.75)
).loc[top_cities]
city_summary.round(2).to_csv(output_dir / "city_rent_summary.csv")
print("\nRent in the ten most represented cities:")
print(city_summary.round(2))
selected_count = int(city_summary["n"].sum())
print("\nValid city records:", len(city_data))
print("Records in selected cities:", selected_count)
print("Share of valid city records:", f"{selected_count / len(city_data):.1%}")
groups = [city_data.loc[city_data["city_state"] == city, "price"].to_numpy()
          for city in top_cities]
labels = [f"{city} (n={int(city_summary.loc[city, 'n']):,})" for city in top_cities]
fig, ax = plt.subplots(figsize=(11, 7))
draw_boxplot(ax, groups, labels, "lightblue", horizontal=True)
ax.invert_yaxis()
ax.set_title("Advertised Rent in the 10 Most Represented Cities")
ax.set_xlabel("Monthly advertised rent (USD) — log scale")
ax.set_ylabel("City and state")
fig.text(0.5, 0.02, "Cities are selected and ordered by listing count. Differences are "
         "not adjusted for property size or room counts.", ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.06, 1, 1])
save_figure(fig, "05_rent_by_city.png")


# Rent by recorded amenities
# An unlisted amenity is not necessarily absent.
def parse_amenities(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    tokens = {token.strip() for token in str(value).split(",") if token.strip()}
    return tokens if tokens else None


amenity_sets = df["amenities"].map(parse_amenities)
amenity_counts = Counter()
for tokens in amenity_sets:
    if tokens is not None:
        amenity_counts.update(tokens)
top_amenities = [name for name, count in sorted(
    amenity_counts.items(), key=lambda item: (-item[1], item[0])
)[:6]]
print("\nMost frequently recorded amenities:")
for amenity in top_amenities:
    print(f"{amenity}: {amenity_counts[amenity]:,} listings")

rows = []
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, amenity in zip(axes.flat, top_amenities):
    status = amenity_sets.map(lambda tokens: "Information missing" if tokens is None
                              else "Recorded" if amenity in tokens else "Not listed")
    groups, labels = [], []
    for group_status in ["Recorded", "Not listed", "Information missing"]:
        rents = df.loc[(status == group_status) & (df["price"] > 0), "price"].dropna()
        groups.append(rents.to_numpy())
        labels.append(f"{group_status}\n(n={len(rents):,})")
        rows.append({"amenity": amenity, "status": group_status, "n": len(rents),
                     "median_rent": rents.median(), "q1": rents.quantile(0.25),
                     "q3": rents.quantile(0.75)})
    draw_boxplot(ax, groups, labels, "#c5dfb3")
    ax.set_title(amenity)
    ax.set_ylabel("Monthly rent (USD) — log scale")
    ax.tick_params(axis="x", labelsize=8)
for ax in list(axes.flat)[len(top_amenities):]:
    ax.set_visible(False)
fig.text(0.5, 0.02, "Not listed does not mean absent. Missing amenity information is "
         "shown separately.\nComparisons are descriptive and are not adjusted for "
         "location, floor area, or room counts.", ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.08, 1, 1])
save_figure(fig, "06_rent_by_recorded_amenities.png")
amenity_summary = pd.DataFrame(rows)
amenity_summary.round(2).to_csv(output_dir / "amenity_rent_summary.csv", index=False)
print("\nAmenity records and rent:")
print(amenity_summary.round(2).to_string(index=False))
print("\nFigures and statistical summaries saved to:", output_dir)
