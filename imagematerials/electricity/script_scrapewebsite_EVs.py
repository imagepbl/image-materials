#%%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from pathlib import Path


path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials
path_external_data_raw = Path(path_base, "data", "raw", "vehicles", "data_raw")

#%%

def parse_car_item(item):
    def get_text_or_none(selector):
        el = item.select_one(selector)
        return el.get_text(strip=True) if el else None

    title_tag = item.select_one("a.title")
    brand_span = title_tag.select_one("span") if title_tag else None
    brand = brand_span.get_text(strip=True) if brand_span else None

    model_span = title_tag.select_one("span.model") if title_tag else None
    if model_span:
        for child in model_span.find_all("span"):
            child.extract()
        model = model_span.get_text(strip=True)
    else:
        model = None

    name = f"{brand} {model}".strip() if brand and model else None

    availability_text = get_text_or_none(".availability")
    sale_start, sale_end = None, None
    if availability_text:
        if "Available to order since" in availability_text:
            match = re.search(r"since (.+)", availability_text)
            if match:
                sale_start = match.group(1)
        elif "Discontinued" in availability_text:
            match = re.search(r"Discontinued \(([^–]+)[–-] ?([^)]+)\)", availability_text)
            if match:
                sale_start, sale_end = match.group(1).strip(), match.group(2).strip()

    # === V2X detection ===
    icons_row2 = item.select_one(".icons-row-2")
    def check_v2x(tooltip_text):
        return any(span.get("data-tooltip") == tooltip_text for span in icons_row2.select("span")) if icons_row2 else False

    v2l = int(check_v2x("Vehicle-2-Load Bi-directional charging possible"))
    v2h = int(check_v2x("Vehicle-2-Home Bi-directional charging possible"))
    v2g = int(check_v2x("Vehicle-2-Grid Bi-directional charging possible"))

    return {
        "Brand": brand,
        "Model": model,
        "Full Name": name,
        "Sale_start": sale_start,
        "Sale_end": sale_end,
        "Drive": item.select_one('[data-tooltip*="Wheel Drive"]').get("data-tooltip", None),
        "Segment": get_text_or_none(".size-d"),
        "Seats": get_text_or_none(".tooltip-wrapper i.seats-5 + span"),
        "Heatpump": "Yes" if item.select_one(".fa-fan") else "No",
        "Range": get_text_or_none(".erange_real"),
        "Efficiency": get_text_or_none(".efficiency"),
        "Weight": get_text_or_none(".weight_p"),
        "0-100 km/h": get_text_or_none(".acceleration_p"),
        "1-Stop Range": get_text_or_none(".long_distance_total"),
        "Battery": get_text_or_none(".battery_p"),
        "Fastcharge": get_text_or_none(".fastcharge_speed_print"),
        "Towing": get_text_or_none(".towweight_p"),
        "Cargo Volume": get_text_or_none(".cargo"),
        "Price/Range": get_text_or_none(".priceperrange_p"),
        "V2L": v2l,
        "V2H": v2h,
        "V2G": v2g,
        "Price_DE": get_text_or_none(".country_de"),
        "Price_NL": get_text_or_none(".country_nl"),
        "Price_UK": get_text_or_none(".country_uk"),
    }

# --- Fetch and parse the page (replace URL below with the actual one) ---
url = "https://ev-database.org"  # Replace with actual URL
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# --- Extract car data ---
items = soup.select(".item-data")
cars = [parse_car_item(item) for item in items]
df = pd.DataFrame(cars)

# --- Bring into right format ---

# Convert numeric columns to float
def to_float(val):
    if isinstance(val, str):
        val = val.replace(",", ".")  # Convert comma to dot (e.g., 5,9 → 5.9)
        num = re.findall(r"[\d.]+", val)
        return float(num[0]) if num else None
    return val

numeric_columns = [
    "Range", "Efficiency", "Weight", "0-100 km/h", "1-Stop Range",
    "Battery", "Fastcharge", "Towing", "Cargo Volume", "Price/Range",
    "Price_DE", "Price_NL", "Price_UK"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].apply(to_float)

# Rename columns to include units

df.rename(columns={
    "Range": "Range_km",
    "Efficiency": "Efficiency_Wh_per_km",
    "Weight": "Weight_kg",
    "0-100 km/h": "Acceleration_0to100kmh_s",
    "1-Stop Range": "Range1Stop_km",
    "Battery": "Battery_kWh",
    "Fastcharge": "Fastcharge_kW",
    "Towing": "Towing_kg",
    "Cargo Volume": "CargoVolume_L",
    "Price/Range": "PricePerRange_EUR_per_km",
    "Price_DE": "Price_DE_EUR",
    "Price_NL": "Price_NL_EUR",
    "Price_UK": "Price_UK_GBP"
}, inplace=True)

#  Convert date strings to datetime
df["Sale_start"] = pd.to_datetime(df["Sale_start"], errors="coerce")
df["Sale_end"] = pd.to_datetime(df["Sale_end"], errors="coerce")

# --- Save to Excel ---
df.to_excel(path_external_data_raw / "2025_ev-database-org_EVcar-models.xlsx", index=False)
