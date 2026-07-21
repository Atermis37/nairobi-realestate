import pandas as pd

df = pd.read_csv("data/processed/listings_clean.csv")
locations = [
    "Brookside",
    "Eastleigh",
    "Embakasi",
    "General Mathenge",
    "Gigiri",
    "Karen",
    "Langata",
    "Loresho",
    "Lower Kabete",
    "Mombasa Road",
    "Muthaiga",
    "Ngong Road",
    "Spring Valley",
    "Valley Arcade",
    "Kiambu Road",
    "Parklands",
    "Westlands",
    "Kilimani",
]
summary = (
    df[df["location"].isin(locations)]
    .groupby("location")["price"]
    .agg(["median", "count"])
)
print(summary.sort_values("median", ascending=False))
