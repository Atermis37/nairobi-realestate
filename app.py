import joblib
import numpy as np
import pandas as pd
import streamlit as st

LOW_N = 10  

st.set_page_config(page_title="Nairobi Apartment Price Estimator", page_icon="🏙️")


@st.cache_resource
def load_artifact():
    return joblib.load("models/rf_model.joblib")


a = load_artifact()

st.title("Nairobi Apartment Price Estimator")
st.caption(
    "Trained on asking prices from one listing portal (BuyRentKenya), not "
    "sale prices. Treat results as an estimate of the asking-price range."
)

locations = sorted(a["location_counts"].keys())
location = st.selectbox(
    "Location",
    locations,
    index=locations.index("Westlands") if "Westlands" in locations else 0,
)
bedrooms_label = st.selectbox("Bedrooms", ["1", "2", "3", "4+"])
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=8, value=2, step=1)

if st.button("Estimate price"):
    bedrooms = 4 if bedrooms_label == "4+" else int(bedrooms_label)
    tier = a["tier_map"].get(location, "Mid")
    row = pd.DataFrame(
        [
            {
                "bedrooms": bedrooms,
                "bed_bath_ratio": bedrooms / bathrooms,
                "tier_Premium": 1 if tier == "Premium" else 0,
            }
        ]
    )[a["feature_cols"]]

    log_p = a["model"].predict(row)[0]
    est = np.exp(log_p)
    lo, hi = np.exp(log_p + a["resid_lo"]), np.exp(log_p + a["resid_hi"])

    st.metric("Estimated asking price", f"KES {est / 1e6:.1f}M")
    st.write(f"90% range: **KES {lo / 1e6:.1f}M to {hi / 1e6:.1f}M**")

    n = a["location_counts"].get(location, 0)
    if n < LOW_N:
        st.warning(
            f"Only {n} listing(s) in the training data for {location}. Its "
            f"neighbourhood tier ({tier}) is a placeholder assumption, not "
            f"something the data established. Treat this number as indicative only."
        )
    if bedrooms_label == "4+":
        st.info(
            "4+ is priced as a 4-bedroom; larger units had too few listings to model."
        )
    st.caption(
        f"Neighbourhood tier used: {tier} | Location listings in training data: {n} | Model trained on {a['n_train']:,} listings."
    )
