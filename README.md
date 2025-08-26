# Zwende Nameplates — Sales & Support Agent (Streamlit)

Live chat assistant for Zwende **Nameplates**:
- Detects intents: Pricing, Availability, Customization, Shipping, Similar, Order Support, Discounts.
- Uses `zwende_nameplates_flat.csv` (flattened: one row per variant).
- Hosted via Streamlit Cloud or Hugging Face Spaces.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
