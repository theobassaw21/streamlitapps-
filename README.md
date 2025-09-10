### 🌱 Smart Agri-Advisor

Helping Farmers Make Smarter Decisions.

#### Features
- Simple, clean UI optimized for phones and desktops
- Select crop and region, then tap a big "Get Forecast" button
- Price chart shows past data and a short-term forecast (next ~3 weeks)
- Plain-language advisory to help decide when to sell

#### Tech
- Streamlit, Pandas, NumPy, Altair

#### Run locally
1. Create and activate a virtual environment (optional but recommended)
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Start the app:
```bash
streamlit run app.py
```

The app will open in your browser (usually at `http://localhost:8501`).

#### Notes
- This app uses simple synthetic data and a linear trend to forecast short-term prices. Treat results as guidance only.
# streamlitapps-