
import streamlit as st
import requests
import datetime
import numpy as np
import pandas as pd
import pickle, os

st.set_page_config(page_title="FreightIQ | Cost Predictor",
                   page_icon="🚛", layout="wide")

st.markdown("""
<style>
.main-header{font-size:28px;font-weight:700;color:#1a1a2e;}
.sub-header{font-size:13px;color:#666;margin-top:-8px;}
.badge-dry{background:#dcfce7;color:#166534;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;}
.badge-rainy{background:#dbeafe;color:#1e40af;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;}
.badge-heavy{background:#fef9c3;color:#854d0e;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;}
.factor-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f0f0f0;font-size:13px;}
.footer-note{font-size:11px;color:#aaa;margin-top:24px;text-align:center;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
SL_CITIES = {
    "Colombo":      {"lat": 6.9271,  "lon": 79.8612},
    "Kandy":        {"lat": 7.2906,  "lon": 80.6337},
    "Galle":        {"lat": 6.0535,  "lon": 80.2210},
    "Trincomalee":  {"lat": 8.5874,  "lon": 81.2152},
    "Jaffna":       {"lat": 9.6615,  "lon": 80.0255},
    "Matara":       {"lat": 5.9549,  "lon": 80.5550},
    "Anuradhapura": {"lat": 8.3114,  "lon": 80.4037},
    "Kurunegala":   {"lat": 7.4867,  "lon": 80.3647},
    "Ratnapura":    {"lat": 6.6828,  "lon": 80.3992},
    "Badulla":      {"lat": 6.9934,  "lon": 81.0550},
    "Hambantota":   {"lat": 6.1241,  "lon": 81.1185},
    "Negombo":      {"lat": 7.2083,  "lon": 79.8358},
    "Gampaha":      {"lat": 7.0917,  "lon": 80.0000},
    "Kalutara":     {"lat": 6.5854,  "lon": 79.9607},
    "Vavuniya":     {"lat": 8.7514,  "lon": 80.4972},
    "Batticaloa":   {"lat": 7.7102,  "lon": 81.6924},
    "Puttalam":     {"lat": 8.0362,  "lon": 79.8283},
    "Matale":       {"lat": 7.4675,  "lon": 80.6234},
    "Nuwara Eliya": {"lat": 6.9497,  "lon": 80.7891},
    "Polonnaruwa":  {"lat": 7.9403,  "lon": 81.0188},
    "Monaragala":   {"lat": 6.8728,  "lon": 81.3507},
    "Kegalle":      {"lat": 7.2513,  "lon": 80.3464},
    "Ampara":       {"lat": 7.2980,  "lon": 81.6747},
}
KM_RATES    = {0: 86.0,  1: 110.0, 2: 227.0}
FUEL_CONS   = {0: 9.0,   1: 15.0,  2: 23.0}
VH_NAMES    = {0: "Small (7T)", 1: "Medium (14T)", 2: "Large (24T)"}
VH_AVG_COST = {0: 16912, 1: 35319, 2: 71413}
AVG_KM      = 276.93
URBAN_CITIES = {"Colombo","Negombo","Gampaha","Dehiwala",
                "Moratuwa","Kelaniya","Kaduwela","Maharagama"}

# ── Load model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("rf_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("feature_cols.pkl", "rb") as f:
            cols = pickle.load(f)
        return model, cols
    except:
        return None, None

rf_model, feature_cols = load_model()

# ── API Functions ─────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_road_distance(orig, dest):
    o, d = SL_CITIES[orig], SL_CITIES[dest]
    url = (f"http://router.project-osrm.org/route/v1/driving/"
           f"{o['lon']},{o['lat']};{d['lon']},{d['lat']}"
           f"?overview=false")
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            route = r.json()["routes"][0]
            km = round(route["distance"] / 1000, 1)
            mins = round(route["duration"] / 60)
            return {"one_way_km": km, "round_trip_km": round(km*2,1),
                    "duration_min": mins, "source": "OSRM (OpenStreetMap)"}
    except:
        pass
    import math
    dlat = math.radians(d["lat"]-o["lat"])
    dlon = math.radians(d["lon"]-o["lon"])
    a = math.sin(dlat/2)**2 + math.cos(math.radians(o["lat"]))*math.cos(math.radians(d["lat"]))*math.sin(dlon/2)**2
    km = round(6371*2*math.asin(math.sqrt(a))*1.3, 1)
    return {"one_way_km": km, "round_trip_km": round(km*2,1),
            "duration_min": round(km), "source": "Estimated (OSRM unavailable)"}

@st.cache_data(ttl=1800)
def get_weather(lat, lon, date_str):
    today = datetime.date.today()
    target = datetime.date.fromisoformat(date_str)
    days_ahead = (target - today).days
    if days_ahead >= 0:
        url = "https://api.open-meteo.com/v1/forecast"
        src = f"Open-Meteo Forecast ({days_ahead}d ahead)"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        src = "Open-Meteo Archive (historical)"
    params = {"latitude": lat, "longitude": lon,
              "daily": ["precipitation_sum","temperature_2m_max",
                        "temperature_2m_mean","precipitation_hours"],
              "start_date": date_str, "end_date": date_str,
              "timezone": "Asia/Colombo"}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            d = r.json().get("daily", {})
            rain  = float(d.get("precipitation_sum",[0])[0] or 0)
            tmax  = float(d.get("temperature_2m_max",[30])[0] or 30)
            tmean = float(d.get("temperature_2m_mean",[28])[0] or 28)
            rh    = float(d.get("precipitation_hours",[0])[0] or 0)
            label = ("HEAVY RAIN" if rain>25 else "RAINY" if rain>5 else "DRY")
            return {"rainfall_mm":round(rain,1),"temp_max_c":round(tmax,1),
                    "temp_mean_c":round(tmean,1),"rain_hours":round(rh,1),
                    "is_rainy_day":1 if rain>5 else 0,
                    "is_heavy_rain":1 if rain>25 else 0,
                    "is_hot_day":1 if tmax>33 else 0,
                    "label":label,"source":src,"days_ahead":days_ahead}
    except:
        pass
    return {"rainfall_mm":0,"temp_max_c":30,"temp_mean_c":28,"rain_hours":0,
            "is_rainy_day":0,"is_heavy_rain":0,"is_hot_day":0,
            "label":"DRY (API unavailable)","source":"Fallback","days_ahead":days_ahead}

def get_season(m):
    if m in [5,6,7,8,9]: return 2
    if m in [12,1,2]: return 1
    return 0

def get_complexity(km):
    if km<50: return 3
    if km<150: return 2
    if km<500: return 1
    return 0

def predict(dist_km, weight, cbm, vcat, fuel_price, weather, trip_date, dest, trip_type):
    eff_km = dist_km * (2 if trip_type=="Round trip" else 1)
    m = trip_date.month
    season = get_season(m)
    fc = FUEL_CONS[vcat]
    est_fuel = (eff_km/100)*fc*fuel_price
    is_urban = 1 if dest in URBAN_CITIES else 0
    cplx = get_complexity(eff_km)
    is_monsoon = 1 if season>0 else 0

    if rf_model is not None and feature_cols is not None:
        feat = {
            "approved_km": eff_km, "weight": weight, "cbm": cbm,
            "vehicle_cat": vcat, "month": m, "quarter": (m-1)//3+1,
            "day_of_week": trip_date.weekday(), "is_weekend": 1 if trip_date.weekday()>=5 else 0,
            "season": season, "is_monsoon": is_monsoon,
            "rainfall_mm": weather["rainfall_mm"], "temp_max_c": weather["temp_max_c"],
            "rain_hours": weather["rain_hours"], "is_rainy_day": weather["is_rainy_day"],
            "is_heavy_rain": weather["is_heavy_rain"], "is_hot_day": weather["is_hot_day"],
            "fuel_consumption_per100": fc, "est_fuel_cost": est_fuel,
            "vh_cost_per_km": VH_AVG_COST[vcat]/max(AVG_KM,1),
            "fuel_cost_ratio": min(est_fuel/max(VH_AVG_COST[vcat],1),1),
            "is_urban": is_urban, "road_complexity": cplx,
            "is_peak_day": 1 if trip_date.weekday()<5 else 0,
            "fuel_price_lkr": fuel_price,
            "distance_sq": eff_km**2,
            "weight_distance": (weight*eff_km)/1000,
            "km_gap": 0, "km_gap_pct": 0,
        }
        X = pd.DataFrame([feat])[feature_cols]
        ml_cost = float(rf_model.predict(X)[0])
        model_used = "✅ Random Forest (trained model)"
    else:
        ml_cost = est_fuel*2.1 + eff_km*38 + cbm*520 + weight*0.9
        if is_monsoon: ml_cost *= 1.04
        if weather["is_rainy_day"]: ml_cost *= 1.02
        if weather["is_heavy_rain"]: ml_cost *= 1.04
        if is_urban: ml_cost *= 1.03
        if cplx==3: ml_cost *= 1.08
        ml_cost = ml_cost*0.70 + (eff_km*KM_RATES[vcat])*0.30
        model_used = "⚠️ Simulation (save model first — see below)"

    trad = KM_RATES[vcat]*eff_km
    delta = (ml_cost-trad)/trad*100
    return {"ml_cost":round(ml_cost),"trad_cost":round(trad),
            "est_fuel":round(est_fuel),"delta_pct":round(delta,1),
            "eff_km":eff_km,"model_used":model_used,
            "season_label":["Inter-monsoon","NE Monsoon","SW Monsoon"][season]}

# ── UI ────────────────────────────────────────────────────────────
st.markdown("<p class='main-header'>🚛 FreightIQ — Real-Time Freight Cost Predictor</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Jayathunga Kamkanamge Ridma Sanka &nbsp;·&nbsp; ST87628 &nbsp;·&nbsp; Transport and Telecommunication Institute, Riga &nbsp;·&nbsp; 2026</p>",
            unsafe_allow_html=True)
st.markdown("<p class='sub-header' style='margin-top:2px;'>Sri Lanka Road Freight &nbsp;·&nbsp; Random Forest Model &nbsp;·&nbsp; R²=0.9961 &nbsp;·&nbsp; MAPE=3.54%</p>",
            unsafe_allow_html=True)

if rf_model is None:
    st.warning("⚠️ rf_model.pkl not found — using simulation mode. For full RF predictions, re-run this cell after Cell 7.")

st.markdown("---")
col_form, col_res = st.columns([1, 1.3], gap="large")

with col_form:
    st.subheader("Trip Details")
    c1, c2 = st.columns(2)
    with c1:
        origin = st.selectbox("Origin", list(SL_CITIES.keys()), index=0)
    with c2:
        opts = [c for c in SL_CITIES.keys() if c != origin]
        destination = st.selectbox("Destination", opts,
                                   index=opts.index("Kandy") if "Kandy" in opts else 0)

    trip_type = st.radio("Trip type", ["One way","Round trip"], horizontal=True)

    c3, c4 = st.columns(2)
    with c3:
        vehicle = st.selectbox("Vehicle", list(VH_NAMES.values()))
        vcat = [k for k,v in VH_NAMES.items() if v==vehicle][0]
    with c4:
        trip_date = st.date_input("Trip date",
                                  value=datetime.date.today()+datetime.timedelta(days=1),
                                  min_value=datetime.date(2020,1,1),
                                  max_value=datetime.date.today()+datetime.timedelta(days=16))

    c5, c6 = st.columns(2)
    with c5:
        weight = st.number_input("Weight (kg)", 100, 25000, 2000, 100)
    with c6:
        cbm = st.number_input("Volume (CBM)", 0.1, 72.0, 6.0, 0.5)

    fuel_price = st.slider("Fuel price (LKR/L)", 250, 500, 283, 1)
    st.caption("💡 CPC diesel: LKR 283/L (Oct 2024)")

    btn = st.button("⚡ Predict Cost", type="primary", use_container_width=True)

with col_res:
    if btn:
        with st.spinner("Fetching road distance + weather..."):
            dist_data = get_road_distance(origin, destination)
            dist_km = dist_data["one_way_km"]
            coords = SL_CITIES[destination]
            weather = get_weather(coords["lat"], coords["lon"], trip_date.isoformat())
            result = predict(dist_km, weight, cbm, vcat, fuel_price,
                             weather, trip_date, destination, trip_type)

        st.subheader(f"{origin} → {destination}")
        wl = weather["label"]
        bc = ("badge-heavy" if "HEAVY" in wl else
              "badge-rainy" if "RAINY" in wl else "badge-dry")
        st.markdown(f'<span class="{bc}">{wl}</span> &nbsp;'
                    f'<span style="font-size:12px;color:#888">'
                    f'{weather["rainfall_mm"]}mm · {weather["temp_max_c"]}°C · {weather["source"]}</span>',
                    unsafe_allow_html=True)
        st.markdown("")

        m1,m2,m3 = st.columns(3)
        sign = "+" if result["delta_pct"]>=0 else ""
        with m1:
            st.metric("ML Prediction", f"LKR {result['ml_cost']:,.0f}",
                      f"{sign}{result['delta_pct']}% vs flat rate")
        with m2:
            st.metric("Traditional", f"LKR {result['trad_cost']:,.0f}",
                      f"LKR {KM_RATES[vcat]}/km × {result['eff_km']} km", delta_color="off")
        with m3:
            st.metric("Est. Fuel", f"LKR {result['est_fuel']:,.0f}",
                      f"{FUEL_CONS[vcat]}L/100km", delta_color="off")

        st.markdown("#### Cost comparison")
        chart_df = pd.DataFrame({
            "Method": ["ML Prediction","Traditional","Est. Fuel"],
            "LKR":    [result["ml_cost"],result["trad_cost"],result["est_fuel"]]
        })
        st.bar_chart(chart_df.set_index("Method"), height=200, color="#2563eb")

        st.markdown("#### Trip factors")
        factors = {
            "Road distance":  f"{dist_km} km one-way · {dist_data['round_trip_km']} km round-trip",
            "Duration est.":  f"{dist_data['duration_min']} min",
            "Effective KM":   f"{result['eff_km']} km ({trip_type})",
            "Vehicle":        VH_NAMES[vcat],
            "Season":         result["season_label"],
            "Weather":        f"{wl} ({weather['rainfall_mm']}mm · {weather['temp_max_c']}°C)",
            "Fuel price":     f"LKR {fuel_price}/L",
            "Distance API":   dist_data["source"],
        }
        for k,v in factors.items():
            st.markdown(f'<div class="factor-row"><span style="color:#666">{k}</span>'
                        f'<span style="font-weight:500">{v}</span></div>',
                        unsafe_allow_html=True)

        st.markdown("")
        st.caption(result["model_used"])
        st.caption("MAPE 3.54% · R² 0.9961 · Thesis ST87628")
    else:
        st.info("👈 Fill in trip details and click **Predict Cost**")
        st.markdown("""
        **Features:**
        - 🛣️ Actual road distance via **OSRM** (OpenStreetMap) — not straight-line
        - 🌦️ **Weather forecast** for future trips (Open-Meteo, up to 16 days)
        - 📅 **Historical weather** for past dates (Archive API)
        - 🤖 **Random Forest model** prediction (R²=0.9961, MAPE=3.54%)
        - 🔄 **Round-trip** cost support (empty return leg included)
        """)

st.markdown("---")
st.markdown('<p class="footer-note">Thesis: Real-Time Cost Prediction · ST87628 · TTI Riga · 2026</p>',
            unsafe_allow_html=True)
