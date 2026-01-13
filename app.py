import streamlit as st
import pandas as pd
import numpy as np
import joblib, json

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("xgboost_model.joblib")
    with open("feature_columns.json", "r") as f:
        feature_cols = json.load(f)
    return model, feature_cols

model, FEATURE_COLS = load_assets()

# ---- Preprocess (Notebook mantığına göre) ----
def prepare_features_from_form(form_dict: dict) -> pd.DataFrame:
    """
    Kullanıcı formundan gelen değerleri alır,
    eğitimdeki feature düzenine uygun tek satırlık DataFrame döner.
    """
    # Temel sayısal alanlar
    row = {
        "Age": form_dict["Age"],
        "Tenure": form_dict["Tenure"],
        "Usage Frequency": form_dict["Usage Frequency"],
        "Support Calls": form_dict["Support Calls"],
        "Payment Delay": form_dict["Payment Delay"],
        "Total Spend": form_dict["Total Spend"],
        "Last Interaction": form_dict["Last Interaction"],
    }

    # Gender map: Male=1, Female=0 (notebook ile aynı)
    row["Gender"] = 1 if form_dict["Gender"] == "Male" else 0

    # Contract Length map: Monthly=0, Quarterly=1, Annual=2 (notebook ile aynı)
    contract_map = {"Monthly": 0, "Quarterly": 1, "Annual": 2}
    row["Contract Length"] = contract_map[form_dict["Contract Length"]]

    # Subscription Type dummy (get_dummies(drop_first=True) mantığı)
    # Referans: Basic => Premium=0, Standard=0
    sub = form_dict["Subscription Type"]
    row["Subscription Type_Premium"] = 1 if sub == "Premium" else 0
    row["Subscription Type_Standard"] = 1 if sub == "Standard" else 0

    df = pd.DataFrame([row])

    # Eğitimdeki kolon sırasına zorla
    # Eksik kolon varsa 0 ekle (güvenlik)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0

    df = df[FEATURE_COLS]
    return df


def predict_proba(df: pd.DataFrame) -> float:
    # Bazı modeller predict_proba verir, bazıları vermez; XGBClassifier genelde verir.
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(df)[0, 1])
    # fallback: decision_function -> sigmoid
    if hasattr(model, "decision_function"):
        z = float(model.decision_function(df)[0])
        return 1 / (1 + np.exp(-z))
    # fallback: predict sadece 0/1 döndürür
    pred = int(model.predict(df)[0])
    return float(pred)


# ---- UI ----
st.title("Customer Churn Prediction")

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Info"])

with tab1:
    colL, colR = st.columns([1.05, 0.95])

    with colL:
        st.subheader("Input Fields")

        Age = st.number_input("Age", min_value=0, max_value=120, value=30)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Tenure = st.number_input("Tenure", min_value=0, value=12)
        Usage_Frequency = st.number_input("Usage Frequency", min_value=0, value=10)
        Support_Calls = st.number_input("Support Calls", min_value=0, value=1)
        Payment_Delay = st.number_input("Payment Delay", min_value=0, value=0)
        Contract_Length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
        Total_Spend = st.number_input("Total Spend", min_value=0.0, value=500.0)
        Last_Interaction = st.number_input("Last Interaction", min_value=0, value=5)
        Subscription_Type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

        threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.50, 0.01)

        form_dict = {
            "Age": Age,
            "Gender": Gender,
            "Tenure": Tenure,
            "Usage Frequency": Usage_Frequency,
            "Support Calls": Support_Calls,
            "Payment Delay": Payment_Delay,
            "Contract Length": Contract_Length,
            "Total Spend": Total_Spend,
            "Last Interaction": Last_Interaction,
            "Subscription Type": Subscription_Type,
        }

        if st.button("Predict", use_container_width=True):
            X_one = prepare_features_from_form(form_dict)
            proba = predict_proba(X_one)
            pred = "Churn" if proba >= threshold else "No Churn"

            st.session_state["last_proba"] = proba
            st.session_state["last_pred"] = pred
            st.session_state["last_x"] = X_one

    with colR:
        st.subheader("Output Fields")
        proba = st.session_state.get("last_proba", None)
        pred = st.session_state.get("last_pred", None)

        if proba is None:
            st.info("Soldan değerleri girip **Predict**’e bas.")
        else:
            st.metric("Churn Probability", f"{proba*100:.2f}%")
            st.metric("Prediction", pred)
            st.caption("Not: Threshold slider’ı ile karar sınırını değiştirebilirsin.")

            with st.expander("Model Input (Debug)"):
                st.dataframe(st.session_state["last_x"])

with tab2:
    st.subheader("Batch Prediction (CSV)")
    st.caption("CSV dosyan, eğitimdeki ham kolonları içermeli (Age, Gender, Contract Length, Subscription Type, ...).")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview")
        st.dataframe(df.head(20))

        batch_threshold = st.slider("Batch Threshold", 0.05, 0.95, 0.50, 0.01)

        if st.button("Run Batch Prediction", use_container_width=True):
            # Beklenen kolonları form preprocess mantığıyla dönüştürme
            # Burada df'nin kolon adlarının notebook ile aynı olduğunu varsayıyoruz.
            # Gerekirse burada rename map ekleriz.

            # Subscription dummy üret
            df2 = df.copy()

            # Gender map
            df2["Gender"] = df2["Gender"].map({"Male": 1, "Female": 0}).fillna(df2["Gender"])

            # Contract Length map
            df2["Contract Length"] = df2["Contract Length"].map({"Monthly": 0, "Quarterly": 1, "Annual": 2}).fillna(df2["Contract Length"])

            # Subscription dummies
            df2["Subscription Type_Premium"] = (df2["Subscription Type"] == "Premium").astype(int)
            df2["Subscription Type_Standard"] = (df2["Subscription Type"] == "Standard").astype(int)

            # Eğer ham "Subscription Type" kolonu modelde yoksa silebiliriz
            if "Subscription Type" in df2.columns and "Subscription Type" not in FEATURE_COLS:
                df2 = df2.drop(columns=["Subscription Type"])

            # Eksik kolonları 0 ekle, sırala
            for c in FEATURE_COLS:
                if c not in df2.columns:
                    df2[c] = 0
            Xb = df2[FEATURE_COLS]

            # Tahmin
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(Xb)[:, 1]
            else:
                probas = np.array([predict_proba(Xb.iloc[[i]]) for i in range(len(Xb))])

            preds = (probas >= batch_threshold).astype(int)

            out = df.copy()
            out["churn_probability"] = probas
            out["churn_prediction"] = np.where(preds == 1, "Churn", "No Churn")

            st.success("Batch prediction completed.")
            st.dataframe(out.head(50))

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                data=csv_bytes,
                file_name="batch_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

with tab3:
    st.subheader("Model Information")
    st.write("Model type:", type(model).__name__)
    st.caption("")
