import streamlit as st
import pandas as pd
import boto3
import json
from datetime import datetime
from sqlalchemy import create_engine, text, inspect
import plotly.express as px

# --------------------------------------------------
# 🎯 Streamlit Configuration
# --------------------------------------------------
st.set_page_config(page_title="🏥 Patient Risk Predictor", page_icon="💉", layout="wide")

st.markdown("""
    <style>
        body { background-color: #f8fafc; }
        .main { padding: 1rem 2rem; }
        .title { font-size: 28px; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem; }
        .subheader { color: #0072B1; font-weight: 600; font-size: 20px; }
        .risk-box {
            border-radius: 10px; padding: 1.5rem; margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# ☁️ SageMaker Endpoint Configuration
# --------------------------------------------------
@st.cache_resource
def get_sagemaker_client():
    region = "ap-south-1"  # update if needed
    return boto3.client("sagemaker-runtime", region_name=region)

sagemaker_runtime = get_sagemaker_client()
ENDPOINT_NAME = "diabetes-risk-endpoint"  # SageMaker endpoint name


def predict_with_sagemaker(payload: dict):
    """Send JSON payload to SageMaker endpoint and return the prediction"""
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        return result
    except Exception as e:
        st.error(f"❌ SageMaker prediction failed: {e}")
        return None


# --------------------------------------------------
# 🔐 RDS Connection via AWS Secrets Manager
# --------------------------------------------------
@st.cache_resource
def get_engine():
    secret_name = "patientriskdb/credentials"  # Secret in Secrets Manager
    region_name = "ap-south-1"
    try:
        client = boto3.client("secretsmanager", region_name=region_name)
        secret = json.loads(client.get_secret_value(SecretId=secret_name)["SecretString"])
        db_name = secret.get("dbname", secret.get("dbClusterIdentifier"))
        engine_type = secret.get("engine", "postgresql").replace("postgres", "postgresql")

        engine = create_engine(
            f"{engine_type}://{secret['username']}:{secret['password']}@{secret['host']}/{db_name}"
        )

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()

engine = get_engine()


# --------------------------------------------------
# 🧱 Ensure Table Exists
# --------------------------------------------------
def ensure_table_schema(engine):
    required_columns = {
        "patient_id": "VARCHAR(50)",
        "age": "INT",
        "heart_rate": "INT",
        "bp_systolic": "INT",
        "bp_diastolic": "INT",
        "hemoglobin": "FLOAT",
        "length_of_stay": "INT",
        "gender": "VARCHAR(50)",
        "race": "VARCHAR(50)",
        "diagnosis": "VARCHAR(100)",
        "risk_score": "FLOAT",
        "prediction_label": "VARCHAR(50)",
        "timestamp": "TIMESTAMP"
    }

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS patient_predictions (
        patient_id VARCHAR(50),
        age INT,
        heart_rate INT,
        bp_systolic INT,
        bp_diastolic INT,
        hemoglobin FLOAT,
        length_of_stay INT,
        gender VARCHAR(50),
        race VARCHAR(50),
        diagnosis VARCHAR(100),
        risk_score FLOAT,
        prediction_label VARCHAR(50),
        timestamp TIMESTAMP
    );
    """

    try:
        inspector = inspect(engine)
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()

            existing_cols = [col["name"] for col in inspector.get_columns("patient_predictions")]
            for col, dtype in required_columns.items():
                if col not in existing_cols:
                    conn.execute(text(f"ALTER TABLE patient_predictions ADD COLUMN {col} {dtype};"))
                    conn.commit()
        st.success("✅ Table schema verified.")
    except Exception as e:
        st.error(f"❌ Failed to ensure schema: {e}")
        st.stop()

ensure_table_schema(engine)


# --------------------------------------------------
# 🧭 Sidebar Navigation
# --------------------------------------------------
menu = st.sidebar.radio("Navigation", ["Predict Risk", "Analytics Dashboard"])
st.sidebar.markdown("---")
st.sidebar.info("AI model powered by AWS SageMaker & RandomForest")


# --------------------------------------------------
# 🩺 Prediction Page
# --------------------------------------------------
if menu == "Predict Risk":
    st.markdown("<div class='title'>🏥 Patient Risk Prediction</div>", unsafe_allow_html=True)
    st.markdown("Enter patient details below to predict hospital readmission risk:")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 0, 120, 45)
            heart_rate = st.number_input("Heart Rate", 40, 200, 85)
            hemoglobin = st.number_input("Hemoglobin", 5.0, 20.0, 13.5)
        with col2:
            bp_systolic = st.number_input("BP Systolic", 80, 200, 120)
            bp_diastolic = st.number_input("BP Diastolic", 50, 120, 80)
            length_of_stay = st.number_input("Length of Stay (days)", 0, 100, 3)
        with col3:
            gender = st.selectbox("Gender", ["Male", "Female"])
            race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
            diagnosis = st.text_input("Diagnosis", "Diabetes")

        submitted = st.form_submit_button("🔍 Predict Risk")

    if submitted:
        input_payload = {
            "instances": [{
                "age": age,
                "heart_rate": heart_rate,
                "bp_systolic": bp_systolic,
                "bp_diastolic": bp_diastolic,
                "hemoglobin": hemoglobin,
                "length_of_stay": length_of_stay,
                "gender": gender,
                "race": race,
                "diagnosis": diagnosis
            }]
        }

        st.info("Sending data to SageMaker endpoint...")
        result = predict_with_sagemaker(input_payload)

        if result:
            prediction = result.get("predictions", [0])[0]
            prob = result.get("probabilities", [0.0])[0]

            label = "High Risk" if prediction == 1 else "Low Risk"
            color = "#ff4d4d" if prediction == 1 else "#2ecc71"

            st.markdown(f"""
                <div class='risk-box' style='background-color:{color}1A'>
                    <h3 style='color:{color};'>Prediction: {label}</h3>
                    <p><strong>Risk Probability:</strong> {prob:.2%}</p>
                </div>
            """, unsafe_allow_html=True)

            # Save to RDS
            try:
                with engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO patient_predictions (
                            patient_id, age, heart_rate, bp_systolic, bp_diastolic, hemoglobin,
                            length_of_stay, gender, race, diagnosis, risk_score, prediction_label, timestamp
                        ) VALUES (:pid, :age, :hr, :bps, :bpd, :hgb, :los, :gender, :race, :diag, :score, :label, :ts)
                    """), {
                        "pid": f"P{int(datetime.now().timestamp())}",
                        "age": age, "hr": heart_rate, "bps": bp_systolic, "bpd": bp_diastolic,
                        "hgb": hemoglobin, "los": length_of_stay, "gender": gender, "race": race,
                        "diag": diagnosis, "score": prob, "label": label, "ts": datetime.now()
                    })
                    conn.commit()
                st.success("✅ Prediction logged to database.")
            except Exception as e:
                st.warning(f"⚠️ Failed to save to database: {e}")


# --------------------------
# 📊 ANALYTICS DASHBOARD
# --------------------------
elif menu == "Analytics Dashboard":
    st.markdown('<p class="title">📈 Patient Risk Analytics</p>', unsafe_allow_html=True)

    @st.cache_data(ttl=180)
    def fetch_data():
        query = """
        SELECT patient_id, risk_score, prediction_label, timestamp
        FROM patient_predictions
        ORDER BY timestamp DESC
        LIMIT 500
        """
        return pd.read_sql(query, engine)

    with st.spinner("Fetching analytics data..."):
        try:
            df = fetch_data()
        except Exception as e:
            st.error(f"❌ Database error: {e}")
            st.stop()

    if df.empty:
        st.info("No prediction records found.")
    else:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        col1.metric("📊 Total Predictions", len(df))
        col2.metric("⚠️ High Risk", len(df[df["prediction_label"] == "High Risk"]))
        col3.metric("✅ Low Risk", len(df[df["prediction_label"] == "Low Risk"]))
        col4.metric("📉 Avg Risk Score", f"{df['risk_score'].mean():.2f}")

        fig = px.line(
            df,
            x="timestamp",
            y="risk_score",
            color="prediction_label",
            title="Risk Trends Over Time",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 View Recent Predictions"):
            st.dataframe(df, use_container_width=True)

        if st.button("🔄 Refresh Data"):
            fetch_data.clear()
            st.rerun()

