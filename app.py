import streamlit as st
import pandas as pd
import os
import inference

st.set_page_config(page_title="NIDS Live Monitor", layout="wide")

st.title("🛡️ NIDS Live Production Monitor")
st.markdown("Real-time inference engine powered by PyTorch IntrusionDNN.")

# Setup paths (expecting the user's files)
JOBLIB_PATH = "claudefinaldataset.joblib"
MODEL_PATH = "dnn_model.pth"

if not os.path.exists(JOBLIB_PATH) or not os.path.exists(MODEL_PATH):
    st.error(f"Missing required production files!\n\nPlease ensure both exist:\n1. `{JOBLIB_PATH}`\n2. `{MODEL_PATH}`")
    st.stop()

@st.cache_resource
def load_engine():
    return inference.load_system(JOBLIB_PATH, MODEL_PATH)

bundle, model, device = load_engine()

st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.80, help="Flags predictions below this threshold as Uncertain.")

st.sidebar.markdown("---")
st.sidebar.success("✅ Model Online")
st.sidebar.info(f"Loaded Classes: {len(bundle['label_encoder'].classes_)}")

# Dashboard
tab1, tab2 = st.tabs(["Upload Network Capture (CSV)", "Live SOC Stream 🔴"])

with tab1:
    uploaded_file = st.file_uploader("Upload CSV containing live CICFlowMeter outputs", type=["csv"], key="batch")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file, skipinitialspace=True)
        st.write(f"Analyzed {len(df)} packets.")
        
        with st.spinner("Running deep neural network..."):
            X_numpy = inference.preprocess_live_traffic(df, bundle)
            labels, conf = inference.predict(X_numpy, bundle, model, device, conf_threshold)
            
            df["Predicted_Threat"] = labels
            df["Confidence"] = [f"{x:.2f}%" for x in (conf * 100)]
            
            # 1. Create a Risk Sorting mechanism
            # Convert Confidence string back to float for sorting
            df["_conf_float"] = df["Confidence"].str.replace("%", "").astype(float)
            
            # Create a Severity Rank (Threats = 1, Uncertain = 2, Benign = 3)
            def get_severity(threat):
                if threat == "Benign": return 3
                if threat == "Uncertain / Unrecognized": return 2
                return 1
            df["_severity"] = df["Predicted_Threat"].apply(get_severity)
            
            # Sort by Severity (Threats first), then by Confidence (Highest first)
            df = df.sort_values(by=["_severity", "_conf_float"], ascending=[True, False]).drop(columns=["_severity", "_conf_float"])
            
            is_malicious = (df["Predicted_Threat"] != "Benign") & (df["Predicted_Threat"] != "Uncertain / Unrecognized")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Flows", len(df))
            col2.metric("Benign Flows", len(df) - is_malicious.sum() - (df["Predicted_Threat"] == "Uncertain / Unrecognized").sum())
            col3.metric("Detected Threats 🚨", is_malicious.sum())
            
            # Highlight malicious rows in the dataframe
            def highlight_threats(row):
                if row["Predicted_Threat"] not in ["Benign", "Uncertain / Unrecognized"]:
                    return ['background-color: rgba(255, 75, 75, 0.1); color: #ff4b4b; font-weight: bold;'] * len(row)
                return [''] * len(row)
            
            display_cols = ["Predicted_Threat", "Confidence"] + [c for c in df.columns if c not in ("Predicted_Threat", "Confidence")]
            st.dataframe(df[display_cols].style.apply(highlight_threats, axis=1), use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Ranked Threat Report", data=csv, file_name="NIDS_Threat_Report_Ranked.csv")
            
            st.divider()
            st.subheader("🛡️ Automated Incident Response & Remedies")
            
            detected_attacks = df[is_malicious]["Predicted_Threat"].unique()
            if len(detected_attacks) == 0:
                st.success("No active threats detected. Network is secure.")
            else:
                for attack in detected_attacks:
                    with st.expander(f"Remediation Strategy for: {attack}", expanded=True):
                        if "DDoS" in attack or "DoS" in attack:
                            st.write("- **Immediate Action:** Implement aggressive rate-limiting on the affected edge routers.")
                            st.write("- **Network Tier:** Shift traffic through Cloudflare or AWS Shield scrubbing centers.")
                            st.write("- **Firewall:** Drop all malformed packets (e.g., incomplete SYN handshakes) originating from the identified malicious subnets.")
                        elif "Heartbleed" in attack:
                            st.write("- **Immediate Action:** Terminate the affected TLS/SSL connections immediately to prevent memory leakage.")
                            st.write("- **System Tier:** Force an immediate update of the OpenSSL library on the target server.")
                            st.write("- **Post-Incident:** Revoke and reissue all SSL certificates and force a global password reset for active sessions.")
                        elif "Brute Force" in attack or "Patator" in attack:
                            st.write("- **Immediate Action:** Temporarily lock accounts experiencing high authentication failures.")
                            st.write("- **Network Tier:** Implement Fail2Ban policies to auto-block IPs exceeding 5 failed login attempts per minute.")
                            st.write("- **System Tier:** Enforce Multi-Factor Authentication (MFA) and disable password-based SSH access (require RSA keys).")
                        elif "XSS" in attack or "Sql Injection" in attack:
                            st.write("- **Immediate Action:** Enable Web Application Firewall (WAF) strict-mode rule sets for OWASP Top 10.")
                            st.write("- **App Tier:** Ensure all user inputs are being sanitized and parameterized queries are exclusively used in the backend database.")
                        elif "Bot" in attack or "Infiltration" in attack:
                            st.write("- **Immediate Action:** Isolate the infected internal endpoints from the rest of the corporate network (VLAN quarantine).")
                            st.write("- **Network Tier:** Block all outbound traffic to known Command & Control (C2) servers.")
                            st.write("- **System Tier:** Trigger an immediate EDR (Endpoint Detection and Response) malware sweep on the compromised host.")
                        else:
                            st.write("- **Immediate Action:** Quarantine the affected endpoint.")
                            st.write("- **Analysis:** Forward the packet capture (PCAP) to the Tier 2 SOC Analysts for manual inspection.")

with tab2:
    import time
    st.markdown("### 🔴 Live Security Operations Center (SOC) Stream")
    st.write("Upload a CSV to simulate a real-time network traffic feed.")
    
    live_file = st.file_uploader("Upload CSV for Live Stream", type=["csv"], key="live")
    
    if live_file:
        df_live = pd.read_csv(live_file, skipinitialspace=True)
        
        if st.button("▶️ Start Live Capture"):
            placeholder = st.empty()
            alert_placeholder = st.empty()
            
            metrics_placeholder = st.empty()
            benign_count = 0
            threat_count = 0
            
            # Preprocess all at once for speed, but display one by one
            X_numpy = inference.preprocess_live_traffic(df_live, bundle)
            labels, conf = inference.predict(X_numpy, bundle, model, device, conf_threshold)
            
            display_df = pd.DataFrame(columns=["Timestamp/Index", "Threat", "Confidence", "Flow Duration"])
            
            for i in range(len(df_live)):
                pred = labels[i]
                confidence = f"{conf[i]*100:.1f}%"
                
                if pred == "Benign":
                    benign_count += 1
                    status = "✅ Benign"
                    alert_placeholder.empty()
                else:
                    threat_count += 1
                    status = f"🚨 {pred}"
                    alert_placeholder.error(f"**INTRUSION DETECTED:** {pred} ({confidence})")
                
                with metrics_placeholder.container():
                    c1, c2 = st.columns(2)
                    c1.metric("✅ Safe Flows", benign_count)
                    c2.metric("🚨 Blocked Threats", threat_count)
                
                new_row = pd.DataFrame([{
                    "Timestamp/Index": f"Packet #{i+1}",
                    "Threat": status,
                    "Confidence": confidence,
                    "Flow Duration": df_live.iloc[i].get("Flow Duration", 0)
                }])
                display_df = pd.concat([new_row, display_df]).head(15) # Keep last 15
                
                placeholder.dataframe(display_df, use_container_width=True)
                time.sleep(0.4) # Simulate network delay
                
            st.success("Network stream ended.")

