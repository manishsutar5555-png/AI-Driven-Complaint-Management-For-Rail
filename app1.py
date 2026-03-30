import streamlit as st
import datetime
import os
import pandas as pd
import tempfile
import plotly.express as px
from streamlit_option_menu import option_menu
# Try importing PyPDF2 (optional). If missing, PDFs will be treated as "General Issue"
try:
    import PyPDF2
    _HAVE_PYPDF2 = True
except Exception:
    _HAVE_PYPDF2 = False

# === Import AI Modules ===
from sentimentclassification import predict_priority   # urgency classifier
from department_classification import predict_department as classify_input # department classifier
from image_classifier import classify_image
from video_classifier import classify_video
from translator import translate_to_english

# === Reply Generator === (unchanged)
def generate_reply(department: str, priority: str) -> str:
    if department == "Coach - Maintenance/Facilities":
        return "🔧 Your coach maintenance issue has been recorded. Our team will attend it at the next station."
    elif department == "Electrical Equipment":
        return "💡 We have noted the electrical issue. Technicians will resolve it at the earliest stop."
    elif department == "Medical Assistance":
        return "🚑 Medical help has been prioritized. Please remain calm, our staff will assist you shortly."
    elif department == "Catering / Vending Services":
        return "🍴 Thank you for reporting the catering issue. The pantry team has been informed."
    elif department == "Water Availability":
        return "🚰 We regret the inconvenience. Water refilling will be ensured at the next halt."
    elif department == "Security":
        return "🛡️ Your security concern is taken seriously. The onboard security team has been alerted."
    elif department == "Coach - Cleanliness":
        return "🧹 Cleanliness issue noted. Cleaning staff will attend to this at the next major station."
    elif department == "Staff Behaviour":
        return "👥 Thank you for reporting. The staff concern will be reviewed by the supervisor."
    elif department == "Punctuality":
        return "⏱️ We regret the delay. The control office has been informed."
    else:
        if priority.lower() == "high":
            return "⚠️ Your high-priority complaint has been registered. Action will be taken immediately."
        else:
            return "✅ Your complaint has been registered. Our team will review and take necessary action."

# === Data Storage Helper ===
CSV_FILE = "complaints.csv"

def save_complaint(data: dict):
    data["PNR"] = str(data["PNR"])
    df_new = pd.DataFrame([data])
    if os.path.exists(CSV_FILE):
        df_old = pd.read_csv(CSV_FILE, dtype=str)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(CSV_FILE, index=False)

def load_complaints():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE, dtype=str)
    else:
        return pd.DataFrame()

# === PDF Extractor ===
def extract_text_from_pdf(file_path: str) -> str:
    if not _HAVE_PYPDF2:
        return "General Issue"
    try:
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip() or "General Issue"
    except Exception:
        return "General Issue"

# === Helper for media classification ===
def extract_caption_from_result(results):
    if results is None:
        return "General Issue"
    if isinstance(results, str):
        return results
    if isinstance(results, (list, tuple)) and len(results) > 0:
        first = results[0]
        if isinstance(first, (list, tuple)) and len(first) > 0:
            return str(first[0])
        return str(first)
    return str(results)

# === Streamlit Config & Global Styling ===
st.set_page_config(page_title="RailMadad AI Portal", layout="wide")

# Background + UI theming
# === Background Styling & Text Colors ===
# === Background Styling with Overlay Cards ===
page_bg_img = """
<style>
/* App background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://plus.unsplash.com/premium_photo-1661937674312-89f00c7e44e7?q=80&w=1176&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
    color: #111111;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.95);
    color: #000000;
}

/* Main content cards */
.block-container {
    background-color: rgba(255,255,255,0.85);  /* white overlay */
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

/* Headers */
h1, h2, h3, h4 {
    color: #0d47a1 !important;
    font-weight: 700;
}

/* Paragraphs, labels, etc. */
p, label, span, .stMarkdown {
    color: #212121 !important;
    font-size: 1rem;
}

/* Buttons */
.stButton button {
    background: linear-gradient(to right, #1565c0, #1e88e5);
    color: white !important;
    border-radius: 8px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    border: none;
}
.stButton button:hover {
    background: linear-gradient(to right, #0d47a1, #1565c0);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# === Sidebar Navigation ===
# Sidebar
with st.sidebar:
    st.image(
        "https://railmadad.indianrailways.gov.in/madad/final/images/logog20.png",
        width=120
    )
    st.markdown("<h2 style='color:white;'>🚆 RailMadad</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:white;'><b>For Inquiry, Assistance & Grievance Redressal</b></p>", unsafe_allow_html=True)

    # Sidebar Menu
    menu = option_menu(
        menu_title=None,  # Hide default title
        options=[
            "Home",
            "Submit Complaint",
            "Track Your Concern",
            "Suggestions",
            "Admin Dashboard"
        ],
        icons=[
            "house",           # Home
            "file-earmark-text",  # Complaint
            "search",          # Track
            "lightbulb",       # Suggestions
            "gear"             # Admin
        ],
        menu_icon="cast",      # optional icon for whole menu
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#0d47a1"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "2px",
                "color": "white",
                "border-radius": "8px",
                "padding": "8px",
            },
            "nav-link-selected": {"background-color": "white", "color": "#0d47a1"},
        }
    )

# --- Pages ---

# Home
if menu == "Home":
    st.markdown("## 🌟 Welcome to RailMadad AI Complaint Management")
    st.info("👉 Submit your complaints, get AI-driven prioritization, department classification & instant replies.")
    st.image("rail_home.png", use_container_width=True)

# Submit Complaint
elif menu == "Submit Complaint":
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
        st.session_state.reply = ""

    if not st.session_state.submitted:
        st.markdown("## 📝 Submit Your Complaint")
        st.caption("Please fill in the details carefully. Fields marked with * are mandatory.")

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                phone = st.text_input("📱 Mobile No.*")
                prn = st.text_input("🎟️ PNR Number*")
                incident_date = st.date_input("📅 Incident Date", datetime.date.today())
            with col2:
                complaint_type = st.selectbox("Category*", [
                    "Coach Maintenance", "Cleanliness", "Catering", "Security", "Ticketing", "Other"
                ])
                complaint_subtype = st.text_input("Sub Type (optional)")
                uploaded_file = st.file_uploader("📎 Upload File (PDF/JPG/PNG/MP4 up to 5MB)",
                                                 type=["pdf","jpg","jpeg","png","mp4"])

            complaint_text = st.text_area("📝 Complaint Description (optional)", height=150)

        if st.button("🚀 Submit Complaint"):
            if not phone.isdigit() or len(phone) != 10:
                st.error("⚠️ Please enter a valid 10-digit phone number!")
            elif not prn:
                st.error("⚠️ Please enter a valid PNR number!")
            elif not complaint_text and not uploaded_file:
                st.error("⚠️ Please provide either complaint text or upload an image/video/PDF!")
            else:
                try:
                    # determine source
                    if complaint_text:
                        text_for_ai = complaint_text
                    elif uploaded_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            temp_path = tmp.name
                        try:
                            if uploaded_file.type == "application/pdf":
                                text_for_ai = extract_text_from_pdf(temp_path)
                            elif uploaded_file.type.startswith("image"):
                                results = classify_image(temp_path)
                                text_for_ai = extract_caption_from_result(results)
                            elif uploaded_file.type.startswith("video"):
                                results = classify_video(temp_path)
                                text_for_ai = extract_caption_from_result(results)
                            else:
                                text_for_ai = "General Issue"
                        finally:
                            try: os.remove(temp_path)
                            except Exception: pass
                    else:
                        text_for_ai = "General Issue"

                    translated_text = translate_to_english(text_for_ai)
                    priority = predict_priority(translated_text)
                    department = classify_input(translated_text)
                    custom_reply = generate_reply(department, priority)

                    complaint_data = {
                        "PNR": str(prn),
                        "Phone": phone,
                        "Date": str(incident_date),
                        "Type": complaint_type,
                        "SubType": complaint_subtype,
                        "Original_Complaint": complaint_text,
                        "Translated_Complaint": translated_text,
                        "Priority": priority,
                        "Department": department,
                        "Reply": custom_reply,
                        "Status": "In Progress"
                    }
                    save_complaint(complaint_data)

                    st.session_state.submitted = True
                    st.session_state.reply = custom_reply

                except Exception as e:
                    st.error(f"❌ Error while processing complaint: {e}")

    else:
        st.success("✅ Complaint submitted successfully!")
        st.markdown("### 🙏 Thank you for your feedback!")
        st.info("Our team has received your complaint. You will be updated soon.")
        st.success(f"📩 Custom Reply: {st.session_state.reply}")

# Track
elif menu == "Track Your Concern":
    st.markdown("## 🔍 Track Complaint")
    st.caption("Enter your PNR to see the status of your complaint.")
    track_prn = st.text_input("🎟️ Enter your PNR")
    if st.button("📊 Track"):
        df = load_complaints()
        if not df.empty and track_prn.strip() in df["PNR"].astype(str).values:
            row = df[df["PNR"].astype(str) == track_prn.strip()].iloc[-1]
            st.success(f"""
            **PNR:** {row['PNR']}  
            **Phone:** {row['Phone']}  
            **Date:** {row['Date']}  
            **Priority:** {row['Priority']}  
            **Department:** {row['Department']}  
            **Status:** {row['Status']}  
            """)
        else:
            st.warning("❌ No complaint found for this PNR.")

# Suggestions
elif menu == "Suggestions":
    st.markdown("## 💡 Suggestions")
    st.caption("Help us improve! Share your suggestions below.")
    suggestion_text = st.text_area("🗒️ Your Suggestion:")

    if st.button("📨 Submit Suggestion"):
        if suggestion_text.strip() == "":
            st.warning("⚠️ Please enter a suggestion before submitting.")
        else:
            # Save suggestion to CSV
            SUGGESTIONS_FILE = "suggestions.csv"
            suggestion_data = {
                "Date": str(datetime.date.today()),
                "Suggestion": suggestion_text
            }
            df_new = pd.DataFrame([suggestion_data])
            if os.path.exists(SUGGESTIONS_FILE):
                df_old = pd.read_csv(SUGGESTIONS_FILE, dtype=str)
                df_all = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_all = df_new
            df_all.to_csv(SUGGESTIONS_FILE, index=False)

            st.success("✅ Thank you for your feedback!")


# Admin Dashboard
elif menu == "Admin Dashboard":
    st.markdown("## 🔐 Admin Login")

    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False

    with st.form("login_form"):
        login_user = st.text_input("👤 Username", key="login_user")
        login_pwd = st.text_input("🔑 Password", type="password", key="login_pwd")
        login_submit = st.form_submit_button("🔓 Login")

    if login_submit:
        if login_user == "admin" and login_pwd == "admin123":
            st.session_state["admin_authenticated"] = True
            st.success("✅ Logged in as Admin")
        else:
            st.session_state["admin_authenticated"] = False
            st.error("❌ Invalid credentials")

    if st.session_state.get("admin_authenticated", False):
        st.markdown("### 🛠️ Admin Controls")
        if st.button("🚪 Logout"):
            st.session_state["admin_authenticated"] = False
            st.rerun()

        df = load_complaints()
        if df.empty:
            st.warning("No complaints available yet.")
        else:
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            df["PriorityOrder"] = df["Priority"].map(priority_order).fillna(3)
            df = df.sort_values(by="PriorityOrder").reset_index(drop=True)

            st.markdown("### 📋 All Complaints")
            st.dataframe(df[[
                "PNR", "Phone", "Date", "Type", "SubType",
                "Translated_Complaint", "Priority", "Department",
                "Reply", "Status"
            ]].reset_index(drop=True), use_container_width=True)

            # Delete complaints
            st.markdown("### 🗑️ Manage Complaints")
            with st.form("delete_form"):
                for idx, row in df.iterrows():
                    key = f"del_{row['PNR']}_{idx}"
                    short_text = (row["Translated_Complaint"][:120] + "...") if len(row["Translated_Complaint"]) > 120 else row["Translated_Complaint"]
                    st.checkbox(f"PNR: {row['PNR']} | {short_text} | Priority: {row['Priority']} | Dept: {row['Department']}", key=key)
                delete_selected = st.form_submit_button("🗑️ Delete Selected")

            if delete_selected:
                selected_pnrs = [pnr.split("_")[1] for key, pnr in st.session_state.items() if key.startswith("del_") and pnr]
                df_new = df[~df["PNR"].isin(selected_pnrs)].copy()
                df_new.drop(columns=["PriorityOrder"], inplace=True, errors="ignore")
                df_new.to_csv(CSV_FILE, index=False)
                st.success("✅ Deleted selected complaints.")
                st.rerun()

            if st.button("🗑️ Delete All Complaints"):
                if os.path.exists(CSV_FILE): os.remove(CSV_FILE)
                st.success("✅ All complaints deleted.")
                st.rerun()

            # Update statuses
            st.markdown("### 📌 Update Status")
            with st.form("status_form"):
                for idx, row in df.iterrows():
                    s_key = f"status_{row['PNR']}_{idx}"
                    st.checkbox(f"Mark Resolved: PNR {row['PNR']} | {row['Translated_Complaint'][:80]}...", key=s_key, value=(row['Status'] == "Resolved"))
                update_status = st.form_submit_button("✅ Update Statuses")

            if update_status:
                for idx, row in df.iterrows():
                    s_key = f"status_{row['PNR']}_{idx}"
                    df.at[idx, "Status"] = "Resolved" if st.session_state.get(s_key, False) else "In Progress"
                df.drop(columns=["PriorityOrder"], inplace=True, errors="ignore")
                df.to_csv(CSV_FILE, index=False)
                st.success("✅ Statuses updated.")
                st.rerun()

            st.markdown("### 📈 Complaint Statistics")

            # Bar chart: Complaints by Priority
            priority_counts = df["Priority"].value_counts()
            priority_colors = {"High": "#e74c3c", "Medium": "#f1c40f", "Low": "#2ecc71"}
            fig_priority = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                color=priority_counts.index,
                color_discrete_map=priority_colors,
                labels={"x": "Priority", "y": "Number of Complaints"},
                title="Complaints by Priority"
            )

            # Bar chart: Complaints by Department
            department_counts = df["Department"].value_counts()
            department_colors = px.colors.qualitative.Set3  # using a qualitative color set
            fig_department = px.bar(
                x=department_counts.index,
                y=department_counts.values,
                color=department_counts.index,
                color_discrete_sequence=department_colors,
                labels={"x": "Department", "y": "Number of Complaints"},
                title="Complaints by Department"
            )

            # Pie chart: Complaints distribution by Department
            fig_pie = px.pie(
                names=department_counts.index,
                values=department_counts.values,
                color=department_counts.index,
                color_discrete_sequence=department_colors,
                title="Distribution of Complaints by Department"
            )

            # Layout: 2 columns for bar charts, pie chart below
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_priority, use_container_width=True)
            with col2:
                st.plotly_chart(fig_department, use_container_width=True)

            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("### 💬 User Suggestions")
            SUGGESTIONS_FILE = "suggestions.csv"

            def save_suggestion(data: dict):
                df_new = pd.DataFrame([data])
                if os.path.exists(SUGGESTIONS_FILE):
                    df_old = pd.read_csv(SUGGESTIONS_FILE, dtype=str)
                    df_all = pd.concat([df_old, df_new], ignore_index=True)
                else:
                    df_all = df_new
                df_all.to_csv(SUGGESTIONS_FILE, index=False)

            def load_suggestions():
                if os.path.exists(SUGGESTIONS_FILE):
                    return pd.read_csv(SUGGESTIONS_FILE, dtype=str)
                return pd.DataFrame()

            # Display suggestions
            df_suggestions = load_suggestions()
            if df_suggestions.empty:
                st.info("No suggestions submitted yet.")
            else:
                st.dataframe(df_suggestions[["Date", "Suggestion"]].reset_index(drop=True), use_container_width=True)

            # Optional: Delete suggestions
            st.markdown("### 🗑️ Manage Suggestions")
            with st.form("delete_suggestions_form"):
                for idx, row in df_suggestions.iterrows():
                    key = f"del_sugg_{idx}"
                    short_text = (row["Suggestion"][:120] + "...") if len(row["Suggestion"]) > 120 else row["Suggestion"]
                    st.checkbox(f"{row['Date']} | {short_text}", key=key)
                delete_selected_suggestions = st.form_submit_button("🗑️ Delete Selected Suggestions")

            if delete_selected_suggestions:
                selected_indices = [idx for idx in range(len(df_suggestions)) if st.session_state.get(f"del_sugg_{idx}", False)]
                if selected_indices:
                    df_new = df_suggestions.drop(index=selected_indices)
                    df_new.to_csv(SUGGESTIONS_FILE, index=False)
                    st.success(f"✅ Deleted {len(selected_indices)} suggestion(s).")
                    st.experimental_rerun()
