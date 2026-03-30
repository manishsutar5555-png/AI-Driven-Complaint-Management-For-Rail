import streamlit as st
import datetime
import os
import pandas as pd
import tempfile

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
    """Return a custom reply message based on department & urgency."""
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

# === Data Storage Helper === (unchanged)
CSV_FILE = "complaints.csv"

def save_complaint(data: dict):
    """Save complaint into CSV for admin dashboard."""
    data["PNR"] = str(data["PNR"])   # force string
    df_new = pd.DataFrame([data])
    if os.path.exists(CSV_FILE):
        df_old = pd.read_csv(CSV_FILE, dtype=str)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(CSV_FILE, index=False)

def load_complaints():
    """Load complaints from CSV."""
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE, dtype=str)
    else:
        return pd.DataFrame()

# === PDF Extractor ===
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyPDF2 if available, otherwise fallback."""
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

# === Helper: normalize classifier result to caption text ===
def extract_caption_from_result(results):
    """
    classifier functions may return:
      - a string (caption), or
      - a list like [(caption, score), ...]
    This helper extracts a clean caption string robustly.
    """
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

# === Streamlit Config & Styling ===
st.set_page_config(page_title="RailMadad AI Portal", layout="wide")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://d12aarmt01l54a.cloudfront.net/cms/images/Media-20230301152109/1200-630.png");
    background-size: cover;
}
[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.9);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# === Sidebar Navigation ===
with st.sidebar:
    st.image("https://d12aarmt01l54a.cloudfront.net/cms/images/Media-20240727173411/808-440.png", width=100)
    st.title("🚆 RailMadad")
    st.markdown("**For Inquiry, Assistance & Grievance Redressal**")
    menu = st.radio("Navigation", [
        "Home", "Submit Complaint", "Track Your Concern", "Suggestions", "Admin Dashboard"
    ])

# --- Pages ---

# Home
if menu == "Home":
    st.markdown("## Welcome to RailMadad AI Complaint Management")
    st.info("👉 Submit your complaints, get AI-driven prioritization, department classification & instant replies.")
    st.image("https://images.unsplash.com/photo-1565895405140-d63972a34f3d", use_container_width=True)

# Submit Complaint
elif menu == "Submit Complaint":
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
        st.session_state.reply = ""

    if not st.session_state.submitted:
        st.markdown("## 📝 Submit Your Complaint")
        col1, col2 = st.columns(2)
        with col1:
            phone = st.text_input("📱 Mobile No.")
            prn = st.text_input("🎟️ PNR Number")
            incident_date = st.date_input("📅 Incident Date", datetime.date.today())
        with col2:
            complaint_type = st.selectbox("Type", [
                "Coach Maintenance", "Cleanliness", "Catering", "Security", "Ticketing", "Other"
            ])
            complaint_subtype = st.text_input("Sub Type")
            uploaded_file = st.file_uploader("Upload File (PDF/JPG/PNG/MP4 up to 5MB)",
                                             type=["pdf","jpg","jpeg","png","mp4"])

        complaint_text = st.text_area("📝 Grievance Description (optional)", height=150)

        if st.button("Submit Complaint"):
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
                        # use NamedTemporaryFile to avoid permission issues
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
                            try:
                                os.remove(temp_path)
                            except Exception:
                                pass
                    else:
                        text_for_ai = "General Issue"

                    # translate -> classification
                    translated_text = translate_to_english(text_for_ai)
                    priority = predict_priority(translated_text)
                    department = classify_input(translated_text)
                    custom_reply = generate_reply(department, priority)

                    # save
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
        st.info(f"**Custom Reply:** {st.session_state.reply}")

# Track
elif menu == "Track Your Concern":
    st.markdown("## 🔍 Track Complaint")
    track_prn = st.text_input("Enter your PNR to track status")
    if st.button("Track"):
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
    suggestion = st.text_area("Share your suggestions with us:")
    if st.button("Submit Suggestion"):
        st.success("✅ Thank you for your feedback!")

# Admin Dashboard
elif menu == "Admin Dashboard":
    st.markdown("## 🔐 Admin Login")

    # ensure auth flag exists
    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False

    # Login form (submit to authenticate)
    with st.form("login_form"):
        login_user = st.text_input("Username", key="login_user")
        login_pwd = st.text_input("Password", type="password", key="login_pwd")
        login_submit = st.form_submit_button("Login")

    if login_submit:
        if login_user == "admin" and login_pwd == "admin123":
            st.session_state["admin_authenticated"] = True
            st.success("✅ Logged in as Admin")
        else:
            st.session_state["admin_authenticated"] = False
            st.error("❌ Invalid credentials")

    # show admin UI if authenticated
    if st.session_state.get("admin_authenticated", False):
        st.markdown("**Admin Controls**")
        if st.button("Logout"):
            st.session_state["admin_authenticated"] = False
            st.rerun()

        df = load_complaints()
        if df.empty:
            st.warning("No complaints available yet.")
        else:
            # Sort & reset index for stable ordering/keys
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            df["PriorityOrder"] = df["Priority"].map(priority_order).fillna(3)
            df = df.sort_values(by="PriorityOrder").reset_index(drop=True)

            # table view
            st.markdown("### 📋 All Complaints")
            st.dataframe(df[[
                "PNR", "Phone", "Date", "Type", "SubType",
                "Translated_Complaint", "Priority", "Department",
                "Reply", "Status"
            ]].reset_index(drop=True))

            # Delete selected using a form + stable keys
            st.markdown("### 🗑️ Select complaints to delete")
            with st.form("delete_form"):
                checkbox_items = []
                for idx, row in df.iterrows():
                    # stable key: PNR + index
                    key = f"del_{row['PNR']}_{idx}"
                    short_text = (row["Translated_Complaint"][:120] + "...") if len(row["Translated_Complaint"]) > 120 else row["Translated_Complaint"]
                    st.checkbox(f"PNR: {row['PNR']} | {short_text} | Priority: {row['Priority']} | Dept: {row['Department']}", key=key)
                    checkbox_items.append((key, row["PNR"]))

                delete_selected = st.form_submit_button("🗑️ Delete Selected Complaints")

            if delete_selected:
                # read selected keys from session_state
                selected_pnrs = [pnr for key, pnr in checkbox_items if st.session_state.get(key, False)]
                if selected_pnrs:
                    df_new = df[~df["PNR"].isin(selected_pnrs)].copy()
                    df_new.drop(columns=["PriorityOrder"], inplace=True, errors="ignore")
                    df_new.to_csv(CSV_FILE, index=False)
                    st.success(f"✅ Deleted {len(selected_pnrs)} complaint(s).")
                    st.rerun()
                else:
                    st.warning("⚠️ Please select at least one complaint to delete.")

            # Delete All Complaints
            if st.button("🗑️ Delete All Complaints"):
                if os.path.exists(CSV_FILE):
                    os.remove(CSV_FILE)
                st.success("✅ All complaints deleted.")
                st.rerun()

            # Status toggles (separate form to avoid key conflicts)
            st.markdown("### 🗂️ Update Status (Resolved / In Progress)")
            with st.form("status_form"):
                status_keys = []
                for idx, row in df.iterrows():
                    s_key = f"status_{row['PNR']}_{idx}"
                    checked = st.checkbox(f"Mark Resolved: PNR {row['PNR']} | {row['Translated_Complaint'][:80]}...", key=s_key, value=(row['Status'] == "Resolved"))
                    status_keys.append((s_key, idx))
                update_status = st.form_submit_button("Update Statuses")

            if update_status:
                # apply status updates
                for s_key, idx in status_keys:
                    if st.session_state.get(s_key, False):
                        df.at[idx, "Status"] = "Resolved"
                    else:
                        df.at[idx, "Status"] = "In Progress"
                df.drop(columns=["PriorityOrder"], inplace=True, errors="ignore")
                df.to_csv(CSV_FILE, index=False)
                st.success("✅ Statuses updated.")
                st.rerun()

            # Save final (in case something changed)
            df.drop(columns=["PriorityOrder"], inplace=True, errors="ignore")
            df.to_csv(CSV_FILE, index=False)

            # Stats
            st.markdown("### 📈 Complaint Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(df["Priority"].value_counts())
            with col2:
                st.bar_chart(df["Department"].value_counts())
