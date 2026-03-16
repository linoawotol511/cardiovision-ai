import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import base64
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import uuid
from datetime import datetime, date
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from PIL import Image, ImageDraw

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="CardioVision AI", layout="wide")

# -----------------------------
# Session State
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "report_data" not in st.session_state:
    st.session_state.report_data = None

# -----------------------------
# Logo Loader
# -----------------------------
logo_file = "logo.jpeg"   # change to "logo.jpg" if needed

def get_base64_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64_image(logo_file)

# -----------------------------
# Load and Train Model
# -----------------------------
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

data = pd.read_csv("processed.cleveland.data", names=columns)
data = data.replace("?", pd.NA)
data["ca"] = pd.to_numeric(data["ca"])
data["thal"] = pd.to_numeric(data["thal"])
data = data.dropna()
data["target"] = (data["target"] > 0).astype(int)

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Helper Functions
# -----------------------------
def risk_category(prob):
    if prob < 30:
        return "Low Heart Disease Risk"
    elif prob < 60:
        return "Moderate Heart Disease Risk"
    else:
        return "High Heart Disease Risk"


def preventive_recommendations(prob):
    if prob < 30:
        return [
            "Maintain a balanced diet rich in fruits and vegetables.",
            "Engage in regular physical activity such as jogging or cycling for at least 150 minutes per week.",
            "Avoid smoking and excessive alcohol consumption.",
            "Maintain a healthy body weight and body mass index.",
            "Monitor blood pressure and cholesterol during routine health checkups."
        ]
    elif prob < 60:
        return [
            "Reduce dietary cholesterol and saturated fat intake.",
            "Increase physical activity to at least 150 minutes per week.",
            "Stop smoking to reduce cardiovascular strain.",
            "Maintain healthy body weight and balanced nutrition.",
            "Reduce salt consumption to manage blood pressure."
        ]
    else:
        return [
            "Adopt a strict heart-healthy diet with low fat and low sodium.",
            "Avoid smoking and alcohol consumption.",
            "Engage only in doctor-approved physical activities.",
            "Maintain proper sleep and stress management.",
            "Monitor sugar intake if diabetic or pre-diabetic."
        ]


def medical_advisory(prob):
    if prob < 30:
        return [
            "Continue routine annual health checkups.",
            "Monitor blood pressure and cholesterol periodically."
        ]
    elif prob < 60:
        return [
            "Schedule a cardiovascular health screening.",
            "Monitor blood pressure regularly.",
            "Check cholesterol levels periodically.",
            "Discuss cardiovascular risk factors with a healthcare professional."
        ]
    else:
        return [
            "Seek immediate evaluation by a cardiologist.",
            "Perform diagnostic tests such as ECG, echocardiogram, or stress test.",
            "Monitor blood pressure and cholesterol closely.",
            "Begin treatment or medication as prescribed by a healthcare professional.",
            "Schedule regular medical follow-ups."
        ]


def risk_explanation(age, trestbps, chol, thalach, exang, oldpeak):
    factors = []

    if age >= 50:
        factors.append("Age above 50 increases cardiovascular risk.")
    if trestbps >= 140:
        factors.append("Elevated resting blood pressure is a significant risk factor.")
    if chol >= 240:
        factors.append("High cholesterol level may increase the likelihood of heart disease.")
    if thalach < 150:
        factors.append("Lower maximum heart rate may indicate reduced cardiovascular fitness.")
    if exang == 1:
        factors.append("Exercise-induced angina suggests possible cardiac stress.")
    if oldpeak >= 2:
        factors.append("High oldpeak value may indicate abnormal heart response during exercise.")

    if not factors:
        factors.append("No major high-risk indicators were detected from the entered values.")

    return factors


def build_patient_graph(report):
    age = report["age"]
    chol = report["chol"]
    trestbps = report["trestbps"]
    thalach = report["thalach"]

    labels = ["Cholesterol", "Systolic BP", "Max Heart Rate"]
    patient_values = [chol, trestbps, thalach]
    reference_values = [200, 120, 220 - age]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, patient_values, width, label="Patient Value")
    ax.bar(x + width / 2, reference_values, width, label="Reference Value")

    ax.set_title("Patient vs Accepted Cardiovascular Reference Values")
    ax.set_ylabel("Measurement Value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    return fig


def make_circular_logo(input_path=logo_file):
    img = Image.open(input_path).convert("RGBA")
    size = min(img.size)
    left = (img.width - size) // 2
    top = (img.height - size) // 2
    img = img.crop((left, top, left + size, top + size))

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    circular = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    circular.paste(img, (0, 0), mask=mask)

    temp_logo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    circular.save(temp_logo.name, format="PNG")
    return temp_logo.name


def blue_section_heading(text, width=520, height=22):
    table = Table([[text]], colWidths=[width], rowHeights=[height])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1E88E5")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def add_page_header(canvas, doc, patient_id):
    canvas.setFont("Helvetica-Bold", 10)
    canvas.setFillColor(colors.HexColor("#1E88E5"))
    canvas.drawString(40, A4[1] - 30, f"Patient ID: {patient_id}")


def create_pdf_report(report, fig, logo_path=logo_file):
    graph_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(graph_file.name, bbox_inches="tight", dpi=200)

    circular_logo_path = make_circular_logo(logo_path)

    pdf_file = "CardioVision_AI_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=50,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=28,
        textColor=colors.HexColor("#1E88E5"),
        leading=32,
        spaceAfter=12,
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        textColor=colors.black,
        leading=18,
        spaceAfter=20,
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=colors.black,
    )

    small_gray = ParagraphStyle(
        "SmallGray",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.grey,
        leading=12,
    )

    story = []

    # -----------------------------
    # COVER PAGE
    # -----------------------------
    story.append(Spacer(1, 0.3 * inch))
    story.append(RLImage(circular_logo_path, width=1.5 * inch, height=1.5 * inch))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("CARDIOVISION AI", title_style))
    story.append(Paragraph("An Intelligent Cardiovascular Risk Prediction and Analysis System", subtitle_style))

    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph(f"<b>Patient ID:</b> {report['patient_id']}", normal_style))
    story.append(Paragraph(f"<b>Patient Name:</b> {report['patient_name']}", normal_style))
    story.append(Paragraph(f"<b>Date of Analysis:</b> {report['current_date']}", normal_style))
    story.append(Paragraph(f"<b>Time of Data Collection:</b> {report['current_time']}", normal_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("<b>Prepared by:</b> James Ntiamoah & Kelvin Awotol", normal_style))
    story.append(Spacer(1, 0.45 * inch))

    cover_box = Table(
        [[Paragraph("CARDIO AI HEALTH REPORT", ParagraphStyle(
            "CoverBoxText",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=colors.HexColor("#1E88E5"),
            leading=24
        ))]],
        colWidths=[450],
        rowHeights=[100]
    )
    cover_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#D9E8FF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 25),
    ]))
    story.append(cover_box)
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "This report summarizes patient cardiovascular risk assessment, predictive analytics, "
        "clinical explanations, recommendations, medical advisory, and graphical comparison.",
        normal_style
    ))
    story.append(PageBreak())

    # -----------------------------
    # PATIENT INFORMATION PAGE
    # -----------------------------
    story.append(blue_section_heading("PATIENT INFORMATION"))
    story.append(Spacer(1, 0.2 * inch))

    patient_table_data = [
        ["Health Indicator", "Value"],
        ["Patient Name", str(report["patient_name"])],
        ["Age", str(report["age"])],
        ["Sex", str(report["sex"])],
        ["Chest Pain Type", str(report["cp"])],
        ["Resting Blood Pressure", f"{report['trestbps']} mmHg"],
        ["Cholesterol", f"{report['chol']} mg/dL"],
        ["Fasting Blood Sugar > 120 mg/dL", str(report["fbs"])],
        ["Rest ECG Result", str(report["restecg"])],
        ["Maximum Heart Rate Achieved", str(report["thalach"])],
        ["Exercise Induced Angina", str(report["exang"])],
        ["Oldpeak", str(report["oldpeak"])],
        ["Slope", str(report["slope"])],
        ["Major Vessels", str(report["ca"])],
        ["Thal", str(report["thal"])],
    ]

    patient_table = Table(patient_table_data, colWidths=[230, 230])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E88E5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFF")]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.25 * inch))

    story.append(blue_section_heading("PREDICTION SUMMARY"))
    story.append(Spacer(1, 0.15 * inch))

    prediction_text = "Heart Disease Detected" if report["prediction"] == 1 else "No Heart Disease Detected"

    summary_table_data = [
        ["Result", "Value"],
        ["Predicted Cardiovascular Risk", f"{round(report['probability'], 2)}%"],
        ["Risk Classification", report["category"]],
        ["Prediction", prediction_text],
        ["Date of Analysis", str(report["current_date"])],
        ["Time of Data Collection", str(report["current_time"])],
    ]

    summary_table = Table(summary_table_data, colWidths=[230, 230])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E88E5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFF")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(PageBreak())

    # -----------------------------
    # EXPLANATION / RECOMMENDATIONS / ADVISORY
    # -----------------------------
    story.append(blue_section_heading("RISK EXPLANATION"))
    story.append(Spacer(1, 0.12 * inch))
    for item in report["explanations"]:
        story.append(Paragraph(f"• {item}", normal_style))
        story.append(Spacer(1, 0.05 * inch))

    story.append(Spacer(1, 0.2 * inch))
    story.append(blue_section_heading("PREVENTIVE RECOMMENDATIONS"))
    story.append(Spacer(1, 0.12 * inch))
    for item in report["recommendations"]:
        story.append(Paragraph(f"• {item}", normal_style))
        story.append(Spacer(1, 0.05 * inch))

    story.append(Spacer(1, 0.2 * inch))
    story.append(blue_section_heading("MEDICAL ADVISORY"))
    story.append(Spacer(1, 0.12 * inch))
    for item in report["advisory"]:
        story.append(Paragraph(f"• {item}", normal_style))
        story.append(Spacer(1, 0.05 * inch))

    story.append(PageBreak())

    # -----------------------------
    # GRAPH PAGE
    # -----------------------------
    story.append(blue_section_heading("PATIENT GRAPH COMPARISON"))
    story.append(Spacer(1, 0.2 * inch))
    story.append(RLImage(graph_file.name, width=6.3 * inch, height=3.8 * inch))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Reference Guide:", ParagraphStyle(
        "RefHead", parent=normal_style, fontName="Helvetica-Bold", fontSize=11
    )))
    story.append(Paragraph("• Total cholesterol: desirable if below 200 mg/dL", normal_style))
    story.append(Paragraph("• Normal systolic blood pressure: below 120 mmHg", normal_style))
    story.append(Paragraph("• Estimated maximum heart rate: approximately 220 minus age", normal_style))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("CardioVision AI | Done by James & Kelvin", small_gray))

    def first_page(canvas, doc):
        pass

    def later_pages(canvas, doc):
        add_page_header(canvas, doc, report["patient_id"])

    doc.build(story, onFirstPage=first_page, onLaterPages=later_pages)
    return pdf_file


def show_footer():
    st.markdown(
        f"""
        <style>
        .bottom-right-logo {{
            position: fixed;
            bottom: 10px;
            right: 10px;
            text-align: center;
            z-index: 1000;
        }}
        .bottom-right-logo img {{
            width: 90px;
            border-radius: 10px;
        }}
        .bottom-right-logo p {{
            margin: 0;
            font-size: 12px;
            color: gray;
        }}
        </style>

        <div class="bottom-right-logo">
            <img src="data:image/jpeg;base64,{logo_base64}">
            <p>Done by James & Kelvin</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# HOME PAGE
# -----------------------------
if st.session_state.page == "home":

    components.html(
        f"""
        <html>
        <head>
        <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: Arial, sans-serif;
        }}

        .home-container {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 75vh;
            text-align: center;
            padding: 20px;
        }}

        .logo-circle {{
            width: 180px;
            height: 180px;
            border-radius: 50%;
            overflow: hidden;
            border: 4px solid #0b5ed7;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }}

        .logo-circle img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}

        .typing-title {{
            font-size: 4vw;
            font-weight: 900;
            color: #1E88E5;
            font-family: 'Arial Black', sans-serif;
            white-space: nowrap;
            overflow: hidden;
            border-right: 3px solid #0b3d91;
            width: 0;
            animation: typing 4s steps(16, end) forwards, blink 0.8s infinite;
        }}

        @keyframes typing {{
            from {{ width: 0; }}
            to {{ width: 16ch; }}
        }}

        @keyframes blink {{
            50% {{ border-color: transparent; }}
        }}

        .welcome-text {{
            margin-top: 30px;
            font-size: 1.4vw;
            color: #757575;
            font-weight: bold;
        }}

        .question-text {{
            margin-top: 130px;
            font-size: 1.8vw;
            font-weight: bold;
            color: #616161;
        }}

        @media (max-width: 900px) {{
            .typing-title {{ font-size: 7vw; }}
            .welcome-text {{ font-size: 3.5vw; }}
            .question-text {{ font-size: 3.5vw; }}
            .logo-circle {{ width: 140px; height: 140px; }}
        }}

        @media (max-width: 500px) {{
            .typing-title {{ font-size: 9vw; }}
            .welcome-text {{ font-size: 4.5vw; }}
            .question-text {{ font-size: 4.5vw; }}
            .logo-circle {{ width: 110px; height: 110px; }}
        }}
        </style>
        </head>

        <body>
            <div class="home-container">
                <div class="logo-circle">
                    <img src="data:image/jpeg;base64,{logo_base64}">
                </div>

                <div class="typing-title">CardioVision AI</div>

                <div class="welcome-text">
                    Welcome to the CardioVision AI health risk prediction system
                </div>

                <div class="question-text">
                    Would you like to proceed to entering patient’s health information?
                </div>
            </div>
        </body>
        </html>
        """,
        height=520,
        scrolling=False
    )

    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 3])

    with col3:
        if st.button("Yes"):
            st.session_state.page = "form"
            st.rerun()

    with col4:
        if st.button("No"):
            st.warning("You chose not to proceed.")

# -----------------------------
# FORM PAGE
# -----------------------------
elif st.session_state.page == "form":
    st.title("CardioVision AI")
    st.subheader("An Intelligent Cardiovascular Risk Prediction and Analysis System")
    st.write("Enter patient health information below.")

    patient_name = st.text_input("Patient Name")

    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3, value=0)
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)
    chol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    restecg = st.number_input("Resting ECG Result (0–2)", min_value=0, max_value=2, value=0)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.number_input("Slope (0–2)", min_value=0, max_value=2, value=1)
    ca = st.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=3, value=0)
    thal = st.number_input("Thal (1–3)", min_value=1, max_value=3, value=2)

    predict_button = st.button("Predict Cardiovascular Risk")

    if predict_button:
        patient_data = [[
            age,
            1 if sex == "Male" else 0,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]]

        prediction = model.predict(patient_data)[0]
        probability = model.predict_proba(patient_data)[0][1] * 100

        current_date = datetime.now().strftime("%d-%m-%Y")
        current_time = datetime.now().strftime("%H:%M:%S")
        patient_id = str(uuid.uuid4())[:8].upper()

        st.session_state.report_data = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "current_date": current_date,
            "current_time": current_time,
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "prediction": prediction,
            "probability": probability,
            "category": risk_category(probability),
            "explanations": risk_explanation(age, trestbps, chol, thalach, exang, oldpeak),
            "recommendations": preventive_recommendations(probability),
            "advisory": medical_advisory(probability)
        }

    if st.session_state.report_data is not None:
        report = st.session_state.report_data

        st.subheader("CardioVision AI Health Report")
        st.write("Patient ID:", report["patient_id"])
        st.write("Patient Name:", report["patient_name"])
        st.write("Date of Analysis:", report["current_date"])
        st.write("Time of Data Collection:", report["current_time"])
        st.write("Predicted Cardiovascular Risk:", round(report["probability"], 2), "%")
        st.write("Risk Classification:", report["category"])

        if report["prediction"] == 1:
            st.write("Prediction: Heart Disease Detected")
        else:
            st.write("Prediction: No Heart Disease Detected")

        st.subheader("Risk Explanation")
        for factor in report["explanations"]:
            st.write("-", factor)

        st.subheader("Preventive Recommendations")
        for rec in report["recommendations"]:
            st.write("-", rec)

        st.subheader("Medical Advisory")
        for adv in report["advisory"]:
            st.write("-", adv)

        st.write("Do you want to view graph comparison for this patient?")

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Yes, View Graph"):
                st.session_state.page = "graph"
                st.rerun()

        with col_b:
            if st.button("No"):
                st.info("You can remain on this report page.")

    show_footer()

# -----------------------------
# GRAPH PAGE
# -----------------------------
elif st.session_state.page == "graph":
    st.title("Patient Graph Comparison")
    st.write("This chart compares the patient's values with accepted cardiovascular reference values.")

    if st.session_state.report_data is not None:
        report = st.session_state.report_data

        fig = build_patient_graph(report)
        st.pyplot(fig)

        st.subheader("Reference Guide")
        st.write("- Total cholesterol: desirable if below 200 mg/dL")
        st.write("- Normal systolic blood pressure: below 120 mmHg")
        st.write("- Estimated maximum heart rate: approximately 220 minus age")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Back to Report"):
                st.session_state.page = "form"
                st.rerun()

        with col2:
            pdf_file = create_pdf_report(report, fig, logo_file)
            with open(pdf_file, "rb") as file:
                st.download_button(
                    label="Download / Print Full Report",
                    data=file,
                    file_name="CardioVision_AI_Report.pdf",
                    mime="application/pdf"
                )
    else:
        st.warning("No patient report found. Please return and generate a report first.")
        if st.button("Back to Form"):
            st.session_state.page = "form"
            st.rerun()

    show_footer()