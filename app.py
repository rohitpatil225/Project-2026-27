from flask import Flask, render_template, request, send_file, redirect, url_for, flash, session, abort
from database import (
    initialize_database,
    save_report,
    get_all_reports,
    get_report,
    create_user,
    get_user_by_email,
    get_user_by_id,
    get_report_by_pdf_filename,
    update_doctor_remarks,
    update_user_profile,
    email_exists_for_other_user,
    update_user_password
)
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from urllib.parse import urlsplit
import os
import secrets
import uuid
import cv2
import json
# ============================================================
# REPORTLAB IMPORTS
# ============================================================

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
print("FLASK STATIC FOLDER:", app.static_folder)
# Set RETINAAI_SECRET_KEY in your environment for deployments. A local key file
# keeps sessions valid across restarts during local development.
_secret_file = Path(__file__).resolve().parent / ".retinaai_secret_key"
if os.environ.get("RETINAAI_SECRET_KEY"):
    app.config["SECRET_KEY"] = os.environ["RETINAAI_SECRET_KEY"]
else:
    if not _secret_file.exists():
        _secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
    app.config["SECRET_KEY"] = _secret_file.read_text(encoding="utf-8").strip()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    # Enable this when deployed behind HTTPS; local HTTP development needs False.
    SESSION_COOKIE_SECURE=bool(os.environ.get("RETINAAI_HTTPS", "").lower() == "true"),
    MAX_CONTENT_LENGTH=15 * 1024 * 1024,
)
initialize_database()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to access RetinaAI.", "info")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


def csrf_token():
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


app.jinja_env.globals["csrf_token"] = csrf_token

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = (
    BASE_DIR
    / "static"
    / "results"
)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# YOLO MODEL
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "dr_yolo_v2"
    / "weights"
    / "best.pt"
)

model = YOLO(
    str(MODEL_PATH)
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Microaneurysm",
    1: "Haemorrhage",
    2: "Hard Exudate",
    3: "Soft Exudate"
}


# ============================================================
# COLORS
#
# OpenCV uses BGR
#
# Microaneurysm = Blue
# Haemorrhage   = Red
# Hard Exudate  = Yellow
# Soft Exudate  = Purple
# ============================================================

CLASS_COLORS = {
    0: (255, 80, 80),
    1: (80, 80, 255),
    2: (0, 215, 255),
    3: (200, 80, 255)
}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        if request.form.get("csrf_token") != session.get("csrf_token"):
            abort(400, description="Invalid form token. Please refresh and try again.")

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or len(name) > 100:
            flash("Please enter your name (up to 100 characters).", "error")
        elif not email or len(email) > 254 or "@" not in email:
            flash("Please enter a valid email address.", "error")
        elif len(password) < 10:
            flash("Use a password with at least 10 characters.", "error")
        elif password != confirm_password:
            flash("The passwords do not match.", "error")
        elif get_user_by_email(email):
            flash("An account with that email already exists. Please log in.", "error")
        else:
            user_id = create_user(name, email, generate_password_hash(password))
            session.clear()
            session["user_id"] = user_id
            session["user_name"] = name
            session["csrf_token"] = secrets.token_urlsafe(32)
            flash("Your RetinaAI account has been created.", "success")
            return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        if request.form.get("csrf_token") != session.get("csrf_token"):
            abort(400, description="Invalid form token. Please refresh and try again.")

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_user_by_email(email)

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
        else:
            session.clear()  # Avoid retaining any pre-login session state.
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["csrf_token"] = secrets.token_urlsafe(32)
            flash("Welcome back to RetinaAI.", "success")
            return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    if request.form.get("csrf_token") != session.get("csrf_token"):
        abort(400, description="Invalid form token. Please refresh and try again.")
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))



@app.route("/")
def home():
    return render_template("landing.html")


@app.route("/screening")
@login_required
def screening():
    return render_template("index.html")



# ============================================================
# DASHBOARD & ANALYTICS
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():
    # Fetch reports belonging to the logged-in user.
    reports = get_all_reports(session["user_id"])

    total_reports = len(reports)

    total_detections = sum(
        int(report["total_detections"] or 0)
        for report in reports
    )

    # Count unique patients.
    unique_patients = set()

    for report in reports:
        patient_id = (report["patient_id"] or "").strip()
        patient_name = (report["patient_name"] or "").strip()

        if patient_id:
            unique_patients.add("ID:" + patient_id.lower())
        elif patient_name:
            unique_patients.add("NAME:" + patient_name.lower())

    total_patients = len(unique_patients)

    detected_reports = sum(
        1 for report in reports
        if (report["result_status"] or "").lower() == "detected"
    )

    no_detection_reports = sum(
        1 for report in reports
        if (report["result_status"] or "").lower() == "not-detected"
    )

    status_counts = [
        {
            "label": "Lesions detected",
            "count": detected_reports,
            "color": "#0f9d8a",
            "percentage": round(detected_reports / total_reports * 100)
            if total_reports else 0
        },
        {
            "label": "No target lesions detected",
            "count": no_detection_reports,
            "color": "#1597e5",
            "percentage": round(no_detection_reports / total_reports * 100)
            if total_reports else 0
        }
    ]

    recent_reports = reports[:5]

    # Calculate totals for each lesion type.
    lesion_totals = {
        "Microaneurysm": 0,
        "Haemorrhage": 0,
        "Hard_Exudate": 0,
        "Soft_Exudate": 0
    }

    for report in reports:
        try:
            class_counts = json.loads(report["lesion_counts"] or "{}")
        except (json.JSONDecodeError, TypeError):
            class_counts = {}

        for class_name in lesion_totals:
            lesion_totals[class_name] += int(
                class_counts.get(class_name, 0) or 0
            )

    lesion_items = [
        {"label": class_name.replace("_", " "), "count": count}
        for class_name, count in lesion_totals.items()
    ]

    # Build examination trend data.
    trend_counts = {}

    for report in reports:
        date_value = (
            report["examination_date"]
            or report["analysis_datetime"]
        )

        if date_value:
            date_label = str(date_value)[:10]
            trend_counts[date_label] = (
                trend_counts.get(date_label, 0) + 1
            )

    trend_items = [
        {"date": date_label, "count": count}
        for date_label, count in sorted(trend_counts.items())
    ][-12:]

    return render_template(
        "dashboard.html",
        total_reports=total_reports,
        total_detections=total_detections,
        total_patients=total_patients,
        detected_reports=detected_reports,
        no_detection_reports=no_detection_reports,
        status_counts=status_counts,
        recent_reports=recent_reports,
        trend_items=trend_items,
        lesion_items=lesion_items
    )
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():

    user = get_user_by_id(session["user_id"])

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()

        if not name or not email:
            flash("Name and email are required.", "error")

        elif email_exists_for_other_user(email, session["user_id"]):
            flash("This email is already registered.", "error")

        else:
            update_user_profile(
                session["user_id"],
                name,
                email
            )

            flash("Profile updated successfully!", "success")
            return redirect(url_for("profile"))

    user = get_user_by_id(session["user_id"])

    return render_template("profile.html", user=user)
@app.route("/change-password", methods=["POST"])
@login_required
def change_password():

    current_password = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")

    user = get_user_by_email(
        get_user_by_id(session["user_id"])["email"]
    )

    if not check_password_hash(user["password_hash"], current_password):
        flash("Current password is incorrect.", "error")

    elif len(new_password) < 8:
        flash("New password must be at least 8 characters.", "error")

    elif new_password != confirm_password:
        flash("New passwords do not match.", "error")

    elif current_password == new_password:
        flash("New password must be different from the current password.", "error")

    else:
        password_hash = generate_password_hash(new_password)

        update_user_password(
            session["user_id"],
            password_hash
        )

        flash("Password changed successfully!", "success")

    return redirect(url_for("profile"))

# ============================================================
# REPORT HISTORY
# ============================================================

@app.route("/history")
@login_required
def history():

    reports = get_all_reports(session["user_id"])

    return render_template(
        "history.html",
        reports=reports
    )
# ============================================================
# VIEW SAVED REPORT
# ============================================================

@app.route("/report/<report_id>")
@login_required
def view_report(report_id):

    report = get_report(report_id, session["user_id"])

    if report is None:
        return "Report not found.", 404

    lesion_counts = json.loads(
        report["lesion_counts"] or "{}"
    )

    all_reports = get_all_reports(session["user_id"])

    previous_report = None
    current_found = False

    for saved_report in all_reports:
        if saved_report["report_id"] == report_id:
            current_found = True
            continue

        if current_found and saved_report["patient_id"] == report["patient_id"]:
            previous_report = saved_report
            break

    previous_lesion_counts = {}

    if previous_report:
        previous_lesion_counts = json.loads(
            previous_report["lesion_counts"] or "{}"
        )

    return render_template(
        "report.html",
        report=report,
        lesion_counts=lesion_counts,
        previous_report=previous_report,
        previous_lesion_counts=previous_lesion_counts
    )
@app.route("/save-doctor-remarks/<report_id>", methods=["POST"])
@login_required
def save_doctor_remarks(report_id):
    doctor_remarks = request.form.get("doctor_remarks", "").strip()

    update_doctor_remarks(
        report_id,
        session["user_id"],
        doctor_remarks
    )

    return redirect(url_for("view_report", report_id=report_id))
# ============================================================
# CREATE CLEAN DETECTION IMAGE
#
# No class names
# No confidence labels
# Only square boxes
# ============================================================

def create_clean_detection_image(
    image_path,
    result,
    output_path
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            "Unable to read uploaded image."
        )

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # Check detections
    # --------------------------------------------------------

    if result.boxes is None:
        cv2.imwrite(
            str(output_path),
            image
        )
        return

    # --------------------------------------------------------
    # Draw each detection
    # --------------------------------------------------------

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        # Original YOLO bounding box
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # ----------------------------------------------------
        # Find center
        # ----------------------------------------------------

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        # ----------------------------------------------------
        # Make square
        # ----------------------------------------------------

        width = x2 - x1
        height = y2 - y1

        square_size = max(
            width,
            height,
            12
        )

        half_size = square_size // 2

        sx1 = center_x - half_size
        sy1 = center_y - half_size

        sx2 = center_x + half_size
        sy2 = center_y + half_size

        # ----------------------------------------------------
        # Keep square inside image
        # ----------------------------------------------------

        if sx1 < 0:

            sx2 += abs(sx1)
            sx1 = 0

        if sy1 < 0:

            sy2 += abs(sy1)
            sy1 = 0

        if sx2 > image_width - 1:

            shift = (
                sx2 -
                (image_width - 1)
            )

            sx1 -= shift
            sx2 = image_width - 1

        if sy2 > image_height - 1:

            shift = (
                sy2 -
                (image_height - 1)
            )

            sy1 -= shift
            sy2 = image_height - 1

        sx1 = max(
            0,
            int(sx1)
        )

        sy1 = max(
            0,
            int(sy1)
        )

        sx2 = min(
            image_width - 1,
            int(sx2)
        )

        sy2 = min(
            image_height - 1,
            int(sy2)
        )

        # ----------------------------------------------------
        # Color
        # ----------------------------------------------------

        color = CLASS_COLORS.get(
            class_id,
            (255, 255, 255)
        )

        # ----------------------------------------------------
        # Draw square
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (sx1, sy1),
            (sx2, sy2),
            color,
            4
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cv2.imwrite(
        str(output_path),
        image
    )


# ============================================================
# CREATE PDF REPORT
# ============================================================

def create_pdf_report(
    pdf_path,
    original_image_path,
    detection_image_path,
    patient_id,
    patient_name,
    patient_age,
    patient_gender,
    examiner_name,
    examination_date,
    detection_summary,
    total_detections,
    final_result,
    report_id,
    analysis_timestamp
):

    # --------------------------------------------------------
    # PDF DOCUMENT
    # --------------------------------------------------------

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    # --------------------------------------------------------
    # STYLES
    # --------------------------------------------------------

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor(
            "#087f6c"
        ),
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor(
            "#627d98"
        ),
        spaceAfter=8
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor(
            "#087f6c"
        ),
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor(
            "#334e68"
        )
    )

    conclusion_style = ParagraphStyle(
        "Conclusion",
        parent=styles["Normal"],
        fontSize=13,
        leading=18,
        textColor=colors.HexColor(
            "#087f6c"
        ),
        spaceAfter=8
    )

    story = []

    # ========================================================
    # HEADER
    # ========================================================

    story.append(
        Paragraph(
            "RetinaAI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Assisted Retinal Screening Report",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "YOLO11s • Retinal Lesion Detection",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Report ID:</b> {report_id} &nbsp;&nbsp; "
            f"<b>Analysis:</b> {analysis_timestamp}",
            subtitle_style
        )
    )

    story.append(
        Spacer(
            1,
            8
        )
    )

    # ========================================================
    # PATIENT INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "Patient Information",
            heading_style
        )
    )

    patient_data = [

        [
            "Patient ID",
            patient_id or "Not provided"
        ],

        [
            "Patient Name",
            patient_name or "Not provided"
        ],

        [
            "Age",
            (
                patient_age + " years"
                if patient_age
                else "Not provided"
            )
        ],

        [
            "Gender",
            patient_gender or "Not provided"
        ],

        [
            "Examiner / Doctor",
            examiner_name or "Not provided"
        ],

        [
            "Examination Date",
            examination_date or "Not provided"
        ],

        [
            "Report ID",
            report_id
        ],

        [
            "Analysis Time",
            analysis_timestamp
        ]

    ]

    patient_table = Table(
        patient_data,
        colWidths=[
            45 * mm,
            125 * mm
        ]
    )

    patient_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#e8faf7"
                )
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.HexColor(
                    "#334e68"
                )
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTNAME",
                (1, 0),
                (1, -1),
                "Helvetica"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor(
                    "#d9e5e8"
                )
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])
    )

    story.append(
        patient_table
    )

    story.append(
        Spacer(
            1,
            12
        )
    )

    # ========================================================
    # IMAGES
    # ========================================================

    story.append(
        Paragraph(
            "Retinal Image Analysis",
            heading_style
        )
    )

    # --------------------------------------------------------
    # Keep aspect ratio approximately
    # --------------------------------------------------------

    original_img = Image(
        str(original_image_path),
        width=75 * mm,
        height=75 * mm
    )

    detected_img = Image(
        str(detection_image_path),
        width=75 * mm,
        height=75 * mm
    )

    image_table = Table(

        [
            [
                original_img,
                detected_img
            ],

            [
                Paragraph(
                    "<b>Original Image</b>",
                    normal_style
                ),

                Paragraph(
                    "<b>AI Detection Image</b>",
                    normal_style
                )
            ]
        ],

        colWidths=[
            82 * mm,
            82 * mm
        ]
    )

    image_table.setStyle(
        TableStyle([

            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, 0),
                6
            )

        ])
    )

    story.append(
        image_table
    )

    story.append(
        Spacer(
            1,
            12
        )
    )

    # ========================================================
    # DETECTION LEGEND
    # ========================================================

    story.append(
        Paragraph(
            "Detection Legend",
            heading_style
        )
    )

    legend_data = [

        [
            "Color",
            "Lesion Type"
        ],

        [
            "Blue",
            "Microaneurysm"
        ],

        [
            "Red",
            "Haemorrhage"
        ],

        [
            "Yellow",
            "Hard Exudate"
        ],

        [
            "Purple",
            "Soft Exudate"
        ]

    ]

    legend_table = Table(
        legend_data,
        colWidths=[
            45 * mm,
            125 * mm
        ]
    )

    legend_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#087f6c"
                )
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor(
                    "#d9e5e8"
                )
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )

        ])
    )

    story.append(
        legend_table
    )

    # ========================================================
    # DETECTION SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "AI Detection Summary",
            heading_style
        )
    )

    summary_data = [

        [
            "Lesion",
            "Detections",
            "Highest Confidence"
        ]

    ]

    for class_name, data in detection_summary.items():

        confidence_text = (

            str(
                data["highest_confidence"]
            )
            + "%"

            if data["count"] > 0

            else

            "No detection"

        )

        summary_data.append(

            [
                class_name,
                str(data["count"]),
                confidence_text
            ]

        )

    summary_table = Table(
        summary_data,
        colWidths=[
            75 * mm,
            40 * mm,
            55 * mm
        ]
    )

    summary_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#087f6c"
                )
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor(
                    "#d9e5e8"
                )
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])
    )

    story.append(
        summary_table
    )

    story.append(
        Spacer(
            1,
            15
        )
    )

    # ========================================================
    # TOTAL DETECTIONS
    # ========================================================

    story.append(
        Paragraph(
            f"<b>Total detected lesion regions:</b> "
            f"{total_detections}",
            normal_style
        )
    )

    story.append(
        Spacer(
            1,
            10
        )
    )

    # ========================================================
    # FINAL SCREENING CONCLUSION
    # ========================================================

    story.append(
        Paragraph(
            "Final Screening Conclusion",
            heading_style
        )
    )

    story.append(
        Paragraph(
            f"<b>{final_result}</b>",
            conclusion_style
        )
    )

    if total_detections > 0:

        conclusion_text = (

            "The AI screening model detected "
            "retinal lesion regions in the submitted "
            "image. Further evaluation by a qualified "
            "ophthalmologist or eye-care professional "
            "is recommended."

        )

    else:

        conclusion_text = (

            "No target retinal lesions were detected "
            "above the configured model confidence "
            "threshold."

        )

    story.append(
        Paragraph(
            conclusion_text,
            normal_style
        )
    )

    story.append(
        Spacer(
            1,
            18
        )
    )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    disclaimer = (

        "<b>Important:</b> This report is generated "
        "by a machine-learning research prototype "
        "for screening and educational purposes. "
        "It is not a medical diagnosis and should "
        "not replace professional clinical examination."

    )

    disclaimer_table = Table(

        [
            [
                Paragraph(
                    disclaimer,
                    normal_style
                )
            ]
        ],

        colWidths=[
            170 * mm
        ]

    )

    disclaimer_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor(
                    "#fff8e8"
                )
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor(
                    "#f5dfad"
                )
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                10
            )

        ])
    )

    story.append(
        disclaimer_table
    )

    # ========================================================
    # FOOTER
    # ========================================================

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        Paragraph(
            "RetinaAI • YOLO11s Retinal Lesion Screening",
            subtitle_style
        )
    )

    # ========================================================
    # BUILD PDF
    # ========================================================

    doc.build(
        story
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
def predict():

    # ========================================================
    # PATIENT INFORMATION
    # ========================================================

    patient_id = request.form.get(
        "patient_id",
        ""
    ).strip()

    patient_name = request.form.get(
        "patient_name",
        ""
    ).strip()

    patient_age = request.form.get(
        "patient_age",
        ""
    ).strip()

    patient_gender = request.form.get(
        "patient_gender",
        ""
    ).strip()

    examiner_name = request.form.get(
        "examiner_name",
        ""
    ).strip()

    examination_date = request.form.get(
    "examination_date",
    ""
).strip()

    print("RECEIVED EXAMINATION DATE:", examination_date)

    if not examination_date:
        return "Please select an examination date."

    try:
        selected_date = datetime.strptime(
            examination_date, "%Y-%m-%d"
        ).date()

        if selected_date < datetime.today().date():
            return "Past examination dates are not allowed."

    except ValueError:
        return "Invalid examination date."
    
    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if "image" not in request.files:

        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":

        return "Please select an image."

    # ========================================================
    # UNIQUE ID
    # ========================================================

    unique_id = str(
        uuid.uuid4()
    )

    # ========================================================
    # PROFESSIONAL REPORT ID + ANALYSIS TIMESTAMP
    # ========================================================

    report_id = (
        "RAI-"
        + datetime.now().strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:6].upper()
    )

    analysis_timestamp = datetime.now().strftime(
        "%d %B %Y • %I:%M %p"
    )

    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    original_filename = (
        unique_id
        + ".jpg"
    )

    input_path = (
        UPLOAD_FOLDER
        / original_filename
    )

    file.save(
        input_path
    )

    # ========================================================
    # YOLO PREDICTION
    # ========================================================

    results = model.predict(

        source=str(
            input_path
        ),

        imgsz=1024,

        conf=0.50,

        device=0,

        verbose=False
    )

    result = results[0]

    # ========================================================
    # CREATE CLEAN DETECTION IMAGE
    # ========================================================

    detection_filename = (
        unique_id
        + "_detected.jpg"
    )

    detection_path = (
        UPLOAD_FOLDER
        / detection_filename
    )

    create_clean_detection_image(

        input_path,

        result,

        detection_path

    )

    # ========================================================
    # INITIALIZE SUMMARY
    # ========================================================

    detection_summary = {}

    for class_id, class_name in CLASS_NAMES.items():

        detection_summary[
            class_name
        ] = {

            "count": 0,

            "percentage": 0,

            "highest_confidence": 0,

            "class_id": class_id

        }

    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = (
                float(box.conf[0])
                * 100
            )

            class_name = CLASS_NAMES.get(
                class_id
            )

            if class_name is None:

                continue

            detection_summary[
                class_name
            ]["count"] += 1

            if (
                confidence
                >
                detection_summary[
                    class_name
                ]["highest_confidence"]
            ):

                detection_summary[
                    class_name
                ]["highest_confidence"] = (
                    confidence
                )

    # ========================================================
    # ROUND CONFIDENCE
    # ========================================================

    for class_name in detection_summary:

        detection_summary[
            class_name
        ]["highest_confidence"] = round(

            detection_summary[
                class_name
            ]["highest_confidence"],

            2

        )

    # ========================================================
    # TOTAL DETECTIONS
    # ========================================================

    total_detections = sum(

        data["count"]

        for data in
        detection_summary.values()

    )
        # CALCULATE LESION PERCENTAGES

    for class_name, data in detection_summary.items():

        if total_detections > 0:
            data["percentage"] = round(
                (data["count"] / total_detections) * 100,
                2
            )
        else:
            data["percentage"] = 0
    # ========================================================
    # FINAL SCREENING RESULT
    # ========================================================

    if total_detections > 0:

        final_result = (

            "Possible diabetic retinopathy "
            "lesions detected"

        )

        result_status = "detected"

    else:

        final_result = (

            "No target retinal lesions detected"

        )

        result_status = "not-detected"

    # ========================================================
    # EXAMINATION DATE
    # ========================================================

    if examination_date:

        display_date = (
            examination_date
        )

    else:

        display_date = (
            datetime.now().strftime(
                "%d %B %Y"
            )
        )

    # ========================================================
    # WEB IMAGE PATHS
    # ========================================================

    original_image = (
        "results/"
        + original_filename
    )

    result_image = (
        "results/"
        + detection_filename
    )

    # ========================================================
    # CREATE PDF
    # ========================================================

    pdf_filename = (
        unique_id
        + "_retinal_report.pdf"
    )

    pdf_path = (
        UPLOAD_FOLDER
        / pdf_filename
    )

    create_pdf_report(

        pdf_path=pdf_path,

        original_image_path=input_path,

        detection_image_path=detection_path,

        patient_id=patient_id,

        patient_name=patient_name,

        patient_age=patient_age,

        patient_gender=patient_gender,

        examiner_name=examiner_name,

        examination_date=display_date,

        detection_summary=detection_summary,

        total_detections=total_detections,

        final_result=final_result,

        report_id=report_id,

        analysis_timestamp=analysis_timestamp

    )
# ========================================================
# SAVE REPORT TO DATABASE
# ========================================================

    # ========================================================
    # SAVE REPORT TO DATABASE
    # ========================================================

    analysis_datetime = datetime.now().strftime(
        "%d %B %Y, %I:%M %p"
    )

    save_report(
        user_id=session["user_id"],
        report_id=report_id,
        patient_id=patient_id,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_gender=patient_gender,
        examiner_name=examiner_name,
        examination_date=display_date,
        analysis_datetime=analysis_datetime,
        total_detections=total_detections,
        final_result=final_result,
        result_status=result_status,
        original_image=original_image,
        detection_image=result_image,
        pdf_filename=pdf_filename,
        lesion_counts={
            class_name: data["count"]
            for class_name, data in detection_summary.items()
        },
        doctor_remarks=request.form.get("doctor_remarks", "").strip()
    )

    # ========================================================
    # RETURN RESULT PAGE
    # ========================================================

    return render_template(
        "index.html",
        original_image=original_image,
        result_image=result_image,
        pdf_filename=pdf_filename,
        report_id=report_id,
        analysis_datetime=analysis_datetime,
        detection_summary=detection_summary,
        final_result=final_result,
        result_status=result_status,
        total_detections=total_detections,
        patient_id=patient_id,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_gender=patient_gender,
        examiner_name=examiner_name,
        examination_date=display_date,
        lesion_counts=None
    )

# ============================================================
# DOWNLOAD PDF
# ============================================================

@app.route(
    "/download-report/<filename>"
)
@login_required
def download_report(filename):

    # Only allow a user to download a PDF attached to their own report.
    report = get_report_by_pdf_filename(filename, session["user_id"])
    if report is None:
        return "PDF report not found.", 404

    pdf_path = (
        UPLOAD_FOLDER
        / filename
    )

    # --------------------------------------------------------
    # Security check
    # --------------------------------------------------------

    try:

        pdf_path = pdf_path.resolve()

        upload_folder_resolved = (
            UPLOAD_FOLDER.resolve()
        )

        pdf_path.relative_to(
            upload_folder_resolved
        )

    except ValueError:

        return "Invalid file.", 400

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not pdf_path.exists():

        return (
            "PDF report not found.",
            404
        )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    return send_file(

        pdf_path,

        as_attachment=True,

        download_name=(
            "RetinaAI_Report.pdf"
        ),

        mimetype=(
            "application/pdf"
        )

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )