import sqlite3 
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "retinaai.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT UNIQUE NOT NULL,
            user_id INTEGER,
            patient_id TEXT,
            patient_name TEXT,
            patient_age TEXT,
            patient_gender TEXT,
            examiner_name TEXT,
            examination_date TEXT,
            analysis_datetime TEXT,
            total_detections INTEGER,
            final_result TEXT,
            result_status TEXT,
            original_image TEXT,
            detection_image TEXT,
            pdf_filename TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    
    # Upgrade an existing RetinaAI database without deleting saved reports.
    columns = {
        row["name"]
        for row in cursor.execute("PRAGMA table_info(reports)").fetchall()
    }

    if "user_id" not in columns:
        cursor.execute(
            "ALTER TABLE reports ADD COLUMN user_id INTEGER REFERENCES users(id)"
        )

    # Refresh the column list after any schema change.
    columns = {
        row["name"]
        for row in cursor.execute("PRAGMA table_info(reports)").fetchall()
    }

    if "lesion_counts" not in columns:
        cursor.execute(
            "ALTER TABLE reports ADD COLUMN lesion_counts TEXT"
        )
    if "doctor_remarks" not in columns:
        cursor.execute(
        "ALTER TABLE reports ADD COLUMN doctor_remarks TEXT"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_reports_user_id ON reports(user_id)"
    )

    connection.commit()
    connection.close()


def create_user(name, email, password_hash):
    connection = get_connection()
    try:
        cursor = connection.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash),
        )
        connection.commit()
        return cursor.lastrowid
    finally:
        connection.close()


def get_user_by_email(email):
    connection = get_connection()
    try:
        return connection.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ? COLLATE NOCASE",
            (email,),
        ).fetchone()
    finally:
        connection.close()
def email_exists_for_other_user(email, user_id):
    connection = get_connection()
    try:
        return connection.execute(
            """
            SELECT id FROM users
            WHERE email = ? COLLATE NOCASE
            AND id != ?
            """,
            (email, user_id)
        ).fetchone() is not None
    finally:
        connection.close()


def get_user_by_id(user_id):
    connection = get_connection()
    try:
        return connection.execute(
            "SELECT id, name, email FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    finally:
        connection.close()
def update_user_profile(user_id, name, email):
    connection = get_connection()
    try:
        connection.execute(
            """
            UPDATE users
            SET name = ?, email = ?
            WHERE id = ?
            """,
            (name, email, user_id)
        )
        connection.commit()
    finally:
        connection.close()

def update_user_password(user_id, password_hash):
    connection = get_connection()
    try:
        connection.execute(
            """
            UPDATE users
            SET password_hash = ?
            WHERE id = ?
            """,
            (password_hash, user_id)
        )
        connection.commit()
    finally:
        connection.close()



def save_report(
    user_id, report_id, patient_id, patient_name, patient_age, patient_gender,
    examiner_name, examination_date, analysis_datetime, total_detections,
    final_result, result_status, original_image, detection_image, pdf_filename,
    lesion_counts=None,
    doctor_remarks=None,

):
    connection = get_connection()
    try:
        connection.execute("""
            INSERT INTO reports (
                user_id, report_id, patient_id, patient_name, patient_age,
                patient_gender, examiner_name, examination_date, analysis_datetime,
                total_detections, final_result, result_status, original_image,
                detection_image, pdf_filename, lesion_counts,doctor_remarks
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, report_id, patient_id, patient_name, patient_age,
            patient_gender, examiner_name, examination_date, analysis_datetime,
            total_detections, final_result, result_status, original_image,
            detection_image, pdf_filename,
            json.dumps(lesion_counts or {}),doctor_remarks,
        ))
        connection.commit()
    finally:
        connection.close()

def update_doctor_remarks(report_id, user_id, doctor_remarks):
    connection = get_connection()
    try:
        connection.execute(
            """
            UPDATE reports
            SET doctor_remarks = ?
            WHERE report_id = ? AND user_id = ?
            """,
            (doctor_remarks, report_id, user_id)
        )
        connection.commit()
    finally:
        connection.close()



def get_all_reports(user_id):
    connection = get_connection()
    try:
        return connection.execute(
            "SELECT * FROM reports WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall()
    finally:
        connection.close()


def get_report(report_id, user_id):
    connection = get_connection()
    try:
        return connection.execute(
            "SELECT * FROM reports WHERE report_id = ? AND user_id = ?",
            (report_id, user_id),
        ).fetchone()
    finally:
        connection.close()


def get_report_by_pdf_filename(pdf_filename, user_id):
    connection = get_connection()
    try:
        return connection.execute(
            "SELECT * FROM reports WHERE pdf_filename = ? AND user_id = ?",
            (pdf_filename, user_id),
        ).fetchone()
    finally:
        connection.close()


if __name__ == "__main__":
    initialize_database()
    print("RetinaAI database initialized at:", DATABASE_PATH)
