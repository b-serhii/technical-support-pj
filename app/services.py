from __future__ import annotations

import html
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import func
from sqlalchemy.orm import Session

from .config import BASE_DIR, CONFIG
from .models import Group, PredictionLog, Student
from .schemas import Metrics

LOG_PATH = BASE_DIR / CONFIG['log_file']
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = BASE_DIR / CONFIG['reports_dir']
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = BASE_DIR / CONFIG['model_path']
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(LOG_PATH, encoding='utf-8'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

FEATURES = ['attendance', 'average_score', 'activity_level', 'missed_classes', 'module_score']


def seed_demo_data(db: Session) -> None:
    if db.query(Group).count() > 0:
        return

    g1 = Group(name='КН-21', specialty='Комп’ютерні науки', curator='Ірина Мельник', semester='4', description='Демо-група для тестування системи.')
    g2 = Group(name='ІПЗ-22', specialty='Інженерія програмного забезпечення', curator='Олег Кравчук', semester='3', description='Навчальна група з прикладного програмування.')
    db.add_all([g1, g2])
    db.flush()

    demo_students = [
        (g1.id, 'Андрій Савчук', 'andrii@example.com', 95, 91, 9, 1, 88, 1),
        (g1.id, 'Марія Гнатюк', 'mariia@example.com', 88, 84, 8, 2, 79, 1),
        (g1.id, 'Олег Бойко', 'oleh@example.com', 61, 58, 4, 8, 55, 0),
        (g1.id, 'Софія Романюк', 'sofiia@example.com', 72, 69, 6, 5, 70, 1),
        (g1.id, 'Іван Петрик', 'ivan@example.com', 45, 50, 3, 12, 47, 0),
        (g2.id, 'Наталія Коваль', 'nataliia@example.com', 97, 93, 10, 0, 95, 1),
        (g2.id, 'Юрій Дячук', 'yurii@example.com', 67, 64, 5, 6, 62, 0),
        (g2.id, 'Вікторія Клименко', 'vika@example.com', 80, 77, 7, 3, 81, 1),
        (g2.id, 'Максим Ільницький', 'maksym@example.com', 54, 57, 4, 10, 59, 0),
        (g2.id, 'Аліна Паламар', 'alina@example.com', 90, 86, 8, 2, 84, 1),
        (g2.id, 'Роман Федорук', 'roman@example.com', 76, 73, 6, 4, 71, 1),
        (g1.id, 'Христина Сенько', 'kh@example.com', 52, 49, 4, 11, 53, 0),
    ]

    for group_id, full_name, email, attendance, average_score, activity_level, missed_classes, module_score, exam_passed in demo_students:
        db.add(Student(
            group_id=group_id,
            full_name=full_name,
            email=email,
            attendance=attendance,
            average_score=average_score,
            activity_level=activity_level,
            missed_classes=missed_classes,
            module_score=module_score,
            exam_passed=exam_passed,
        ))

    db.commit()
    logger.info('Створено демо-дані для груп і студентів')


def validate_student_form(form: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, float | int | str]]:
    errors: Dict[str, str] = {}

    def as_float(name: str, min_v: float, max_v: float) -> float:
        raw = str(form.get(name, '')).strip().replace(',', '.')
        try:
            value = float(raw)
        except ValueError:
            errors[name] = 'Введіть коректне число.'
            return 0.0
        if not (min_v <= value <= max_v):
            errors[name] = f'Значення має бути в межах {min_v}–{max_v}.'
        return value

    def as_int(name: str, min_v: int, max_v: int) -> int:
        raw = str(form.get(name, '')).strip()
        try:
            value = int(raw)
        except ValueError:
            errors[name] = 'Введіть коректне ціле число.'
            return 0
        if not (min_v <= value <= max_v):
            errors[name] = f'Значення має бути в межах {min_v}–{max_v}.'
        return value

    full_name = str(form.get('full_name', '')).strip()
    if len(full_name) < 5:
        errors['full_name'] = 'Вкажіть ПІБ студента.'

    email = str(form.get('email', '')).strip()
    attendance = as_float('attendance', 0, 100)
    average_score = as_float('average_score', 0, 100)
    activity_level = as_float('activity_level', 0, 10)
    missed_classes = as_int('missed_classes', 0, 100)
    module_score = as_float('module_score', 0, 100)
    exam_passed = 1 if str(form.get('exam_passed', '1')).strip() == '1' else 0

    payload = {
        'full_name': full_name,
        'email': email,
        'attendance': attendance,
        'average_score': average_score,
        'activity_level': activity_level,
        'missed_classes': missed_classes,
        'module_score': module_score,
        'exam_passed': exam_passed,
    }
    return errors, payload


def validate_group_form(form: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    errors = {}
    payload = {
        'name': str(form.get('name', '')).strip(),
        'specialty': str(form.get('specialty', '')).strip(),
        'curator': str(form.get('curator', '')).strip(),
        'semester': str(form.get('semester', '')).strip(),
        'description': str(form.get('description', '')).strip(),
    }
    for field in ['name', 'specialty', 'curator', 'semester']:
        if len(payload[field]) < 2:
            errors[field] = 'Поле обов’язкове.'
    return errors, payload


def get_dashboard_stats(db: Session) -> Dict[str, float | int]:
    total_groups = db.query(func.count(Group.id)).scalar() or 0
    total_students = db.query(func.count(Student.id)).scalar() or 0
    passed_students = db.query(func.count(Student.id)).filter(Student.exam_passed == 1).scalar() or 0
    predicted_count = db.query(func.count(PredictionLog.id)).scalar() or 0
    pass_rate = round((passed_students / total_students) * 100, 1) if total_students else 0
    return {
        'total_groups': total_groups,
        'total_students': total_students,
        'pass_rate': pass_rate,
        'predicted_count': predicted_count,
    }


def train_model(db: Session) -> Metrics:
    students = db.query(Student).all()
    if len(students) < 8:
        raise ValueError('Недостатньо даних для навчання моделі. Додайте більше студентів.')

    X = np.array([[getattr(s, f) for f in FEATURES] for s in students], dtype=float)
    y = np.array([s.exam_passed for s in students], dtype=int)

    if len(np.unique(y)) < 2:
        raise ValueError('Для навчання потрібні студенти обох класів: і ті, хто склав, і ті, хто не склав.')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = Metrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        train_size=len(X_train),
        test_size=len(X_test),
    )

    joblib.dump({'model': model, 'metrics': metrics.__dict__}, MODEL_PATH)
    logger.info('Модель успішно навчено. Accuracy=%.3f', metrics.accuracy)
    return metrics


def load_model_info() -> Dict | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_student(db: Session, student: Student) -> Tuple[str, float]:
    payload = load_model_info()
    if not payload:
        raise ValueError('Модель ще не навчена. Спочатку виконайте навчання.')

    model = payload['model']
    X = np.array([[student.attendance, student.average_score, student.activity_level, student.missed_classes, student.module_score]], dtype=float)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    result = 'Склав' if pred == 1 else 'Не склав'

    student.prediction_result = result
    student.prediction_probability = proba
    db.add(student)
    db.add(PredictionLog(
        student_name=student.full_name,
        group_name=student.group.name,
        model_name='Logistic Regression',
        result=result,
        probability=proba,
    ))
    db.commit()
    logger.info('Виконано прогноз для студента %s: %s (%.2f%%)', student.full_name, result, proba * 100)
    return result, proba


def generate_group_report(db: Session, group: Group) -> Path:
    students = db.query(Student).filter(Student.group_id == group.id).order_by(Student.full_name.asc()).all()
    total = len(students)
    passed = sum(1 for s in students if s.exam_passed == 1)
    predicted = sum(1 for s in students if s.prediction_result != '—')
    avg_attendance = round(sum(s.attendance for s in students) / total, 1) if total else 0
    avg_score = round(sum(s.average_score for s in students) / total, 1) if total else 0

    rows = ''.join(
        f"<tr><td>{idx}</td><td>{html.escape(s.full_name)}</td><td>{s.attendance:.0f}%</td><td>{s.average_score:.1f}</td><td>{s.activity_level:.1f}</td><td>{s.module_score:.1f}</td><td>{'Склав' if s.exam_passed else 'Не склав'}</td><td>{html.escape(s.prediction_result)}</td><td>{s.prediction_probability * 100:.1f}%</td></tr>"
        for idx, s in enumerate(students, start=1)
    )
    if not rows:
        rows = '<tr><td colspan="9">У групі ще немає студентів.</td></tr>'

    generated_at = datetime.now().strftime('%d.%m.%Y %H:%M')
    report_html = f"""<!doctype html>
<html lang='uk'>
<head>
<meta charset='utf-8'>
<title>Звіт групи {html.escape(group.name)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #132238; }}
header {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:24px; }}
.card {{ border:1px solid #d9e2ef; border-radius:16px; padding:16px; background:#f8fbff; }}
.grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin:20px 0; }}
.value {{ font-size:28px; font-weight:700; margin-top:8px; }}
table {{ width:100%; border-collapse: collapse; margin-top:18px; }}
th, td {{ border:1px solid #d9e2ef; padding:10px; text-align:left; font-size:14px; }}
th {{ background:#ecf4ff; }}
.badge {{ display:inline-block; padding:6px 10px; border-radius:999px; background:#dff5e8; }}
.footer {{ margin-top:20px; font-size:12px; color:#5b6b7e; }}
</style>
</head>
<body>
<header>
<div>
<h1>Звіт по групі {html.escape(group.name)}</h1>
<p><strong>Спеціальність:</strong> {html.escape(group.specialty)}<br>
<strong>Куратор:</strong> {html.escape(group.curator)}<br>
<strong>Семестр:</strong> {html.escape(group.semester)}</p>
</div>
<div class='badge'>Сформовано: {generated_at}</div>
</header>
<div class='grid'>
<div class='card'><div>Кількість студентів</div><div class='value'>{total}</div></div>
<div class='card'><div>Склали фактично</div><div class='value'>{passed}</div></div>
<div class='card'><div>Є прогнозів</div><div class='value'>{predicted}</div></div>
<div class='card'><div>Середній бал</div><div class='value'>{avg_score}</div></div>
</div>
<div class='card'><strong>Середня відвідуваність:</strong> {avg_attendance}%<br><strong>Опис групи:</strong> {html.escape(group.description or '—')}</div>
<table>
<thead>
<tr><th>#</th><th>Студент</th><th>Відвідуваність</th><th>Сер. бал</th><th>Активність</th><th>Модуль</th><th>Факт</th><th>Прогноз</th><th>Ймовірність</th></tr>
</thead>
<tbody>{rows}</tbody>
</table>
<div class='footer'>Навчальний звіт вебсистеми прогнозування успішності студентів.</div>
</body>
</html>"""

    filename = REPORTS_DIR / f"group_{group.id}_report.html"
    filename.write_text(report_html, encoding='utf-8')
    logger.info('Сформовано звіт для групи %s', group.name)
    return filename
