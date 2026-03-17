from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, joinedload

from .config import BASE_DIR, CONFIG
from .database import Base, engine, get_db
from .models import Group, PredictionLog, Student
from .services import (
    generate_group_report,
    get_dashboard_stats,
    load_model_info,
    predict_student,
    seed_demo_data,
    train_model,
    validate_group_form,
    validate_student_form,
)

app = FastAPI(title=CONFIG['app_name'])
app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'app' / 'static')), name='static')
templates = Jinja2Templates(directory=str(BASE_DIR / 'app' / 'templates'))

Base.metadata.create_all(bind=engine)

@app.on_event('startup')
def startup_event():
    from .database import SessionLocal
    db = SessionLocal()
    try:
        seed_demo_data(db)
    finally:
        db.close()


def render(request: Request, template: str, **context):
    model_payload = load_model_info()
    base_context = {
        'request': request,
        'app_name': CONFIG['app_name'],
        'model_metrics': model_payload['metrics'] if model_payload else None,
    }
    base_context.update(context)
    return templates.TemplateResponse(template, base_context)

@app.get('/', response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    stats = get_dashboard_stats(db)
    groups = db.query(Group).options(joinedload(Group.students)).order_by(Group.name.asc()).all()
    latest_predictions = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(5).all()
    return render(request, 'index.html', stats=stats, groups=groups, latest_predictions=latest_predictions)

@app.get('/groups', response_class=HTMLResponse)
def groups_page(request: Request, db: Session = Depends(get_db)):
    groups = db.query(Group).options(joinedload(Group.students)).order_by(Group.created_at.desc()).all()
    return render(request, 'groups.html', groups=groups, errors={}, form={})

@app.post('/groups', response_class=HTMLResponse)
def create_group(
    request: Request,
    name: str = Form(...),
    specialty: str = Form(...),
    curator: str = Form(...),
    semester: str = Form(...),
    description: str = Form(''),
    db: Session = Depends(get_db),
):
    form = {'name': name, 'specialty': specialty, 'curator': curator, 'semester': semester, 'description': description}
    errors, payload = validate_group_form(form)
    if errors:
        groups = db.query(Group).options(joinedload(Group.students)).order_by(Group.created_at.desc()).all()
        return render(request, 'groups.html', groups=groups, errors=errors, form=form)

    group = Group(**payload)
    db.add(group)
    db.commit()
    return RedirectResponse('/groups', status_code=303)

@app.get('/groups/{group_id}', response_class=HTMLResponse)
def group_detail(group_id: int, request: Request, db: Session = Depends(get_db)):
    group = db.query(Group).options(joinedload(Group.students)).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail='Групу не знайдено')
    return render(request, 'group_detail.html', group=group, errors={}, form={})

@app.get('/groups/{group_id}/edit', response_class=HTMLResponse)
def edit_group_page(group_id: int, request: Request, db: Session = Depends(get_db)):
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404)
    return render(request, 'group_edit.html', group=group, errors={})

@app.post('/groups/{group_id}/edit', response_class=HTMLResponse)
def update_group(
    group_id: int,
    request: Request,
    name: str = Form(...),
    specialty: str = Form(...),
    curator: str = Form(...),
    semester: str = Form(...),
    description: str = Form(''),
    db: Session = Depends(get_db),
):
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404)
    form = {'name': name, 'specialty': specialty, 'curator': curator, 'semester': semester, 'description': description}
    errors, payload = validate_group_form(form)
    if errors:
        return render(request, 'group_edit.html', group=group, errors=errors)
    for key, value in payload.items():
        setattr(group, key, value)
    db.commit()
    return RedirectResponse(f'/groups/{group_id}', status_code=303)

@app.post('/groups/{group_id}/delete')
def delete_group(group_id: int, db: Session = Depends(get_db)):
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404)
    db.delete(group)
    db.commit()
    return RedirectResponse('/groups', status_code=303)

@app.post('/groups/{group_id}/students/create', response_class=HTMLResponse)
def create_student(
    group_id: int,
    request: Request,
    full_name: str = Form(...),
    email: str = Form(''),
    attendance: str = Form(...),
    average_score: str = Form(...),
    activity_level: str = Form(...),
    missed_classes: str = Form(...),
    module_score: str = Form(...),
    exam_passed: str = Form('1'),
    db: Session = Depends(get_db),
):
    group = db.query(Group).options(joinedload(Group.students)).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404)
    form = {
        'full_name': full_name, 'email': email, 'attendance': attendance, 'average_score': average_score,
        'activity_level': activity_level, 'missed_classes': missed_classes, 'module_score': module_score,
        'exam_passed': exam_passed,
    }
    errors, payload = validate_student_form(form)
    if errors:
        return render(request, 'group_detail.html', group=group, errors=errors, form=form)
    student = Student(group_id=group_id, **payload)
    db.add(student)
    db.commit()
    return RedirectResponse(f'/groups/{group_id}', status_code=303)

@app.get('/students/{student_id}/edit', response_class=HTMLResponse)
def edit_student_page(student_id: int, request: Request, db: Session = Depends(get_db)):
    student = db.query(Student).options(joinedload(Student.group)).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404)
    return render(request, 'student_edit.html', student=student, errors={})

@app.post('/students/{student_id}/edit', response_class=HTMLResponse)
def update_student(
    student_id: int,
    request: Request,
    full_name: str = Form(...),
    email: str = Form(''),
    attendance: str = Form(...),
    average_score: str = Form(...),
    activity_level: str = Form(...),
    missed_classes: str = Form(...),
    module_score: str = Form(...),
    exam_passed: str = Form('1'),
    db: Session = Depends(get_db),
):
    student = db.query(Student).options(joinedload(Student.group)).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404)
    form = {
        'full_name': full_name, 'email': email, 'attendance': attendance, 'average_score': average_score,
        'activity_level': activity_level, 'missed_classes': missed_classes, 'module_score': module_score,
        'exam_passed': exam_passed,
    }
    errors, payload = validate_student_form(form)
    if errors:
        return render(request, 'student_edit.html', student=student, errors=errors)
    for key, value in payload.items():
        setattr(student, key, value)
    db.commit()
    return RedirectResponse(f'/groups/{student.group_id}', status_code=303)

@app.post('/students/{student_id}/delete')
def delete_student(student_id: int, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404)
    group_id = student.group_id
    db.delete(student)
    db.commit()
    return RedirectResponse(f'/groups/{group_id}', status_code=303)

@app.post('/students/{student_id}/predict')
def predict(student_id: int, db: Session = Depends(get_db)):
    student = db.query(Student).options(joinedload(Student.group)).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404)
    predict_student(db, student)
    return RedirectResponse(f'/groups/{student.group_id}', status_code=303)

@app.get('/train', response_class=HTMLResponse)
def train_page(request: Request):
    return render(request, 'train.html', trained=False, error=None)

@app.post('/train', response_class=HTMLResponse)
def train_page_post(request: Request, db: Session = Depends(get_db)):
    try:
        metrics = train_model(db)
        return render(request, 'train.html', trained=True, metrics=metrics, error=None)
    except ValueError as exc:
        return render(request, 'train.html', trained=False, error=str(exc), metrics=None)

@app.get('/predictions', response_class=HTMLResponse)
def predictions_page(request: Request, db: Session = Depends(get_db)):
    predictions = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).all()
    return render(request, 'predictions.html', predictions=predictions)

@app.get('/reports/{group_id}', response_class=HTMLResponse)
def report_preview(group_id: int, request: Request, db: Session = Depends(get_db)):
    group = db.query(Group).options(joinedload(Group.students)).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404)
    path = generate_group_report(db, group)
    html_content = Path(path).read_text(encoding='utf-8')
    return HTMLResponse(content=html_content)

@app.get('/reports/{group_id}/download')
def report_download(group_id: int, db: Session = Depends(get_db)):
    group = db.query(Group).options(joinedload(Group.students)).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404)
    path = generate_group_report(db, group)
    return FileResponse(path, filename=path.name, media_type='text/html')
