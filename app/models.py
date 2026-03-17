from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from .database import Base

class Group(Base):
    __tablename__ = 'groups'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    specialty = Column(String(120), nullable=False)
    curator = Column(String(120), nullable=False)
    semester = Column(String(50), nullable=False)
    description = Column(Text, default='')
    created_at = Column(DateTime, default=datetime.utcnow)

    students = relationship('Student', back_populates='group', cascade='all, delete-orphan')

class Student(Base):
    __tablename__ = 'students'

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(150), nullable=False)
    email = Column(String(150), default='')
    attendance = Column(Float, nullable=False)
    average_score = Column(Float, nullable=False)
    activity_level = Column(Float, nullable=False)
    missed_classes = Column(Integer, nullable=False)
    module_score = Column(Float, nullable=False)
    exam_passed = Column(Integer, nullable=False, default=1)
    prediction_result = Column(String(50), default='—')
    prediction_probability = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    group_id = Column(Integer, ForeignKey('groups.id', ondelete='CASCADE'), nullable=False)
    group = relationship('Group', back_populates='students')

class PredictionLog(Base):
    __tablename__ = 'prediction_logs'

    id = Column(Integer, primary_key=True, index=True)
    student_name = Column(String(150), nullable=False)
    group_name = Column(String(100), nullable=False)
    model_name = Column(String(100), nullable=False)
    result = Column(String(50), nullable=False)
    probability = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
