# ---
# CampusAI Suite ReportLab PDF Exporter
# Owner: Member 3 (Backend and database engineer)
# ---

import os
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# ReportLab flowables and layout imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("report_generator")

def generate_attendance_pdf(
    session_id: int,
    section: str,
    subject: str,
    teacher_id: str,
    created_at: datetime,
    records: List[Dict[str, Any]]
) -> bytes:
    """
    Generates a professional attendance sheet PDF for SmartAttend.
    
    Args:
        session_id: int, the session database ID.
        section: str, class section name.
        subject: str, course subject.
        teacher_id: str, teacher ID.
        created_at: datetime, session creation timestamp.
        records: List of dicts representing attendance status of students:
                 [{"student_id": "S001", "name": "Arif", "status": "present", "confidence": 0.8200, "overridden": 0}]
                 
    Returns:
        bytes: The generated PDF binary data.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=15
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569')
    )
    
    th_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.white
    )
    
    td_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#0F172A')
    )
    
    story = []
    
    # Header Section
    story.append(Paragraph("SmartAttend AI — Attendance Session Report", title_style))
    
    meta_text = f"""
    <b>Subject:</b> {subject} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Section:</b> {section}<br/>
    <b>Teacher ID:</b> {teacher_id} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Session ID:</b> #{session_id}<br/>
    <b>Date & Time:</b> {created_at.strftime('%Y-%m-%d %I:%M %p')}
    """
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 20))
    
    # Calculate statistics
    total = len(records)
    present_cnt = sum(1 for r in records if r["status"] == "present")
    low_conf_cnt = sum(1 for r in records if r["status"] == "low_confidence")
    absent_cnt = sum(1 for r in records if r["status"] == "absent")
    overridden_cnt = sum(1 for r in records if r.get("overridden", 0) == 1)
    
    stats_data = [
        [
            Paragraph(f"<b>Total Enrolled:</b> {total}", meta_style),
            Paragraph(f"<b>Present:</b> {present_cnt}", meta_style),
            Paragraph(f"<b>Low Confidence:</b> {low_conf_cnt}", meta_style),
            Paragraph(f"<b>Absent:</b> {absent_cnt}", meta_style),
            Paragraph(f"<b>Manual Overrides:</b> {overridden_cnt}", meta_style),
        ]
    ]
    stats_table = Table(stats_data, colWidths=[1.5*inch, 1.2*inch, 1.6*inch, 1.2*inch, 1.7*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 20))
    
    # Records Table
    table_data = [[
        Paragraph("Student ID", th_style),
        Paragraph("Name", th_style),
        Paragraph("Status", th_style),
        Paragraph("Confidence", th_style),
        Paragraph("Manual Correction", th_style)
    ]]
    
    for record in records:
        status_label = record["status"].replace("_", " ").title()
        status_color = '#10B981' if record["status"] == "present" else ('#F59E0B' if record["status"] == "low_confidence" else '#EF4444')
        
        status_paragraph = Paragraph(f"<font color='{status_color}'><b>{status_label}</b></font>", td_style)
        
        override_label = "Yes (Teacher Override)" if record.get("overridden", 0) == 1 else "No"
        override_color = '#3B82F6' if record.get("overridden", 0) == 1 else '#64748B'
        override_paragraph = Paragraph(f"<font color='{override_color}'>{override_label}</font>", td_style)
        
        conf_val = record.get("confidence", 0.0)
        conf_str = f"{conf_val:.4f}" if conf_val > 0 else "N/A"
        
        table_data.append([
            Paragraph(record["student_id"], td_style),
            Paragraph(record["name"], td_style),
            status_paragraph,
            Paragraph(conf_str, td_style),
            override_paragraph
        ])
        
    records_table = Table(table_data, colWidths=[1.1*inch, 2.3*inch, 1.4*inch, 1.1*inch, 1.6*inch])
    records_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('TOPPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    
    story.append(records_table)
    
    # Build Document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def generate_exam_report_pdf(
    session_id: int,
    course: str,
    hall: str,
    invigilator: str,
    started_at: datetime,
    ended_at: Optional[datetime],
    student_summaries: List[Dict[str, Any]],
    events: List[Dict[str, Any]]
) -> bytes:
    """
    Generates a high-quality integrity report for ExamShield sessions.
    Includes student overall anomaly scores, event timelines, and screenshots.
    
    Args:
        session_id: int, exam session ID.
        course: str, course code/title.
        hall: str, exam hall room.
        invigilator: str, proctor/invigilator name.
        started_at: datetime, exam start timestamp.
        ended_at: datetime, exam end timestamp.
        student_summaries: List of student records with final anomaly scores:
                           [{"student_id": "S001", "name": "Arif", "score": 85, "risk": "High Risk", "color": "red"}]
        events: List of individual logged anomaly events with screenshots:
                [{"student_id": "S001", "event_type": "phone", "score_delta": 50, "occurred_at": datetime, "screenshot_path": "path"}]
                
    Returns:
        bytes: PDF binary.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'ExamTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=12
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )
    
    meta_style = ParagraphStyle(
        'ExamMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#475569')
    )
    
    th_style = ParagraphStyle(
        'ExamTH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.white
    )
    
    td_style = ParagraphStyle(
        'ExamTD',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#1E293B')
    )
    
    story = []
    
    # Cover / Header
    story.append(Paragraph("ExamShield AI — Integrity Verification Report", title_style))
    
    end_str = ended_at.strftime('%Y-%m-%d %I:%M %p') if ended_at else "Active/In-Progress"
    meta_text = f"""
    <b>Course:</b> {course} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Invigilator:</b> {invigilator}<br/>
    <b>Exam Hall:</b> {hall} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Session ID:</b> #{session_id}<br/>
    <b>Start Time:</b> {started_at.strftime('%Y-%m-%d %I:%M %p')} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>End Time:</b> {end_str}
    """
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 15))
    
    # 1. Summary of Student Scores
    story.append(Paragraph("Student Anomaly Summary", section_style))
    
    table_data = [[
        Paragraph("Student ID", th_style),
        Paragraph("Name", th_style),
        Paragraph("Anomaly Score", th_style),
        Paragraph("Risk Status", th_style)
    ]]
    
    for summary in student_summaries:
        risk_color = '#EF4444' if summary["color"] == "red" else ('#F59E0B' if summary["color"] == "amber" else '#10B981')
        risk_paragraph = Paragraph(f"<font color='{risk_color}'><b>{summary['risk']}</b></font>", td_style)
        
        table_data.append([
            Paragraph(summary["student_id"], td_style),
            Paragraph(summary["name"], td_style),
            Paragraph(str(summary["score"]), td_style),
            risk_paragraph
        ])
        
    summary_table = Table(table_data, colWidths=[1.2*inch, 2.5*inch, 1.8*inch, 2.0*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # 2. Flagged Incidents & Screenshots
    high_risk_events = [e for e in events if e.get("screenshot_path")]
    
    if high_risk_events:
        story.append(Paragraph("Flagged Incident Details & Verification Screenshots", section_style))
        
        for ev in high_risk_events:
            event_type = ev["event_type"].replace("_", " ").title()
            occ_time = ev["occurred_at"].strftime('%I:%M:%S %p')
            
            ev_desc = f"""
            <b>Student:</b> {ev['student_id']} ({ev.get('name', 'Enrolled Student')}) <br/>
            <b>Incident:</b> {event_type} (+{ev['score_delta']} pts) <br/>
            <b>Time Logged:</b> {occ_time}
            """
            
            # Load screenshot image
            img_flowable = None
            scr_path = ev["screenshot_path"]
            if scr_path and os.path.exists(scr_path):
                try:
                    # Resize to fit neatly in report
                    img_flowable = Image(scr_path, width=3.2*inch, height=2.4*inch)
                    img_flowable.hAlign = 'LEFT'
                except Exception as ex:
                    logger.warning(f"Could not load image {scr_path}: {ex}")
                    
            # Put text and image in a side-by-side table layout
            incident_data = [
                [Paragraph(ev_desc, td_style), img_flowable or Paragraph("<i>Screenshot Unavailable</i>", td_style)]
            ]
            
            incident_table = Table(incident_data, colWidths=[3.5*inch, 3.5*inch])
            incident_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FEF2F2')),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#FCA5A5')),
                ('PADDING', (0,0), (-1,-1), 8),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]))
            
            # Keep each incident layout block together to avoid page-split ugly layouts
            story.append(KeepTogether([
                incident_table,
                Spacer(1, 10)
            ]))
    else:
        story.append(Spacer(1, 10))
        story.append(Paragraph("<i>No high-risk physical events with screenshots were logged in this session.</i>", td_style))
        
    # Build Document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
