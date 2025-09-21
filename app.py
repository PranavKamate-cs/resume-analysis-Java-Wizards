import streamlit as st
import os
import sqlite3
import re
import pandas as pd
import numpy as np
import PyPDF2
import docx
import spacy
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime, date

# --- App Setup and Constants ---
DB_FILE = "placement_portal.db"
ANALYSIS_DB_FILE = "analysis_results.db"

# --- Resource Loading (Cached for Performance) ---
@st.cache_resource
def load_resources():
    """Loads models and initializes databases once."""
    print("Loading resources...")
    nlp = spacy.load("en_core_web_sm")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API key not found. Please add it to your Streamlit secrets.", icon="🚨")
        return None, None, None
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=google_api_key,
        convert_system_message_to_human=True
    )
    
    init_db()
    init_analysis_db()
    print("Resources loaded successfully.")
    return nlp, semantic_model, llm

# --- Database Functions ---
def get_db_connection(db_file):
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME NOT NULL, title TEXT NOT NULL,
                description TEXT NOT NULL, due_date DATE NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL, timestamp DATETIME NOT NULL,
                candidate_name TEXT NOT NULL, candidate_email TEXT NOT NULL, final_score REAL,
                ai_feedback TEXT, verdict TEXT, lacking_skills TEXT,
                status TEXT DEFAULT 'Applied', sim_gender TEXT, sim_university_tier TEXT,
                FOREIGN KEY (job_id) REFERENCES jobs (id)
            )""")
        conn.commit()

def init_analysis_db():
    with get_db_connection(ANALYSIS_DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME NOT NULL, job_title TEXT NOT NULL,
                filename TEXT NOT NULL, score REAL, verdict TEXT, lacking_skills TEXT, feedback TEXT
            )""")
        conn.commit()

def add_job(title, description, due_date):
    with get_db_connection(DB_FILE) as conn:
        conn.execute("INSERT INTO jobs (timestamp, title, description, due_date) VALUES (?, ?, ?, ?)",
                       (datetime.now(), title, description, due_date))
        conn.commit()

def get_all_jobs():
    with get_db_connection(DB_FILE) as conn:
        return pd.read_sql_query("SELECT id, title, description, due_date FROM jobs ORDER BY timestamp DESC", conn)

def add_application(job_id, name, email, score, feedback, verdict, lacking_skills, gender, uni_tier):
    with get_db_connection(DB_FILE) as conn:
        conn.execute("""
            INSERT INTO applications (job_id, timestamp, candidate_name, candidate_email, final_score, ai_feedback, verdict, lacking_skills, sim_gender, sim_university_tier)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (job_id, datetime.now(), name, email, score, feedback, verdict, lacking_skills, gender, uni_tier))
        conn.commit()

def get_applications_for_job(job_id):
    with get_db_connection(DB_FILE) as conn:
        return pd.read_sql_query("SELECT id, candidate_name, candidate_email, final_score, verdict, status, lacking_skills, sim_gender, sim_university_tier FROM applications WHERE job_id = ? ORDER BY final_score DESC", conn, params=(job_id,))

def update_candidate_status(application_id, new_status):
    with get_db_connection(DB_FILE) as conn:
        conn.execute("UPDATE applications SET status = ? WHERE id = ?", (new_status, application_id))
        conn.commit()

def get_student_applications(email):
    with get_db_connection(DB_FILE) as conn:
        query = """
        SELECT j.title, a.status, a.final_score, a.ai_feedback, a.verdict
        FROM applications a JOIN jobs j ON a.job_id = j.id
        WHERE a.candidate_email = ? ORDER BY a.timestamp DESC
        """
        return pd.read_sql_query(query, conn, params=(email,))

def add_analysis_record(job_title, result):
    with get_db_connection(ANALYSIS_DB_FILE) as conn:
        conn.execute("""
            INSERT INTO analyses (timestamp, job_title, filename, score, verdict, lacking_skills, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now(), job_title, result['filename'], result['score'], result['verdict'], result['lacking_skills'], result['feedback']))
        conn.commit()

def get_all_analyses():
    with get_db_connection(ANALYSIS_DB_FILE) as conn:
        return pd.read_sql_query("SELECT timestamp, job_title, filename, score, verdict, lacking_skills FROM analyses ORDER BY timestamp DESC", conn)

# --- Helper & Analysis Functions ---
def read_pdf(file_object):
    pdf_reader = PyPDF2.PdfReader(file_object)
    return "".join([page.extract_text() for page in pdf_reader.pages])

def read_docx(file_object):
    doc = docx.Document(file_object)
    return "\n".join([para.text for para in doc.paragraphs])

def read_txt(file_object):
    return file_object.read().decode('utf-8')

def redact_pii(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(\(?\d{3}\)?[-.\s]?)?(\d{3}[-.\s]?\d{4})'
    redacted_text = re.sub(email_pattern, '[REDACTED EMAIL]', text)
    redacted_text = re.sub(phone_pattern, '[REDACTED PHONE]', redacted_text)
    return redacted_text

def improved_extract_keywords(text):
    doc = nlp(text.lower())
    keywords = [token.lemma_ for token in doc if (not token.is_stop and not token.is_punct and token.pos_ in ['PROPN', 'NOUN', 'ADJ'])]
    return [word for word, _ in Counter(keywords).most_common(15)]

def generate_ai_feedback_langchain(_jd_text, _resume_text, _job_title):
    prompt_template = """
    You are an expert, impartial, and ethical career coach AI. Your primary goal is to provide fair and objective feedback.
    **CRITICAL INSTRUCTIONS FOR FAIRNESS:**
    1.  **Evaluate based ONLY on skills and experience** directly relevant to the job description.
    2.  **DO NOT penalize for employment gaps, non-traditional career paths, or unconventional phrasing.** Focus on transferable skills.
    3.  **Give fair consideration to soft skills** (e.g., leadership, communication) demonstrated through project descriptions or roles.
    4.  **Ignore any personally identifiable information** such as names, emails, or phone numbers. Your analysis must be blind to personal identity.
    ---
    **JOB DESCRIPTION:**
    {jd}
    ---
    **RESUME:**
    {resume}
    ---
    Now, analyze the provided Resume against the Job Description and generate a feedback report strictly in the following Markdown format.
    **Overall Score:** [Provide a single integer score from 0 to 100.]
    **Verdict:** [A short, objective one-line verdict.]
    **Lacking Skills:** [List 2-3 key skills from the job description that are missing or weakly represented in the resume.]
    ---
    ### Resume Analysis for {job_title}
    #### ✅ Key Strengths (Job-Relevant)
    * **[Strength 1]:** [Explain why this is a strength by linking a specific part of the resume to a key requirement in the job description.]
    * **[Strength 2]:** [Provide another specific example of a strong, objective alignment.]
    #### 💡 Areas for Improvement
    * **[Suggestion 1]:** [Provide a concrete suggestion on how to better quantify achievements or tailor the resume.]
    **Final Summary:** [A brief, objective closing statement.]
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain.invoke({"jd": _jd_text, "resume": _resume_text, "job_title": _job_title})

def calculate_hybrid_score(jd_embedding, resume_text, jd_keywords, llm_score):
    resume_lower = resume_text.lower()
    matched_keywords = [kw for kw in jd_keywords if kw in resume_lower]
    hard_score = (len(matched_keywords) / len(jd_keywords)) * 100 if jd_keywords else 0
    resume_embedding = semantic_model.encode(resume_text, convert_to_tensor=True)
    soft_score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item() * 100
    return (0.3 * hard_score) + (0.5 * soft_score) + (0.2 * llm_score)

def analyze_resume(job_details, resume_file_object):
    job_description = job_details['description']
    job_title = job_details['title']
    
    file_name = resume_file_object.name
    if file_name.endswith('.pdf'): resume_text_raw = read_pdf(resume_file_object)
    elif file_name.endswith('.docx'): resume_text_raw = read_docx(resume_file_object)
    else: resume_text_raw = read_txt(resume_file_object)

    resume_text = redact_pii(resume_text_raw)
    raw_feedback = generate_ai_feedback_langchain(job_description, resume_text, job_title)
    
    llm_score, verdict, lacking_skills = 0.0, "N/A", "Not specified"
    score_match = re.search(r"\*\*Overall Score:\*\*\s*(\d{1,3})", raw_feedback)
    if score_match: llm_score = float(score_match.group(1))
    verdict_match = re.search(r"\*\*Verdict:\*\*\s*(.*)", raw_feedback)
    if verdict_match: verdict = verdict_match.group(1).strip()
    lacking_skills_match = re.search(r"\*\*Lacking Skills:\*\*\s*(.*)", raw_feedback)
    if lacking_skills_match: lacking_skills = lacking_skills_match.group(1).strip()

    jd_keywords = improved_extract_keywords(job_description)
    jd_embedding = semantic_model.encode(job_description, convert_to_tensor=True)
    final_score = calculate_hybrid_score(jd_embedding, resume_text, jd_keywords, llm_score)
    
    return {"filename": file_name, "score": final_score, "verdict": verdict, "lacking_skills": lacking_skills, "feedback": raw_feedback}

# --- UI Views ---
def student_view():
    st.title("🎓 Student Job Portal")
    st.info("Explore open positions below. After you apply, check your status to see AI-powered feedback.")
    jobs_df = get_all_jobs()
    if jobs_df.empty:
        st.info("No jobs posted yet.")
        return

    today = date.today()
    for _, job in jobs_df.iterrows():
        with st.expander(f"**{job['title']}**"):
            st.markdown(f"##### Job Description\n{job['description']}")
            job_due_date = datetime.strptime(job['due_date'], '%Y-%m-%d').date()
            if job_due_date:
                st.warning(f"**Application Deadline:** {job_due_date.strftime('%B %d, %Y')}")

            if job_due_date and job_due_date < today:
                st.error("Applications for this position are now closed.")
            else:
                with st.form(key=f"apply_form_{job['id']}"):
                    st.markdown("--- \n##### Apply Now")
                    student_name = st.text_input("Your Full Name")
                    student_email = st.text_input("Your Email Address")
                    uploaded_resume = st.file_uploader("Upload your resume", type=['pdf', 'docx', 'txt'])
                    if st.form_submit_button("Submit Application"):
                        if student_name and student_email and uploaded_resume:
                            with st.spinner("Analyzing and submitting..."):
                                job_details = {'title': job['title'], 'description': job['description']}
                                analysis = analyze_resume(job_details, uploaded_resume)
                                sim_gender = np.random.choice(["Male", "Female"], p=[0.6,0.4])
                                sim_uni_tier = np.random.choice(["Tier 1", "Tier 2/3"], p=[0.3,0.7])
                                add_application(job['id'], student_name, student_email, analysis['score'], 
                                                analysis['feedback'], analysis['verdict'], analysis['lacking_skills'], 
                                                sim_gender, sim_uni_tier)
                                st.success("Application submitted successfully!")
                        else: st.warning("Please fill all fields and upload your resume.")
    
    st.write("---")
    st.header("📋 Check Your Application Status")
    email_check = st.text_input("Enter your email address to check your applications:")
    if st.button("Check Status"):
        if email_check:
            apps_df = get_student_applications(email_check)
            if not apps_df.empty:
                for _, row in apps_df.iterrows():
                    with st.container(border=True):
                        st.subheader(row['title'])
                        cols = st.columns(3)
                        cols[0].metric("Your Final Score", f"{row['final_score']:.2f}%")
                        cols[1].metric("AI Verdict", row['verdict'])
                        status = row['status']
                        if status == 'Shortlisted': cols[2].success(f"Status: {status} 🎉")
                        elif status == 'Not Shortlisted': cols[2].error(f"Status: {status}")
                        else: cols[2].info(f"Status: {status}")
                        with st.expander("💡 View Detailed Feedback"):
                            st.markdown(row['ai_feedback'])
            else: st.info("No applications found for that email.")
        else: st.warning("Please enter your email.")


def bias_audit_dashboard(df):
    st.header("Bias & Fairness Audit Dashboard")
    st.info("This dashboard helps monitor the system for potential biases in shortlisting outcomes. Data shown here is simulated for demonstration purposes.")
    if len(df) < 10:
        st.warning("Insufficient data for a meaningful bias analysis. At least 10 applications are recommended.")
        return
    st.subheader("Success Rate Parity")
    st.markdown("This metric checks if candidates from different groups are being shortlisted at similar rates.")
    gender_df = df.groupby('sim_gender')['status'].value_counts(normalize=True).unstack().fillna(0)
    if 'Shortlisted' in gender_df.columns:
        st.markdown("**By Gender**"); st.bar_chart(gender_df['Shortlisted'])
    uni_df = df.groupby('sim_university_tier')['status'].value_counts(normalize=True).unstack().fillna(0)
    if 'Shortlisted' in uni_df.columns:
        st.markdown("**By University Tier (Proxy for Background)**"); st.bar_chart(uni_df['Shortlisted'])
    st.subheader("Adverse Impact Ratio")
    st.markdown("The 'Four-Fifths Rule' states the selection rate for a minority group should be at least 80% of the rate for the majority group.")
    if 'Shortlisted' in gender_df.columns and len(gender_df) > 1:
        majority_group = df['sim_gender'].value_counts().idxmax()
        minority_group = df['sim_gender'].value_counts().idxmin()
        if majority_group != minority_group:
            rate_majority = gender_df.loc[majority_group, 'Shortlisted']
            rate_minority = gender_df.loc[minority_group, 'Shortlisted']
            if rate_majority > 0:
                impact_ratio = (rate_minority / rate_majority) * 100
                st.metric(label=f"Adverse Impact Ratio ({minority_group} vs {majority_group})", value=f"{impact_ratio:.2f}%")
                if impact_ratio < 80: st.error("Adverse impact detected! Manual review recommended.", icon="🚨")
                else: st.success("No significant adverse impact detected.", icon="✅")
            else: st.info("Cannot calculate Adverse Impact Ratio as majority group has 0% selection rate.")

def placement_team_view():
    st.title("💼 Placement Team Dashboard")
    
    placement_password = st.secrets.get("PLACEMENT_PASSWORD")
    if 'password_correct' not in st.session_state: st.session_state.password_correct = False
    def check_password():
        if placement_password and st.session_state["password"] == placement_password:
            st.session_state.password_correct = True; del st.session_state["password"]
        else: st.session_state.password_correct = False
    if not st.session_state.password_correct:
        st.text_input("Password", type="password", on_change=check_password, key="password")
        if "password" in st.session_state and not st.session_state.password_correct: st.error("Wrong password.")
        return

    jobs_df = get_all_jobs()
    job_titles = {row['id']: row['title'] for _, row in jobs_df.iterrows()}
    
    with st.expander("Post a New Job"):
        with st.form(key="post_job_form"):
            job_title = st.text_input("Job Title")
            job_description = st.text_area("Job Description", height=200)
            due_date = st.date_input("Application Due Date", min_value=date.today())
            if st.form_submit_button("Post Job"):
                if job_title and job_description and due_date:
                    add_job(job_title, job_description, due_date.strftime('%Y-%m-%d'))
                    st.success(f"Job '{job_title}' posted successfully!"); st.rerun()
                else: st.warning("Please fill in all fields.")
        
    with st.expander("Analyze External Resumes"):
        if not job_titles:
            st.info("Please post a job first to enable resume analysis.")
        else:
            analysis_job_id = st.selectbox("Select job to screen against:", options=list(job_titles.keys()), format_func=lambda x: job_titles.get(x, 'N/A'))
            uploaded_files = st.file_uploader("Upload one or more resumes", accept_multiple_files=True, key="multi_uploader")
            
            if st.button("Analyze Uploaded Resumes"):
                if analysis_job_id and uploaded_files:
                    job_details = {'title': job_titles[analysis_job_id], 'description': jobs_df.loc[jobs_df['id'] == analysis_job_id, 'description'].iloc[0]}
                    with st.spinner(f"Analyzing {len(uploaded_files)} resumes..."):
                        results = [analyze_resume(job_details, f) for f in uploaded_files]
                        for res in results:
                            add_analysis_record(job_details['title'], res)
                        st.session_state.analysis_results = results
                else: st.warning("Please select a job and upload at least one resume.")

    if 'analysis_results' in st.session_state:
        st.subheader("Analysis Results")
        results_df = pd.DataFrame(st.session_state.analysis_results).sort_values(by="score", ascending=False)
        st.dataframe(results_df, use_container_width=True, hide_index=True, column_config={
            "filename": "Filename", "score": st.column_config.ProgressColumn("Score", format="%.2f%%", min_value=0, max_value=100),
            "verdict": "AI Verdict", "lacking_skills": "Lacking Skills"
        })
        del st.session_state.analysis_results

    st.write("---")
    st.header("Manage Portal Data")
    tab1, tab2, tab3 = st.tabs(["Student Applications", "Bias & Fairness Audit", "Past Analysis Results"])
    
    with tab1:
        st.subheader("Applications Submitted via the Student Portal")
        if not job_titles:
            st.info("No jobs have been posted yet.")
        else:
            selected_job_id = st.selectbox("Select a job to view applications:", options=list(job_titles.keys()), format_func=lambda x: job_titles.get(x, 'N/A'))
            if selected_job_id:
                apps_df = get_applications_for_job(selected_job_id)
                if not apps_df.empty:
                    cols = st.columns([2, 3, 1, 2, 3, 2]); cols[0].markdown("**Name**"); cols[1].markdown("**Email**"); cols[2].markdown("**Score**"); cols[3].markdown("**Verdict**"); cols[4].markdown("**Lacking Skills**"); cols[5].markdown("**Status**")
                    for _, row in apps_df.iterrows():
                        cols = st.columns([2, 3, 1, 2, 3, 2]); cols[0].text(row['candidate_name']); cols[1].text(row['candidate_email']); cols[2].text(f"{row['final_score']:.1f}%"); cols[3].text(row['verdict']); cols[4].text(row['lacking_skills'])
                        status_options = ["Applied", "Shortlisted", "Not Shortlisted"]
                        current_status_index = status_options.index(row['status']) if row['status'] in status_options else 0
                        new_status = cols[5].selectbox("Set Status", status_options, index=current_status_index, key=f"status_{row['id']}", label_visibility="collapsed")
                        if new_status != row['status']:
                            update_candidate_status(row['id'], new_status); st.toast(f"Updated {row['candidate_name']}'s status."); st.rerun()
                else: st.info("No applications for this job yet.")
    
    with tab2:
        st.subheader("Bias & Fairness Audit for Student Applications")
        if not job_titles:
            st.info("No jobs have been posted yet.")
        else:
            bias_job_id = st.selectbox("Select a job to audit:", options=list(job_titles.keys()), format_func=lambda x: job_titles.get(x, 'N/A'), key="bias_job_select")
            if bias_job_id:
                bias_apps_df = get_applications_for_job(bias_job_id)
                bias_audit_dashboard(bias_apps_df)

    with tab3:
        st.subheader("History of Analyzed External Resumes")
        analyses_df = get_all_analyses()
        if not analyses_df.empty:
            analyses_df['timestamp'] = pd.to_datetime(analyses_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            grouped = analyses_df.groupby('job_title')
            for job_title, group_df in grouped:
                with st.expander(f"Resumes Analyzed for: **{job_title}** ({len(group_df)} files)"):
                    st.dataframe(group_df, use_container_width=True, hide_index=True, column_order=("filename", "score", "verdict", "lacking_skills", "timestamp"),
                        column_config={"filename": "Filename", "score": st.column_config.ProgressColumn("Score", format="%.1f%%", min_value=0, max_value=100),
                                       "verdict": "AI Verdict", "lacking_skills": "Lacking Skills", "timestamp": "Analyzed On"})
        else: st.info("No external resumes have been analyzed yet.")

# --- Main App Execution ---
st.set_page_config(layout="wide", page_title="AI Resume Ranker")
nlp, semantic_model, llm = load_resources()

if llm:
    st.sidebar.title("👨‍💻 User Role")
    user_role = st.sidebar.radio("Select role:", ["Student", "Placement Team"])
    if user_role == "Student":
        student_view()
    else:
        placement_team_view()

