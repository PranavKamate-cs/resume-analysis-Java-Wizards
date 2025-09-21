🤖 AI-Powered Resume Ranker & Placement Portal
An intelligent, dual-view web application designed to streamline the campus placement process. This tool uses AI to bridge the gap between students applying for jobs and placement teams screening resumes, ensuring a fair, efficient, and data-driven workflow.

✨ Key Features
For Students 🎓
Job Board: View and search for job openings posted by the placement team.

Easy Application: Apply for jobs by simply uploading a resume and filling in basic details.

Instant AI Feedback: Receive an immediate, detailed report on your resume's alignment with the job description, including strengths and areas for improvement.

Application Status Tracking: Check the status of all your submitted applications in one place.

For Placement Teams 💼
Job Management: Easily post, update, and manage job descriptions with application deadlines.

Automated Screening: Every student application is automatically scored and ranked using a hybrid AI model.

Simplified Dashboard: View all applicants for a job in a clean, sorted list showing their name, email, score, AI verdict, and identified skill gaps.

Interactive Status Updates: Update an applicant's status ("Shortlisted", "Not Shortlisted") with a single click.

Bulk Analysis Tool: Analyze external resumes (sourced outside the portal) by uploading multiple files at once.

Analysis History: Access a persistent log of all external resume analyses for future reference.

Bias & Fairness Audit: A dedicated dashboard to monitor key fairness metrics like Success Rate Parity and the Adverse Impact Ratio, helping to ensure an equitable hiring process.

AI Core 🧠
Hybrid Scoring Model: Combines keyword matching, semantic similarity (via Sentence Transformers), and contextual analysis (via Google's Gemini Pro LLM) for a nuanced and accurate score.

Bias Mitigation: Implements blind screening (PII redaction) and a fairness-aware AI prompt to reduce algorithmic and historical bias.

Intelligent Skill Extraction: The AI identifies and lists key skills that may be lacking in a resume compared to the job description.

🛠️ Tech Stack
Application Framework: Streamlit

Core AI/ML Libraries: LangChain, Google Generative AI (Gemini Pro), spaCy, Sentence Transformers

Data Handling: Pandas, NumPy

Document Parsing: PyPDF2, python-docx

Database: SQLite (for both student applications and analysis history)

📂 Project Structure
The project is a monolithic Streamlit application, designed for easy deployment.

/
├── .streamlit/
│   └── secrets.toml
├── app.py
├── requirements.txt
├── packages.txt
├── placement_portal.db
├── analysis_results.db
└── README.md
🚀 Getting Started
Follow these instructions to set up and run the project locally.

Prerequisites
Python 3.9 or higher

Git

1. Clone the Repository
Bash

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
2. Create and Activate a Virtual Environment
Windows:

Bash

python -m venv venv
venv\Scripts\activate
macOS / Linux:

Bash

python -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
python -m spacy download en_core_web_sm
4. Set Up Your Secrets
Create a file at .streamlit/secrets.toml and add your credentials:

Ini, TOML

# .streamlit/secrets.toml
GOOGLE_API_KEY = "your_google_api_key_here"
PLACEMENT_PASSWORD = "your_admin_password_here"

# Optional: LangSmith for tracing
# LANGCHAIN_TRACING_V2 = "true"
# LANGCHAIN_API_KEY = "your_langsmith_key_here"
# LANGCHAIN_PROJECT = "Resume Ranker"
5. Run the Application
Bash

streamlit run app.py
The application should now be running and accessible in your web browser!

☁️ Deployment
This application is designed for deployment on Streamlit Community Cloud.

Push your project to a public GitHub repository.

Ensure your requirements.txt and packages.txt files are up to date.

On the Streamlit Cloud dashboard, create a new app and link it to your repository.

In the "Advanced settings," add the contents of your secrets.toml file.

Deploy!
