🤖 AI-Powered Resume Ranker & Placement Portal
A modern, AI-driven web application designed to revolutionize the campus placement process for both students and placement officers.

## 🎯 Problem Statement
Traditional campus recruitment processes are often manual, time-consuming, and prone to human bias. Placement teams are inundated with hundreds of resumes for each job opening, making it challenging to:

Screen effectively and identify the best-fit candidates in a timely manner.

Provide personalized feedback to students to help them improve.

Ensure a fair and unbiased evaluation process, giving every student an equal opportunity.

Scale the process efficiently as the number of applicants and job roles grows.

This project aims to solve these challenges by leveraging AI to create a transparent, efficient, and equitable placement ecosystem.

## 💡 Our Approach
This application tackles the problem with a multi-faceted, AI-first approach:

Dual-View System: Provides two distinct interfaces—a Student Portal for applying to jobs and tracking status, and a Placement Team Dashboard for managing jobs and candidates.

Hybrid AI Scoring Model: Instead of relying on a single metric, our core analysis engine combines three techniques for a robust and nuanced evaluation:

Keyword Matching: A baseline check for essential skills and qualifications.

Semantic Similarity: Uses Sentence Transformers to understand the contextual meaning of a resume, even if it doesn't use exact keywords.

LLM-Powered Analysis: Leverages Google's Gemini Pro via LangChain for deep contextual understanding, identifying strengths, and pinpointing skill gaps.

Automated & Fair Feedback: Every submitted resume is automatically analyzed. The system uses a fairness-aware prompt and redacts personal information (blind screening) to mitigate bias, ensuring the evaluation is based on merit.

Efficient Administrative Tools: The placement team is equipped with tools for bulk resume analysis, historical data access, and a dedicated Bias & Fairness Audit dashboard to monitor hiring outcomes.

## 🛠️ Installation Steps
Follow these steps to set up and run the project on your local machine.

### Prerequisites
Python 3.11

Git

### 1. Clone the Repository
Open your terminal and clone the project repository:

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/PranavKamate-cs/resume-analysis-Java-Wizards/tree/main)
cd resume-analysis-Java-Wizards

Of course. Here is a comprehensive README.md file for your project, structured with the sections you requested.

You can create a new file named README.md in your project's root directory and paste this content into it.

🤖 AI-Powered Resume Ranker & Placement Portal
A modern, AI-driven web application designed to revolutionize the campus placement process for both students and placement officers.

## 🎯 Problem Statement
Traditional campus recruitment processes are often manual, time-consuming, and prone to human bias. Placement teams are inundated with hundreds of resumes for each job opening, making it challenging to:

Screen effectively and identify the best-fit candidates in a timely manner.

Provide personalized feedback to students to help them improve.

Ensure a fair and unbiased evaluation process, giving every student an equal opportunity.

Scale the process efficiently as the number of applicants and job roles grows.

This project aims to solve these challenges by leveraging AI to create a transparent, efficient, and equitable placement ecosystem.

## 💡 Our Approach
This application tackles the problem with a multi-faceted, AI-first approach:

Dual-View System: Provides two distinct interfaces—a Student Portal for applying to jobs and tracking status, and a Placement Team Dashboard for managing jobs and candidates.

Hybrid AI Scoring Model: Instead of relying on a single metric, our core analysis engine combines three techniques for a robust and nuanced evaluation:

Keyword Matching: A baseline check for essential skills and qualifications.

Semantic Similarity: Uses Sentence Transformers to understand the contextual meaning of a resume, even if it doesn't use exact keywords.

LLM-Powered Analysis: Leverages Google's Gemini Pro via LangChain for deep contextual understanding, identifying strengths, and pinpointing skill gaps.

Automated & Fair Feedback: Every submitted resume is automatically analyzed. The system uses a fairness-aware prompt and redacts personal information (blind screening) to mitigate bias, ensuring the evaluation is based on merit.

Efficient Administrative Tools: The placement team is equipped with tools for bulk resume analysis, historical data access, and a dedicated Bias & Fairness Audit dashboard to monitor hiring outcomes.

## 🛠️ Installation Steps
Follow these steps to set up and run the project on your local machine.

### Prerequisites
Python 3.11

Git

### 1. Clone the Repository
Open your terminal and clone the project repository:

Bash

git clone h[ttps://github.com/your-username/your-repository-name.git](https://github.com/PranavKamate-cs/resume-analysis-Java-Wizards/tree/main)
cd resume-analysis-Java-Wizards
### 2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

On Windows:

python -m venv venv
venv\Scripts\activate

On macOS / Linux:

python -m venv venv
source venv/bin/activate

Of course. Here is a comprehensive README.md file for your project, structured with the sections you requested.

You can create a new file named README.md in your project's root directory and paste this content into it.

🤖 AI-Powered Resume Ranker & Placement Portal
A modern, AI-driven web application designed to revolutionize the campus placement process for both students and placement officers.

## 🎯 Problem Statement
Traditional campus recruitment processes are often manual, time-consuming, and prone to human bias. Placement teams are inundated with hundreds of resumes for each job opening, making it challenging to:

Screen effectively and identify the best-fit candidates in a timely manner.

Provide personalized feedback to students to help them improve.

Ensure a fair and unbiased evaluation process, giving every student an equal opportunity.

Scale the process efficiently as the number of applicants and job roles grows.

This project aims to solve these challenges by leveraging AI to create a transparent, efficient, and equitable placement ecosystem.

## 💡 Our Approach
This application tackles the problem with a multi-faceted, AI-first approach:

Dual-View System: Provides two distinct interfaces—a Student Portal for applying to jobs and tracking status, and a Placement Team Dashboard for managing jobs and candidates.

Hybrid AI Scoring Model: Instead of relying on a single metric, our core analysis engine combines three techniques for a robust and nuanced evaluation:

Keyword Matching: A baseline check for essential skills and qualifications.

Semantic Similarity: Uses Sentence Transformers to understand the contextual meaning of a resume, even if it doesn't use exact keywords.

LLM-Powered Analysis: Leverages Google's Gemini Pro via LangChain for deep contextual understanding, identifying strengths, and pinpointing skill gaps.

Automated & Fair Feedback: Every submitted resume is automatically analyzed. The system uses a fairness-aware prompt and redacts personal information (blind screening) to mitigate bias, ensuring the evaluation is based on merit.

Efficient Administrative Tools: The placement team is equipped with tools for bulk resume analysis, historical data access, and a dedicated Bias & Fairness Audit dashboard to monitor hiring outcomes.

## 🛠️ Installation Steps
Follow these steps to set up and run the project on your local machine.

### Prerequisites
Python 3.11

Git

### 1. Clone the Repository
Open your terminal and clone the project repository:

Bash

git clone https:[//github.com/your-username/your-repository-name.git](https://github.com/PranavKamate-cs/resume-analysis-Java-Wizards)
cd your-repository-name
### 2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

On Windows:

Bash

python -m venv venv
venv\Scripts\activate
On macOS / Linux:

Bash

python -m venv venv
source venv/bin/activate
### 3. Install Dependencies
Install all required Python packages and the spaCy language model.

pip install -r requirements.txt
python -m spacy download en_core_web_sm

### 4. Set Up Your Secrets
Create a secrets file for your API keys and password.

1. Create a folder named .streamlit in your project's root directory.

2. Inside that folder, create a file named secrets.toml.

3. Add your credentials to the file as shown below:
# .streamlit/secrets.toml

GOOGLE_API_KEY = "your_google_api_key_here"
PLACEMENT_PASSWORD = "your_admin_password_here"

## 🚀 Usage
### Running the Application
Once the installation is complete, run the following command in your terminal from the project's root directory:
streamlit run app.py

Your web browser will automatically open with the application running.

### User Roles
The application has two primary roles, which can be selected from the sidebar:

Student: Can view open jobs, submit applications with their resume, and check the status and AI-generated feedback on their past submissions.

Placement Team: After entering a password, they can post new jobs, view all applicants for a selected job, update application statuses, and use the advanced tools for external resume analysis and bias auditing.
