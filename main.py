from dotenv import load_dotenv
load_dotenv()

import os
import pickle
from datetime import datetime, timedelta, timezone
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_community.tools import ArxivQueryRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import tool
from rag import search_course_materials

# Google Calendar imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ---- Pydantic Model ----
class MentorResponse(BaseModel):
    analysis: str = Field(description="Comprehensive analysis of the user's request")
    documentation_snippet: str = Field(description="Technical code or architectural documentation")
    suggested_improvements: List[str] = Field(description="List of specific ways to improve the project")
    academic_references: List[str] = Field(description="Relevant papers or citations found via ArXiv")

parser = PydanticOutputParser(pydantic_object=MentorResponse)

# ---- Google Calendar Auth ----
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('calendar', 'v3', credentials=creds)

# ---- Calendar Tool ----
@tool
def create_calendar_event(summary: str, start_time: str, end_time: str = None):
    """
    Creates a REAL meeting in Google Calendar with Google Meet link.
    ALWAYS format times with +05:00. Example: 9AM = 2026-05-05T09:00:00+05:00
    Never use Z or UTC. Always use Almaty local time with +05:00.
    end_time is optional, defaults to 1 hour after start.
    """
    try:
        # Check if credentials exist (not available on Streamlit Cloud)
        if not os.path.exists('credentials.json') and not os.path.exists('token.pickle'):
            return f"✅ Meeting '{summary}' noted for {start_time} Almaty time! (Google Calendar integration requires local setup with credentials.json)"

        print(f"DEBUG received: start={start_time}, end={end_time}")

        def fix_time(t):
            t = t.strip()
            if t.endswith("Z"):
                t = t[:-1]
                dt = datetime.fromisoformat(t)
                dt = dt + timedelta(hours=5)
                return dt.isoformat() + "+05:00"
            if "+00:00" in t:
                dt = datetime.fromisoformat(t)
                dt = dt + timedelta(hours=5)
                return dt.replace(tzinfo=None).isoformat() + "+05:00"
            if "+" not in t and "-" not in t[10:]:
                return t + "+05:00"
            return t

        start_time = fix_time(start_time)
        start_dt = datetime.fromisoformat(start_time)

        if end_time:
            end_time = fix_time(end_time)
            end_dt = datetime.fromisoformat(end_time)
        else:
            end_dt = start_dt + timedelta(hours=1)

        print(f"DEBUG fixed: start={start_dt}, end={end_dt}")

        service = get_calendar_service()

        event = {
            'summary': summary,
            'start': {
                'dateTime': start_dt.isoformat(),
                'timeZone': 'Asia/Oral',
            },
            'end': {
                'dateTime': end_dt.isoformat(),
                'timeZone': 'Asia/Oral',
            },
            'conferenceData': {
                'createRequest': {
                    'requestId': f"meet-{summary}-{start_dt.isoformat()}",
                    'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                }
            }
        }

        event_result = service.events().insert(
            calendarId='primary',
            body=event,
            conferenceDataVersion=1
        ).execute()

        meet_link = event_result.get('conferenceData', {}).get('entryPoints', [{}])[0].get('uri', 'No Meet link generated')
        calendar_link = event_result.get('htmlLink')

        return f"✅ Meeting '{summary}' created from {start_dt.strftime('%H:%M')} to {end_dt.strftime('%H:%M')} Almaty time!\n📅 Calendar: {calendar_link}\n📹 Google Meet: {meet_link}"

    except Exception as e:
        return f"❌ Failed to create meeting: {str(e)}"

# ---- Tools & LLM ----
tavily_tool = TavilyAnswer()
arxiv_tool = ArxivQueryRun()
tools = [tavily_tool, arxiv_tool, create_calendar_event, search_course_materials]

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    max_tokens=4000
)

# ---- Prompt ----
today = datetime.now().strftime("%Y-%m-%d")
format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are 'Ansar's AI Mentor', an expert in Informatics and Digital Pedagogy.
    Today's date is {today}.

    TIMEZONE RULE: You are in Almaty, Kazakhstan (UTC+5, Asia/Oral).
    ALWAYS format times with +05:00 suffix. NEVER use Z or UTC.

    CORRECT example: 9AM Almaty → {today}T09:00:00+05:00
    WRONG example:   9AM Almaty → {today}T04:00:00Z  ← NEVER DO THIS

    CALENDAR RULES:
    - Schedule meetings IMMEDIATELY when asked, no clarifying questions.
    - Pass both start_time and end_time if user specifies a range (e.g. 9AM to 10AM).
    - Respond with just a short confirmation for calendar requests.

    RAG RULES:
    - If the student asks about course content, lectures, or curriculum, ALWAYS use the search_course_materials tool first.
    - Combine course material results with ArXiv papers for comprehensive answers.

    RESPONSE RULES:
    - For academic/technical questions provide DETAILED and COMPREHENSIVE answers.
    - Always write at least 3-5 sentences for each section.
    - Only use the structured format below for academic or technical questions.

    For academic/technical questions follow this structure:
    {format_instructions}
    """),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{query}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ---- Agent ----
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
