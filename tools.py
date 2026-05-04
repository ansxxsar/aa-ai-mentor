import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_community.tools import ArxivQueryRun
from langchain_core.tools import Tool
from datetime import datetime

load_dotenv()


MY_TAVILY_KEY = "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" 


tavily_search = TavilySearch(
    tavily_api_key=MY_TAVILY_KEY, 
    k=3
)

search_tool = Tool(
    name="search",
    func=tavily_search.run,
    description="Поиск технической документации и примеров кода в интернете."
)


arxiv = ArxivQueryRun()
arxiv_tool = Tool(
    name="arxiv",
    func=arxiv.run,
    description="Поиск научных статей и теоретических основ программирования."
)


def get_time(query=None):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

time_tool = Tool(
    name="time",
    func=get_time,
    description="Возвращает текущую дату и время."
)


tools = [search_tool, arxiv_tool, time_tool]