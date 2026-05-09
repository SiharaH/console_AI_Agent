from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from datetime import datetime
from langchain.tools import StructuredTool
from pydantic import BaseModel


class SearchInput(BaseModel):
    query: str

def search(query: str):
    return f"Searching for: {query}"

search_tool = StructuredTool.from_function(
    func=search,
    name="search",
    description="Search for information",
    args_schema=SearchInput
)