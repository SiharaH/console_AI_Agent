from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

class ResponseModel(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatGroq( model="llama-3.1-8b-instant")

parser = PydanticOutputParser(pydantic_object=ResponseModel)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agentC = create_tool_calling_agent(
    llm=llm, 
    prompt=prompt, 
    tools = []
    )

agent_Executor = AgentExecutor(agent = agentC, tools = [], verbose = True)

raw_response = agent_Executor.invoke({"query": "What is the current state of research on AI agents?"})
print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response: ", e, "\nRaw response: ", raw_response)