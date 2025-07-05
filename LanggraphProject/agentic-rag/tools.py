from langchain_community.tools import DuckDuckGoSearchRun
from retrievr import retrieve_tool,get_weather_info
from langchain.tools import Tool,tool #to create tool in 2 ways


# return serach tool
def createSearchTool():
    search_tool = DuckDuckGoSearchRun()
    return search_tool

def createRagTool():
    print("Inside Rag Toll creation")
    rag_tool = Tool(                                    #first method to create tool
    name="rag_retriever",
    func=retrieve_tool,
    description="Retrieves information about uploaded pdf/doc/txt."
    )

    return rag_tool

def createWeatherTool():
    weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)
    return weather_info_tool

@tool                                                   #second method to create tool
def createAddTool(a:int,b:int)->int:  
   """Adds two number and return the result"""
   return a+b
@tool
def createMultiplyTool(a:int,b:int)->int: 
    """Multiply two number and return the result"""
    return a*b


    








