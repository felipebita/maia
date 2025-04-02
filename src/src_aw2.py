from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

def classification_node(state: State):
    ''' Classifies the text into one of the categories: News, Blog, Research, or Other '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText:{text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}


def entity_extraction_node(state: State):
    ''' Extracts all entities (Person, Organization, Location) from the text '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}


def summarization_node(state: State):
    ''' Summarizes the text into a short sentence '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text into a short sentence.\n\nText:{text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}

def create_workflow():
    """
    Creates and compiles a workflow using a StateGraph with nodes for classification,
    entity extraction, and summarization.
    """
    workflow = StateGraph(State)
    
    # Add nodes to the graph
    workflow.add_node("classification_node", classification_node)
    workflow.add_node("entity_extraction", entity_extraction_node)
    workflow.add_node("summarization", summarization_node)
    
    # Add edges to the graph
    workflow.set_entry_point("classification_node")  # Set the entry point of the graph
    workflow.add_edge("classification_node", "entity_extraction")
    workflow.add_edge("entity_extraction", "summarization")
    workflow.add_edge("summarization", END)
    
    # Compile the graph
    app = workflow.compile()
    return app

def process_text(app, sample_text):
    """
    Processes a given text using the provided app workflow and returns the results
    as a formatted string containing the classification, extracted entities, and summary.
    
    Parameters:
        app: The compiled workflow to invoke.
        sample_text: The input text to be processed.
    
    Returns:
        A string containing the classification, entities, and summary.
    """
    state_input = {"text": sample_text}
    result = app.invoke(state_input)
    
    output = (
        f"Classification: {result['classification']}\n\n"
        f"Entities: {result['entities']}\n\n"
        f"Summary: {result['summary']}"
    )
    
    return output