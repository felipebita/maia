from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import re

llm = ChatOpenAI(model="gpt-4o-mini")

class State(TypedDict):
    """ Representa o estado do processo de avaliação da redação """
    theme: str
    essay: str
    relevance_score: float
    grammar_score: float
    structure_score: float
    depth_score: float
    final_score: float

def extract_score(content: str) -> float:
    """Extracts the numerical score from the LLM response."""
    match = re.search(r'Score:\s*(\d+(\.\d+)?)', content)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract score from: {content}")

def check_relevance(state: State) -> State:
   """Checks the essay's relevance."""
   prompt = ChatPromptTemplate.from_template(
       "Analyze the relevance of the following essay in relation to the given theme: {theme}. Focusing on excellence in English. "
       "Provide a relevance score between 0 and 1. "
       "Your response should start with 'Score: ' followed by the numerical score, "
       "then provide your explanation.\n\nEssay: {essay}"
   )
   result = llm.invoke(prompt.format(theme=state["theme"],essay=state["essay"]))
   try:
       state["relevance_score"] = extract_score(result.content)
   except ValueError as e:
       print(f"Error in check_relevance: {e}")
       state["relevance_score"] = 0.0
   return state

def check_grammar(state: State) -> State:
   """Checks the essay's grammar."""
   prompt = ChatPromptTemplate.from_template(
       "Analyze the grammar in the following essay. "
       "Provide a grammar score between 0 and 1. "
       "Your response should start with 'Score: ' followed by the numerical score, "
       "then provide your explanation.\n\nEssay: {essay}"
   )
   result = llm.invoke(prompt.format(essay=state["essay"]))
   try:
       state["grammar_score"] = extract_score(result.content)
   except ValueError as e:
       print(f"Error in check_grammar: {e}")
       state["grammar_score"] = 0.0
   return state

def analyze_structure(state: State) -> State:
   """Analyzes the essay's structure."""
   prompt = ChatPromptTemplate.from_template(
       "Analyze the structure of the following essay according to formal English standards. "
       "Provide a structure score between 0 and 1. "
       "Your response should start with 'Score: ' followed by the numerical score, "
       "then provide your explanation.\n\nEssay: {essay}"
   )
   result = llm.invoke(prompt.format(essay=state["essay"]))
   try:
       state["structure_score"] = extract_score(result.content)
   except ValueError as e:
       print(f"Error in analyze_structure: {e}")
       state["structure_score"] = 0.0
   return state

def evaluate_depth(state: State) -> State:
   """Evaluates the depth of analysis in the essay."""
   prompt = ChatPromptTemplate.from_template(
       "Evaluate the depth of analysis in the following essay. "
       "Provide a depth score between 0 and 1. "
       "Your response should start with 'Score: ' followed by the numerical score, "
       "then provide your explanation.\n\nEssay: {essay}"
   )
   result = llm.invoke(prompt.format(essay=state["essay"]))
   try:
       state["depth_score"] = extract_score(result.content)
   except ValueError as e:
       print(f"Error in evaluate_depth: {e}")
       state["depth_score"] = 0.0
   return state

def calculate_final_score(state: State) -> State:
   """Calculates the final score based on individual component scores."""
   state["final_score"] = (
       state["relevance_score"] * 0.3 +
       state["grammar_score"] * 0.2 +
       state["structure_score"] * 0.2 +
       state["depth_score"] * 0.3
   )
   return state

def create_essay_workflow() -> StateGraph:
   """Creates and returns a configured essay grading workflow."""
   
   # Initialize StateGraph
   workflow = StateGraph(State)
   
   # Add nodes to graph
   workflow.add_node("check_relevance", check_relevance)
   workflow.add_node("check_grammar", check_grammar)
   workflow.add_node("analyze_structure", analyze_structure)
   workflow.add_node("evaluate_depth", evaluate_depth)
   workflow.add_node("calculate_final_score", calculate_final_score)
   
   # Define and add conditional edges
   workflow.add_conditional_edges(
       "check_relevance",
       lambda x: "check_grammar" if x["relevance_score"] > 0.5 else "calculate_final_score"
   )
   workflow.add_conditional_edges(
       "check_grammar",
       lambda x: "analyze_structure" if x["grammar_score"] > 0.6 else "calculate_final_score"
   )
   workflow.add_conditional_edges(
       "analyze_structure",
       lambda x: "evaluate_depth" if x["structure_score"] > 0.7 else "calculate_final_score"
   )
   workflow.add_conditional_edges(
       "evaluate_depth",
       lambda x: "calculate_final_score"
   )
   
   # Set entry and exit points
   workflow.set_entry_point("check_relevance")
   workflow.add_edge("calculate_final_score", END)
   
   return workflow.compile()

def grade_essay(app, theme: str, essay: str) -> dict:
    """Evaluates the provided essay using the defined workflow."""
    initial_state = State(
        theme=theme,
        essay=essay,
        relevance_score=0.0,
        grammar_score=0.0,
        structure_score=0.0,
        depth_score=0.0,
        final_score=0.0
    )
    result = app.invoke(initial_state)
    # Remove the essay from the result and return only the scores
    result_text = (
        f"Relevance Score: {result['relevance_score']}\n\n"
        f"Grammar Score: {result['grammar_score']}\n\n"
        f"Structure Score: {result['structure_score']}\n\n"
        f"Depth Score: {result['depth_score']}\n\n"
        f"Final Score: {result['final_score']}"
    )
    return result_text
