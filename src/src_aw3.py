import sqlite3
import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Tuple, Optional, List, Dict, Any
import os
import re
import logging

def load_database():
    app_root = os.path.dirname(os.path.dirname(__file__))  # Go up one level from src folder
    db_path = os.path.join(app_root, "databases", "fs_challenge.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return cursor

def get_database_schema():
    # Get the path relative to your app's root directory
    app_root = os.path.dirname(os.path.dirname(__file__))  # Go up one level from src folder
    db_path = os.path.join(app_root, "databases", "fs_challenge.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema = ''
    for table_name in tables:
        table_name = table_name[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        schema += f"Table: {table_name}\n"
        schema += "Columns:\n"
        for column in columns:
            schema += f" - {column[1]} ({column[2]})\n"
        schema += '\n'
    conn.close()
    return schema

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class AgentState(TypedDict):
    question: str
    table_schemas: str
    metadata_description : str
    database: str
    sql: str
    reflect: List[str]
    accepted: bool
    revision: int
    max_revision: int

def search_engineer_node(state: AgentState):
    # Fornecemos o esquema do banco de dados diretamente
    state['table_schemas'] = get_database_schema()
    state['database'] = '/databases/fs_challenge.db'
    state['metadata_description'] = """ ****metadata table variables****
                                        *** the key variable for join is *ceg_label*
                                        Reference Date: din_instante
                                        Plant Subsystem: id_subsistema
                                        Subsystem Name: nom_subsistema
                                        State where the Plant is located: id_estado
                                        State Name: nom_estado
                                        Plant Operation Mode: cod_modalidadeoperacao
                                        Plant Type: nom_tipousina
                                        Fuel Type: nom_tipocombustivel
                                        Plant Name: nom_usina
                                        ONS Identifier: id_ons
                                        Unique Generation Enterprise Code: ceg_label
                                        Power Generation in MWmed: val_geracao
                                        """
    return {"table_schemas": state['table_schemas'], "database": state['database'],"metadata_description": state['metadata_description']}

def senior_sql_writer_node(state: AgentState):
    role_prompt = """
You are an SQL expert. Your task is to write **only** the SQL query that answers the user's question. 

The query must:
- SQLite3 database
- Use standard SQL syntax in English.
- Always give names to the columns of the result.
- Use the table and column names as defined in the database schema.
- Not include comments, explanations, or any additional text.
- Not use code formatting or markdown.
- Return only the valid SQL query.
"""
    instruction = f"It is a database on powerplants production.There are **two tables** in the data base, one with the production and another with metadata. Pay atention to the names of the columns in each database.Here is the database schema:\n{state['table_schemas']}\n"
    instruction += f"Metadata columns meaning: \n {state['metadata_description']}"
    if len(state['reflect']) > 0:
        instruction += f"Consider the following feedback:\n{chr(10).join(state['reflect'])}\n"
    instruction += f"Write the SQL query that answers the following question: {state['question']}\n"
    messages = [
        SystemMessage(content=role_prompt), 
        HumanMessage(content=instruction)
    ]
    response = llm.invoke(messages)
    return {"sql": response.content.strip(), "revision": state['revision'] + 1}

#Função do QA Engineer
def senior_qa_engineer_node(state: AgentState):
    role_prompt = """
You are a QA engineer specialized in SQL. Your task is to verify if the provided SQL query correctly answers the user's question.
"""
    instruction = f"It is a database on powerplants production.There are two tables in the data base, one with the production and another with metadata. Here is the database schema:\n{state['table_schemas']}\n"
    instruction += f"Metadata columns meaning: \n {state['metadata_description']}"
    instruction += f"And the following SQL query:\n{state['sql']}\n"
    instruction += f"Verify if the SQL query can complete the task: {state['question']}\n"
    instruction += "Respond 'ACCEPTED' if it is correct or 'REJECTED' if it is not.\n"
    messages = [
        SystemMessage(content=role_prompt), 
        HumanMessage(content=instruction)
    ]
    response = llm(messages)
    return {"accepted": 'ACCEPTED' in response.content.upper()}

def chief_dba_node(state: AgentState):
    role_prompt = """
You are an experienced DBA. Your task is to provide detailed feedback to improve the provided SQL query.
"""
    instruction = f"It is a database on powerplants production.There are two tables in the data base, one with the production and another with metadata. Here is the database schema:\n{state['table_schemas']}\n"
    instruction += f"Metadata columns meaning: \n {state['metadata_description']}"
    instruction += f"And the following SQL query:\n{state['sql']}\n"
    instruction += f"Provide useful and detailed recommendations to help improve the SQL query for the task: {state['question']}\n"
    messages = [
        SystemMessage(content=role_prompt), 
        HumanMessage(content=instruction)
    ]
    response = llm(messages)
    return {"reflect": [response.content]}

def create_workflow():
    builder = StateGraph(AgentState)

    builder.add_node("search_engineer", search_engineer_node)
    builder.add_node("sql_writer", senior_sql_writer_node)
    builder.add_node("qa_engineer", senior_qa_engineer_node)
    builder.add_node("chief_dba", chief_dba_node)

    builder.add_edge("search_engineer", "sql_writer")
    builder.add_edge("sql_writer", "qa_engineer")
    builder.add_edge("chief_dba", "sql_writer")

    builder.add_conditional_edges(
        "qa_engineer", 
        lambda state: END if state['accepted'] or state['revision'] >= state['max_revision'] else "reflect", 
        {END: END, "reflect": "chief_dba"}
    )

    memory = MemorySaver()

    builder.set_entry_point("search_engineer")

    app = builder.compile(checkpointer=memory)
    return app


def process_question(question: str, graph: Any, thread_id: str = "1") -> Dict[str, Any]:
    """
    Processes a given question through a graph-based workflow to generate a final state.

    This function initializes the state with the provided question, streams it through the graph,
    and returns the final state after processing.

    Args:
        question (str): The question to be processed.
        graph (Any): The graph object that processes the state.
        thread_id (str, optional): The ID of the thread for processing. Defaults to "1".

    Returns:
        Dict[str, Any]: The final state after processing the question through the graph.
    """
    # Initialize the state with the provided question
    initial_state = {
        'question': question,
        'table_schemas': '',  # Will be populated by 'search_engineer_node'
        'database': '',       # Will be populated by 'search_engineer_node'
        'sql': '',
        'reflect': [],
        'accepted': False,
        'revision': 0,
        'max_revision': 2
    }

    # Configure the thread
    thread = {"configurable": {"thread_id": thread_id}}

    # Stream the initial state through the graph
    for _ in graph.stream(initial_state, thread):
        pass  # Processing is handled internally by the graph

    # Retrieve and return the final state
    final_state = graph.get_state(thread)
    return final_state

def extract_sql(query_text):
    # First try to find SQL between markdown code blocks
    match = re.search(r'```sql\s*(.*?)\s*```', query_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # If not found, try to find a complete SQL query starting with WITH or SELECT
    match = re.search(r'((?:WITH|SELECT).*?);', query_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + ';'
    
    # If still not found, return cleaned original text
    return query_text.strip()


def execute_sql_query(sql_query: str) -> Optional[Tuple[List[Tuple], List[str]]]:
    """
    Executes an SQL query using the provided database cursor and returns the results.

    Args:
        sql_query (str): The SQL query to be executed.

    Returns:
        Optional[Tuple[List[Tuple], List[str]]]: A tuple containing the query results and column names if successful, 
        or None if an error occurs.
    """
    try:
        cursor = load_database()
        if not cursor:
            logging.error("Failed to create a database cursor.")
            return None

        # Execute the SQL query
        cursor.execute(sql_query)
        
        # Fetch all results
        results = cursor.fetchall()
        # Get column names from cursor.description
        column_names = [desc[0] for desc in cursor.description]

        # Log the results for debugging purposes
        logging.info("SQL query executed successfully.")
        logging.debug(f"Query results: {results}")
        logging.debug(f"Column names: {column_names}")
        
        return results, column_names
    
    except Exception as e:
        # Log the error and print a user-friendly message
        logging.error(f"Error executing SQL query: {e}")
        print("Error executing SQL query. Please check the logs for more details.")
        return None

def create_dataframe(column_names, query_results):
    """
    Creates a pandas DataFrame from query results and column names.

    Args:
        column_names (List[str]): A list of column names.
        query_results (List[Tuple]): A list of tuples containing the query results.

    Returns:
        pd.DataFrame: A DataFrame containing the query results with the specified column names.
    """
    # Create a DataFrame from the query results and column names
    df = pd.DataFrame(query_results, columns=column_names)
    return df

def validate_and_execute_sql(query_text):
    sql_query = extract_sql(query_text)
    print("Extracted query:", repr(sql_query))  # For debugging
    
    # Basic validation
    if not sql_query.strip():
        raise ValueError("Empty SQL query")
    
    # Make sure the query ends with a semicolon
    if not sql_query.rstrip().endswith(';'):
        sql_query += ';'
    cursor = load_database()
    return cursor.execute(sql_query)