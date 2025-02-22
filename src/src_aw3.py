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

def initialize_model(model):
    llm = ChatOpenAI(model=model, temperature=0)
    return llm

class AgentState(TypedDict):
    model: Any
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
        You are an expert SQL developer specialized in power plant data analysis. Your task is to generate precise SQLite3 queries that analyze power generation data. You must follow these guidelines:

        1. **Database Context**:
        - Rementer, it is a SQLite3 database
        - Work with two interconnected tables: Production and Metadata
        - Key join field is 'ceg_label'
        - Power generation is measured in 'val_geracao' (MWmed)
        - In SQLite, GROUP is a reserved keyword. To use it as a column name, enclose it in double quotes (e.g., "group").

        2. **Query Structure**:
        - Start complex queries with CTEs (WITH clause) for better organization
        - Use consistent table aliases: 'prod' for production, 'meta' for metadata
        - Include proper table joins with explicit JOIN conditions
        - Always specify column sources using table aliases (e.g., meta.nom_usina)

        3. **Data Quality**:
        - Handle NULL values using COALESCE(column, default_value)
        - Prevent division by zero: NULLIF(denominator, 0)
        - Use CASE statements for data categorization with proper END clause
        - Round numerical results using ROUND(column, decimal_places)

        4. **Performance & Readability**:
        - Filter data early using WHERE clauses before aggregations
        - Include only necessary columns in SELECT statements
        - Use meaningful column aliases for calculated fields
        - Format dates using strftime() when needed
        - Break complex calculations into CTEs

        5. **Output Requirements**:
        - This code is going to be automatically run, do not use ajustable filters or anything like that
        - Provide only the SQL query, not any other letter, word or symbol
        - Include no explanatory text
        - Ensure all table/column references match schema exactly
        - Test query logic before output

        6. **Critical Checks**:
        - Verify all column names exist in schema
        - Confirm temporal logic is correct
        - Validate aggregation groupings
        - Ensure joins won't cause data multiplication
        """
    instruction = rf"""
        You are analyzing a power plant production database with two tables:
        1. Production table (ons_ts): Contains generation data (val_geracao in MWmed) 
        2. Metadata table (ons_metadata): Contains plant information

        When joining the tables, always use ceg_label variable. It is a unique identifier of the PowerPlants and its name.

        Schema details:
        {state['table_schemas']}

        Column definitions:
        {state['metadata_description']}

        {("Previous feedback to incorporate:" + chr(10).join(state['reflect'])) if len(state['reflect']) > 0 else ""}

        Task: Write a SQL query that answers this question: {state['question']}
        """
    messages = [
        SystemMessage(content=role_prompt),
        HumanMessage(content=instruction)
    ]
    llm = state['model']
    response = llm.invoke(messages)
    return {"sql": response.content.strip(), "revision": state['revision'] + 1}

#Função do QA Engineer
def senior_qa_engineer_node(state: AgentState):
    role_prompt = """
        You are a Senior QA Engineer specializing in power plant data analysis and SQL validation. Your critical role is to verify SQL queries against strict quality criteria. Follow this validation checklist:

        1. **Query Structure Validation**:
        - Verify correct JOIN conditions using 'ceg_label'
        - Check for proper table aliases usage
        - Validate column references match schema
        - Confirm appropriate use of aggregation functions
        - Verify GROUP BY includes all non-aggregated columns
        - In SQLite, GROUP is a reserved keyword. To use it as a column name, enclose it in double quotes (e.g., "group").

        2. **Data Quality Checks**:
        - Verify NULL handling with COALESCE or similar functions
        - Check for division by zero protection
        - Validate date/time handling
        - Confirm proper rounding of numerical results

        3. **Business Logic Validation**:
        - Ensure query answers ALL aspects of the question
        - Verify correct filtering conditions
        - Validate temporal logic when time periods are involved
        - Check unit consistency (MWmed for power generation)

        4. **Common Error Detection**:
        - Look for data multiplication from incorrect joins
        - Check for missing WHERE clauses
        - Verify aggregation logic
        - Validate calculation accuracy

        Your response must be exactly 'ACCEPTED' or 'REJECTED'. Choose 'ACCEPTED' only if ALL of the following are true:
        - Query is syntactically correct
        - All business requirements are met
        - Data quality checks pass
        - No logical errors present

        Choose 'REJECTED' if ANY validation fails.
        """
    instruction = rf"""
        Task Context:
        - Database: Power plant production data
        - Tables: Production (generation data) and Metadata (plant information)
        - Key Fields: 
        * ceg_label (join key)
        * val_geracao (power generation in MWmed)
        * din_instante (reference date)

        Schema Definition:
        {state['table_schemas']}

        Column Descriptions:
        {state['metadata_description']}

        Query to Validate:
        {state['sql']}

        Question to Answer:
        {state['question']}

        Validate if this query EXACTLY matches the requirements. Respond ONLY with 'ACCEPTED' or 'REJECTED'.
        """
    messages = [
        SystemMessage(content=role_prompt),
        HumanMessage(content=instruction)
    ]
    llm = state['model']
    response = llm.invoke(messages)
    return {"accepted": 'ACCEPTED' in response.content.upper()}

def chief_dba_node(state: AgentState):
    role_prompt = """
        You are the Chief Database Administrator specializing in power plant data systems. Your crucial role is to analyze rejected SQL queries and provide clear, actionable feedback for improvements. Follow these guidelines:

        1. **Analysis Framework**:
        - First identify the primary issues preventing query acceptance
        - Evaluate both technical correctness and business logic
        - Consider query performance and maintainability
        - Focus on power generation domain-specific requirements

        2. **Feedback Structure**:
        Organize your feedback into these categories:
        a) CRITICAL ISSUES:
            - Logic errors that affect result accuracy
            - Incorrect joins or relationships
            - Missing business requirements
            - Data quality risks
        
        b) OPTIMIZATION SUGGESTIONS:
            - Performance improvements
            - Better readability
            - More maintainable structure
            - Error prevention

        3. **Solution Guidelines**:
        - Provide specific, actionable recommendations
        - Reference exact parts of the query that need changes
        - Explain WHY each change is needed
        - Consider the power plant domain context

        4. **Feedback Format**:
        - Use concise, clear language
        - Focus on most important issues first
        - Provide examples where helpful
        - Keep feedback constructive and specific

        Remember: Your feedback will be used directly by the SQL Writer to improve the query. Be precise and actionable in your recommendations.
        """
    instruction = rf"""
        Context:
        - Database Purpose: Power plant production analysis
        - Critical Fields:
        * ceg_label: Key join field
        * val_geracao: Power generation (MWmed)
        * din_instante: Reference date

        Database Schema:
        {state['table_schemas']}

        Column Definitions:
        {state['metadata_description']}

        Current Query:
        {state['sql']}

        Task Requirements:
        {state['question']}

        Analyze this query and provide structured feedback following the framework in your role description. Focus on making the query both technically correct and optimized for power plant data analysis.
        """
    messages = [
        SystemMessage(content=role_prompt),
        HumanMessage(content=instruction)
    ]
    llm = state['model']
    response = llm.invoke(messages)
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


def process_question(question: str, graph: Any, llm:Any, thread_id: str = "1") -> Dict[str, Any]:
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
        'model':llm,
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


def review_query_results(user_request, sql_query, user_review):
    """
    Function to analyze the original request, SQL query, and user review using GPT
    
    Args:
        user_request (str): Original user's natural language request
        sql_query (str): SQL query generated by the first workflow
        user_review (str): User's review of the results
        openai_api_key (str): OpenAI API key
    
    Returns:
        dict: Contains the revised SQL query and explanation
    """
    
    # Create the system prompt
    system_prompt = """You are an expert SQL developer. Your task is to analyze:
    1. The original user request for data
    2. The SQL query that was generated
    3. The user's review explaining why the results weren't what they expected
    
    Based on this analysis, you should:
    1. Understand what went wrong with the original query
    2. Create a new SQL query that better matches the user's intentions
    3. Explain why the new query better addresses the user's needs
    
    Return your response in this format:
    ANALYSIS: Your analysis of what went wrong
    NEW_QUERY: Your revised SQL query
    EXPLANATION: Why the new query better matches the user's needs"""
    
    # Create the human message template
    human_template = """
    ORIGINAL REQUEST: {user_request}
    
    GENERATED SQL QUERY: {sql_query}
    
    USER REVIEW: {user_review}
    
    Please analyze these and provide a better SQL query that addresses the user's needs."""
    
    # Create messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_template.format(
            user_request=user_request,
            sql_query=sql_query,
            user_review=user_review
        ))
    ]
    # Get response from GPT
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = llm.invoke(messages)
    
    # Parse the response
    response_text = response.content
    
    # Split the response into sections
    sections = {}
    current_section = None
    current_content = []
    
    for line in response_text.split('\n'):
        if line.startswith('ANALYSIS:'):
            current_section = 'analysis'
            current_content = []
        elif line.startswith('NEW_QUERY:'):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'new_query'
            current_content = []
        elif line.startswith('EXPLANATION:'):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'explanation'
            current_content = []
        elif line.strip():
            current_content.append(line.strip())
    
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections