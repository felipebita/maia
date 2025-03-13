import sqlite3
import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Tuple, Optional, List, Dict, Any, Union
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

class AgentStateSQL(TypedDict):
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

def search_engineer_node(state: AgentStateSQL):
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

def senior_sql_writer_node(state: AgentStateSQL):
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
def senior_qa_engineer_node(state: AgentStateSQL):
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

def chief_dba_node(state: AgentStateSQL):
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

def create_workflow_sql():
    builder = StateGraph(AgentStateSQL)

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


def execute_sql_query(sql_query: str) -> Union[Tuple[List[Tuple], List[str]], str, None]:  # Changed return type
    """
    Executes an SQL query using the provided database cursor and returns the results.

    Args:
        sql_query (str): The SQL query to be executed.

    Returns:
        Union[Tuple[List[Tuple], List[str]], str, None]: A tuple containing the query results and column names if successful,
        the error message as a string if an error occurs, or None if the connection fails.
    """
    try:
        cursor = load_database()
        # Execute the SQL query
        cursor.execute(sql_query)
        # Fetch all results
        results = cursor.fetchall()
        # Get column names from cursor.description
        column_names = [desc[0] for desc in cursor.description]

        return results, column_names

    except sqlite3.Error as e:  # Catch specific database errors
        return str(e)  # Return the error message as a string

    except sqlite3.OperationalError as e:
        print(f"Error connecting to the database: {e}")
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


def review_query_results(schema, user_request, sql_query, user_review):
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

        YOU MUST RETURN YOUR RESPONSE IN THIS EXACT FORMAT WITH ALL THREE SECTIONS:
        ANALYSIS: Your detailed analysis of what went wrong with the original query
        NEW_QUERY: Your revised SQL query without any markdown formatting
        EXPLANATION: Your detailed explanation of why the new query better matches the user's needs"""
    
    metadata = """ ****metadata table variables****
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
    # Create the human message template
    human_template = """
    SCHEMA: {schema}

    METADATA: {metadata}

    ORIGINAL REQUEST: {user_request}
    
    GENERATED SQL QUERY: {sql_query}
    
    USER REVIEW: {user_review}
    
    Analyze these and provide a better SQL query that addresses the user's needs."""
    
    # Create messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_template.format(
            schema = schema,
            metadata = metadata,
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
      
    # Define regex patterns for each section
    analysis_pattern = re.compile(r"ANALYSIS:\s*(.*?)(?=NEW_QUERY:|NEW QUERY:|$)", re.DOTALL | re.IGNORECASE)
    query_pattern = re.compile(r"NEW_QUERY:\s*(.*?)(?=EXPLANATION:|$)", re.DOTALL | re.IGNORECASE)
    explanation_pattern = re.compile(r"EXPLANATION:\s*(.*?)$", re.DOTALL | re.IGNORECASE)
    
    # Extract sections
    sections = {}
    
    # Extract analysis
    analysis_match = analysis_pattern.search(response_text)
    sections['analysis'] = analysis_match.group(1).strip() if analysis_match else "Analysis section could not be parsed from the response"
    
    # Extract query
    query_match = query_pattern.search(response_text)
    sections['new_query'] = query_match.group(1).strip() if query_match else "New query section could not be parsed from the response"
    
    # Extract explanation
    explanation_match = explanation_pattern.search(response_text)
    sections['explanation'] = explanation_match.group(1).strip() if explanation_match else "Explanation section could not be parsed from the response"
    
    return sections

class AgentStateViz(TypedDict):
    model: Any
    user_query: str
    sql: str
    user_viz: str
    data: str
    manager_guide: str
    code: str

def viz_manager_node(state: AgentStateViz):
    role_prompt = """
        You are a Data Visualization Manager specializing in converting SQL query results into effective visualizations. Your role is to analyze user requests and data samples to create clear specifications for the Code Specialist.
        Your Tasks:
        Visualization Selection:
        Determine the most appropriate chart type based on data structure and question
        Consider relationships, comparisons, distributions, or trends being explored
        Select appropriate visualization library (ATTENTION: only matplotlib or seaborn)

        Data Mapping:
        Identify which columns should be used for x-axis, y-axis, color, size, etc.
        Recommend any necessary data transformations
        Specify how to handle outliers or missing values

        Design Elements:
        Suggest appropriate title, labels, and annotations
        Recommend color schemes that enhance data readability
        Specify formatting for axes, legends, and other elements

        Implementation Guidance:
        Provide clear direction on how to implement the visualization
        Highlight any special considerations for the data structure
        Suggest any interactive elements if appropriate

        Your output must contain exactly these sections:
        Chart Type: Single sentence specifying the visualization type
        Data Mapping: Brief list of which columns map to which visual elements
        Design Elements: Concise specifications for titles, labels, and formatting
        Implementation Notes: Any special considerations for the Code Specialist

        Keep your response focused and practical. Your goal is to provide clear direction that allows the Code Specialist to implement the visualization without making design decisions.
        """
    instruction = rf"""
        Context:
        - Database Purpose: Power plant production analysis
        - Pay attention to the User viz request!
            *If it is "Write here what kind of data viz you want or let the AI Agent decide." or a non-sense message. It is your task to decide which is the best data viz for the data.*
        
        User query:
        {state['user_query']}

        SQL code:
        {state['sql']}

        User viz request:
        {state['user_viz']}

        Data Information:
        {state['data']}
        """
    messages = [
        SystemMessage(content=role_prompt),
        HumanMessage(content=instruction)
    ]
    llm = state['model']
    response = llm.invoke(messages)
    return {"manager_guide": [response.content]}

def viz_coder_node(state: AgentStateViz):
    role_prompt = """
        You are a Python Data Visualization Code Specialist. Your sole purpose is to implement visualization code based on specifications from the Data Visualization Manager.
        Your Tasks:

        Interpret Specifications:
        Read and understand the visualization requirements
        Follow the Chart Type, Data Mapping, Design Elements, and Implementation Notes exactly
        Do not deviate from the specifications provided

        Write Clean, Efficient Code:
        Use Python with visualization libraries as specified (ATTENTION: only matplotlib or seaborn)
        Write production-quality code with proper variable names and comments
        Include all necessary imports
        Structure code logically with clear data processing and visualization sections

        Handle Data Properly:
        Implement any required data transformations
        Handle NULL values, outliers, and edge cases appropriately
        Ensure code works with the provided DataFrame structure

        Format Output Correctly:
        Return ONLY Python code inside triple quotes (```)
        Include ALL necessary code to produce the visualization
        Ensure the code is complete and ready to run

        Response Format:
        You must respond ONLY with Python code inside triple quotes. Do not include explanations, comments outside the code, or any other text. Your entire response should look like this:
        ```
        # Imports
        import pandas as pd
        import matplotlib.pyplot as plt
        # (rest of the code)

        # Your implementation here
        ```
        Do not ask questions or provide alternatives. Simply implement the visualization exactly as specified by the Data Visualization Manager. Your code should assume the DataFrame is already available in the variable df.
        """
    instruction = rf"""
        Context:
        - Database Purpose: Power plant production analysis

        User query:
        {state['user_query']}

        SQL code:
        {state['sql']}

        User viz request:
        {state['user_viz']}

        Data Information:
        {state['data']}

        Data Visualization Manager Request:
        {state['manager_guide']}
        """
    messages = [
        SystemMessage(content=role_prompt),
        HumanMessage(content=instruction)
    ]
    llm = state['model']
    response = llm.invoke(messages)
    return {"code": [response.content]}

def create_workflow_viz():
    builder = StateGraph(AgentStateViz)

    builder.add_node("viz_manager", viz_manager_node)
    builder.add_node("viz_coder", viz_coder_node)

    builder.add_edge("viz_manager", "viz_coder")

    memory = MemorySaver()

    builder.set_entry_point("viz_manager")

    app = builder.compile(checkpointer=memory)
    return app

def format_df(df, max_sample_rows=3):
    """
    Creates a simple, standardized representation of a DataFrame for a visualization agent.
    
    Args:
        df: The pandas DataFrame to format
        sql_query: The SQL query that generated the data (optional)
        user_question: The original user question/request (optional)
        max_sample_rows: Maximum number of sample rows to include
    
    Returns:
        A formatted string with the DataFrame information
    """
    # Basic shape information
    df_info = f"DataFrame: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
    
    # Column names and types
    df_info += "Columns:\n"
    for col, dtype in zip(df.columns, df.dtypes):
        df_info += f"- {col} ({dtype})\n"
    df_info += "\n"
    
    # Sample data (first few rows)
    sample_rows = min(max_sample_rows, df.shape[0])
    df_info += f"Data Sample ({sample_rows} rows):\n"
    df_info += df.head(sample_rows).to_string()
    
    return df_info

def process_viz(question: str, graph: Any, llm:Any, sql: str, user_viz: str, data: str, thread_id: str = "1") -> Dict[str, Any]:
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
    'model': llm,
    'user_query': question,
    'sql': sql,
    'user_viz': user_viz,
    'data': data,
    'manager_guide': "",
    'code': ""
    }

     # Configure the thread
    thread = {"configurable": {"thread_id": thread_id}}

    # Stream the initial state through the graph
    for _ in graph.stream(initial_state, thread):
        pass  # Processing is handled internally by the graph

    # Retrieve and return the final state
    final_state = graph.get_state(thread)
    return final_state

def get_clean_viz_code(final_state) -> str:
    """
    Extract and clean visualization code from the agent's final state.
    
    Parameters
    ----------
    final_state : StateSnapshot or dict
        The final state returned by the visualization agent workflow
        
    Returns
    -------
    str
        The cleaned visualization code ready for execution
        
    Raises
    ------
    ValueError
        If no code is found in the final state
    """
    # Handle StateSnapshot objects from LangGraph
    try:
        # Try accessing as a StateSnapshot object
        if hasattr(final_state, 'values'):
            code_content = final_state.values.get('code')
        # Try accessing as a direct attribute
        elif hasattr(final_state, 'code'):
            code_content = final_state.code
        # Try dictionary access
        else:
            code_content = final_state.get('code')
    except:
        # Fallback to dict access
        try:
            code_content = final_state['code']
        except (TypeError, KeyError):
            raise ValueError("Cannot extract code from state object. Unsupported state format.")
    
    # Check if code content exists
    if not code_content:
        raise ValueError("No visualization code was generated in the agent output.")
    
    # Handle list format
    if isinstance(code_content, list):
        code_content = code_content[0]
        
    # Clean up code by removing markdown code blocks if present
    code = code_content.strip()
    
    # Remove Python markdown indicators if present
    if code.startswith("```python"):
        code = code[10:]
    elif code.startswith("```"):
        code = code[3:]
        
    if code.endswith("```"):
        code = code[:-3]
    
    # Return the clean code
    # Remove plt.show() from the code before execution
    code = code.replace("plt.show()", "# plt.show() removed for Streamlit compatibility")
    return code.strip()