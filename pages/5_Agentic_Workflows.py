import streamlit as st
from streamlit_ace import st_ace
import src.src_aw1 as src1
import src.src_aw2 as src2
import src.src_aw3 as src3
from langchain_openai import ChatOpenAI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

def main():
    st.set_page_config(page_title="Agentic Workflows", page_icon=":chains:")
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            div[data-testid="stSidebarNav"] {display: none;}
        </style>
        """, unsafe_allow_html=True)
    
    st.sidebar.image("img/logo_sq.png")

    # Create the sidebar navigation
    pages = {
    "🏠 Home": "Home.py",
    "📚 Chat with Documents": "pages/1_Chat_with_Documents.py",
    "📝 Text Summarization": "pages/2_Text_Summarization.py",
    "🖼️ Image Creation": "pages/3_Image_Creation.py",
    "📹 Sentinel": "pages/4_Sentinel.py",
    "🔄 Agentic Workflows": "pages/5_Agentic_Workflows.py"
    }

    # Add some styling to make the sidebar links more visible
    st.markdown(
    """
    <style>
    .css-1d391kg {
        padding-top: 0rem;
    }
    .css-1d391kg a {
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    # Create links for each page
    for page_name, page_path in pages.items():
        st.sidebar.page_link(page_path, label=page_name)

    # Add the messages in the sidebar
    st.sidebar.markdown("---")  # This adds a horizontal line for separation
    st.sidebar.markdown("This is a portfolio project by Felipe Martins. If you want to see the code of this app and other data science projects check my [GitHub](https://github.com/felipebita).")
    st.sidebar.markdown("This is just an example tool. Please, do not abuse on my OpenAI credits, use it only for testing purposes.")

    # Hide fullscreen button for image
    hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''
    st.markdown(hide_img_fs, unsafe_allow_html=True)

    st.header("Agentic Workflows 🔄")
    tab1, tab2, tab3 = st.tabs(["Natural Language to SQL", "Text Processing", "Essay Grading"])

    with tab1:
        st.markdown(
                """
                <h2 style="text-align: center;">Natural Language to SQL (NL2SQL)</h2>
                <p style="text-align: justify;">
                    This project implements an Agentic Workflow that converts natural language queries into SQL to analyze energy generation data in Brazil. 
                    It enables seamless interaction with a powerplant production database, making data retrieval intuitive and efficient.
                </p>
                <h2 style="text-align: center;">ONS Dataset</h2>
                <p style="text-align: justify;">
                    The <strong>National System Operator (ONS)</strong> is responsible for coordinating and overseeing the operation 
                    of electricity generation and transmission facilities within the <strong>National Interconnected System (SIN)</strong>. 
                    Additionally, ONS is tasked with planning the operations of isolated energy systems across the country, 
                    operating under the supervision and regulation of the <strong>National Electric Energy Agency (ANEEL)</strong>.
                </p>

                <p style="text-align: justify;">
                    ONS provides publicly available energy production data on its 
                    <a href="https://dados.ons.org.br/dataset/geracao-usina-2" target="_blank">official website</a>.
                </p>

                <p style="text-align: justify;">
                    The process of acquiring, processing, and analyzing this data is detailed in this  
                    <a href="https://github.com/felipebita/energy_forecasting" target="_blank">forecasting project</a>, available in my portfolio.
                </p>
                """,
                unsafe_allow_html=True
            )

        with st.expander("**Data Information**"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    """
                    <style>
                        .compact-table p { margin: 2px 0; line-height: 1.1; }
                    </style>
                    <div class="compact-table">
                        <p><strong>Table:</strong> <code>ons_ts</code></p>
                        <p><strong>Columns:</strong></p>
                        <p><code>id</code> (INTEGER)</p>
                        <p><code>year</code> (INTEGER)</p>
                        <p><code>quarter</code> (INTEGER)</p>
                        <p><code>month</code> (INTEGER)</p>
                        <p><code>ceg_label</code> (TEXT)</p>
                        <p><code>group</code> (TEXT)</p>
                        <p><code>val_geracao</code> (REAL)</p>
                    </div>
                    """, unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <style>
                        .compact-table p { margin: 2px 0; line-height: 1.1; }
                    </style>
                    <div class="compact-table">
                        <p><strong>Table:</strong> <code>ons_metadata</code></p>
                        <p><strong>Columns:</strong></p>
                        <p><code>id_subsistema</code> (TEXT)</p>
                        <p><code>nom_subsistema</code> (TEXT)</p>
                        <p><code>id_estado</code> (TEXT)</p>
                        <p><code>nom_estado</code> (TEXT)</p>
                        <p><code>cod_modalidadeoperacao</code> (TEXT)</p>
                        <p><code>nom_tipousina</code> (TEXT)</p>
                        <p><code>nom_tipocombustivel</code> (TEXT)</p>
                        <p><code>nom_usina</code> (TEXT)</p>
                        <p><code>id_ons</code> (TEXT)</p>
                        <p><code>ceg</code> (TEXT)</p>
                        <p><code>ceg_label</code> (TEXT)</p>
                        <p><code>group</code> (TEXT)</p>
                    </div>
                    """, unsafe_allow_html=True
                )


            st.markdown(
                    """
                    <div style="text-align: center; font-size: 18px; line-height: 1.2;">
                        <p style="margin: 2px 0;"><strong>Power Generation in MWmed:</strong> <code>val_geracao</code></p>
                        <p style="margin: 2px 0;"><strong>Unique Generation Enterprise Code:</strong> <code>ceg_label</code></p>
                        <p style="margin: 2px 0;"><strong>Plant Subsystem:</strong> <code>id_subsistema</code></p>
                        <p style="margin: 2px 0;"><strong>Subsystem Name:</strong> <code>nom_subsistema</code></p>
                        <p style="margin: 2px 0;"><strong>State where the Plant is located:</strong> <code>id_estado</code></p>
                        <p style="margin: 2px 0;"><strong>State Name:</strong> <code>nom_estado</code></p>
                        <p style="margin: 2px 0;"><strong>Plant Operation Mode:</strong> <code>cod_modalidadeoperacao</code></p>
                        <p style="margin: 2px 0;"><strong>Plant Type:</strong> <code>nom_tipousina</code></p>
                        <p style="margin: 2px 0;"><strong>Fuel Type:</strong> <code>nom_tipocombustivel</code></p>
                        <p style="margin: 2px 0;"><strong>Plant Name:</strong> <code>nom_usina</code></p>
                        <p style="margin: 2px 0;"><strong>ONS Identifier:</strong> <code>id_ons</code></p>
                    </div>    
                    """,
                    unsafe_allow_html=True
                )

        with st.expander("**Examples**"):
            st.markdown(
                    """
                    <div style="text-align: left; font-size: 18px; line-height: 1.2;">
                        <p style="margin: 2px 0;">Which are the powerplants in the dataset?</p>
                        <p style="margin: 2px 0;">What is the total production of each powerplant?</p>
                        <p style="margin: 2px 0;">I want to know the mean monthly production of each powerplant in each of the years available.</p>
                        <p style="margin: 2px 0;">I want to know the mean monthly production of each powerplant in each of the years available. Make the years as columns in the answer.</p>
                        <p style="margin: 2px 0;">Which are the three powerplants with the highest energy production in each of the last three years? </p>
                    </div>    
                    """,
                    unsafe_allow_html=True
                )
        model = st.radio("Choose the model", ["gpt-4o-mini", "gpt-4o"])

        # Store the query input in session state to persist its value
        if "query" not in st.session_state:
            st.session_state.query = ""

        st.session_state.query = st.text_area("What do you want to know about the Powerplants production:", value=st.session_state.query)

        # Initialize other session state variables
        if "query_run" not in st.session_state:
            st.session_state.query_run = False
        if "query_code" not in st.session_state:
            st.session_state.query_code = ""
        if "review" not in st.session_state:
            st.session_state.review = False
        if "query_text" not in st.session_state:
            st.session_state.query_text = ""
        if "review_expander_open" not in st.session_state:
            st.session_state.review_expander_open = False
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'review_response' not in st.session_state:
            st.session_state.review_response = ""

        if st.button("Get Query"):
            st.session_state.llm = ChatOpenAI(model=model, temperature=0)
            app = src3.create_workflow_sql()
            query_processed = src3.process_question(st.session_state.query, app, st.session_state.llm)
            st.session_state.query_code = src3.extract_sql(query_processed.values['sql'])
            st.session_state.query_run = True

        if st.session_state.query_run == True:
            st.session_state.query_text = st_ace(
                value=st.session_state.query_code,  
                language="sql",  
                theme="github",  
                font_size=14,  
                height=400,  
                show_gutter=True,  
                wrap=True,  
            )

        elif st.session_state.review_response:
            st.session_state.query_text = st_ace(
                value= st.session_state.review_response.get("new_query", "New query not found"),  
                language="sql",  
                theme="github",  
                font_size=14,  
                height=400,  
                show_gutter=True,  
                wrap=True 
            )
        else:
            st.session_state.query_text = st_ace(
                value=st.session_state.query_code,  
                language="sql",  
                theme="github",  
                font_size=14,  
                height=400,  
                show_gutter=True,  
                wrap=True 
            )

        # Run the query and show the results
        if st.session_state.query_text:
            if st.button("Run"):
                query_to_execute = st.session_state.query_text
                result = src3.execute_sql_query(query_to_execute)

                if isinstance(result, tuple):
                    results, column_names = result
                    st.session_state.df = src3.create_dataframe(column_names, results)
                    st.session_state.review = True  # Update session state BEFORE rerunning
                    st.rerun()  # Restart execution to reflect changes instantly
                elif isinstance(result, str):
                    st.error(f"SQL Query Error: {result}")
                else:
                    st.error("Failed to connect to the database. Check the console for details.")

        # Always display DataFrame if available
        if 'df' in st.session_state and st.session_state.df is not None:
            st.dataframe(st.session_state.df)

            if st.session_state.review:
                with st.expander("Review results", expanded=st.session_state.get("review_expander_open", True)):
                    user_review = st.text_area(
                            "Is it wrong?",
                            value="""Write here what is wrong with the result or how do you actually want it.""",
                            height=64)
                    if st.button("Review"):
                        st.session_state.review_response = src3.review_query_results(
                            src3.get_database_schema(),
                            st.session_state.query,
                            st.session_state.query_text,
                            user_review)    
                        st.markdown(st.session_state.review_response.get("analysis", "Analysis section not found"))
                        st.markdown(st.session_state.review_response.get("explanation", "Explanation section not found"))
                        st.markdown("Copy the code below and paste in the Edit Query expander")
                        st.code(st.session_state.review_response.get("new_query", "New query not found"), language="sql")
                    st.session_state.query_run = False
                viz_query = st.text_area(
                        "Do you want a visualization for this data?",
                        value="""Write here what kind of data viz you want or let the AI Agent decide.""",
                        height=64)
                if st.button("Get Viz"):
                    app_viz = src3.create_workflow_viz()
                    df_head = src3.format_df(st.session_state.df, 3)
                    viz_state = src3.process_viz(st.session_state.query, app_viz, st.session_state.llm, st.session_state.query_code, viz_query, df_head)
                        # Extract clean code
                    clean_code = src3.get_clean_viz_code(viz_state)
                    
                    # Display the code if desired
                    with st.expander("View Visualization Code"):
                        st.code(clean_code, language="python")
                    
                    # Execute the code (df should be available in this scope)
                    namespace = {
                        'df': st.session_state.df,
                        'pd': pd,
                        'plt': plt,
                        'sns': sns,
                        'np': np,
                        'px': px
                    }
                    # Execute with the custom namespace
                    exec(clean_code, namespace)
                    # If a figure was created, display it using Streamlit
                    if plt.get_fignums():
                        fig = plt.gcf()  # Get current figure
                        
                        # Option 1: Set figure size before displaying
                        fig.set_size_inches(10, 6)  # Width, height in inches
                        
                        # Option 2: Use Streamlit's width parameter
                        st.pyplot(fig, use_container_width=True)  # Scales to container width
                        
                        # Option 3: Specify exact width
                        # st.pyplot(fig, width=800)  # Specific width in pixels
                        
                        plt.close()  # Close the figure to free memory
                    
  





    with tab2:
        st.markdown("""
            ### Text Processing 
                    
            This workflow automates the processing of text through three sequential tasks, each powered by GPT-4o-mini for precise natural language understanding:
            
            Classification: Categorizes the text as News, Blog, Research, or Other.
                    
            Entity Extraction: Identifies entities such as Persons, Organizations, and Locations.
                    
            Summarization: Generates a concise summary of the text.
            """)       
        text = st.text_input("Insert the text:")
        if st.button("Process"):
            app = src2.create_workflow()
            result = src2.process_text(app, text)

            st.write(result)
    

    with tab3:
        st.markdown("""
            ### Essay Grading

            This is an intelligent essay grading system that evaluates essays through a series of sequential checks, each focusing on different aspects of writing quality. The workflow follows these steps:

            1. First checks the essay's relevance to the topic (30% of final grade)
            2. If relevance score > 0.5, proceeds to check grammar (20% of final grade)
            3. If grammar score > 0.6, analyzes essay structure (20% of final grade)
            4. If structure score > 0.7, evaluates depth of analysis (30% of final grade)
            5. Finally calculates a weighted final score

            Each step uses GPT-4 mini to evaluate the essay and provide a score between 0 and 1. If an essay fails to meet the minimum threshold at any step, it skips remaining checks and proceeds directly to final scoring.
            """)       

        # Define maximum character limit
        MAX_CHARS_THEME = 250  # adjust this number as needed
        MAX_CHARS_ESSAY = 3000  # adjust this number as needed

        # Create text area for theme writing
        theme = st.text_area("Insert the essay theme:", 
                height=50,  # adjust height as needed
                max_chars=MAX_CHARS_THEME,
                help=f"Maximum {MAX_CHARS_THEME} characters")
        
        # Create text area for essay writing
        essay = st.text_area("Insert the essay to be evaluated:", 
                height=300,  # adjust height as needed
                max_chars=MAX_CHARS_ESSAY,
                help=f"Maximum {MAX_CHARS_ESSAY} characters")

        # Show character count
        char_count_e = len(essay)
        st.caption(f"Character count: {char_count_e}/{MAX_CHARS_ESSAY}")

        if st.button("Evaluate"):
            if char_count_e == 0:
                st.error("Please enter an essay before evaluating.")
            elif char_count_e < 100:  # Optional: set minimum length
                st.warning("Essay is too short. Please write at least 100 characters.")
            else:
                app = src1.create_essay_workflow()
                result = src1.grade_essay(app, theme, essay)
                st.write(f"The essay has been evaluated.\n\n{result}")
                
    
if __name__ == '__main__':
    main()
    