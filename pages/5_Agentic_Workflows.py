import streamlit as st
import src.src_aw1 as src1
import src.src_aw2 as src2
import src.src_aw3 as src3

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
    tab1, tab2, tab3 = st.tabs(["Essay Grading ", "Text Processing", "Natural Language to SQL"])


    with tab1:
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
        **Reference Date:** din_instante  
        **Plant Subsystem:** id_subsistema  
        **Subsystem Name:** nom_subsistema  
        **State where the Plant is located:** id_estado  
        **State Name:** nom_estado  
        **Plant Operation Mode:** cod_modalidadeoperacao  
        **Plant Type:** nom_tipousina  
        **Fuel Type:** nom_tipocombustivel  
        **Plant Name:** nom_usina  
        **ONS Identifier:** id_ons  
        **Unique Generation Enterprise Code:** ceg_label  
        **Power Generation in MWmed:** val_geracao  
        """)
        if st.button("Get Schema"):
            schema = src3.get_database_schema()
            st.write(schema)
        query = st.text_input("What do you want to know in this database:")
        if st.button("Query"):
            app = src3.create_workflow()
            query_processed = src3.process_question(query,app)
            st.code(query_processed.values['sql'], language="sql")
            query_code = src3.extract_sql(query_processed.values['sql'])
            print(src3.validate_and_execute_sql(query_code))
            results, column_names = src3.execute_sql_query(query_code)
            df = src3.create_dataframe(column_names, results)
            st.dataframe(df)



            
    
if __name__ == '__main__':
    main()
    