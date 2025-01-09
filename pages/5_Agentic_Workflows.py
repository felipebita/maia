import streamlit as st

def main():
    st.set_page_config(page_title="Agentic Workflows", page_icon=":chains:")
    with st.sidebar:
        st.image("img/logo_sq.png")
        st.markdown("This is a portfolio project by Felipe Martins. If you want to see the code of this app and other data science projects check my [GitHub](https://github.com/felipebita).")
        st.markdown("This is just an example tool. Please, do not abuse on my OpenAI credits, use it only for testing purposes.")

    st.header("Agentic Workflows :chains:")
    tab1, tab2 = st.tabs(["Essay Evaluator ", "Text Analysis"])

    with tab1:
        st.write("This is the content of the first tab.")
        essay = st.text_input("Insert the essay to be evaluated:")
        if st.button("Evaluate"):
            st.write("The essay has been evaluated.")

    with tab2:
        st.write("This is the content of the second tab.")
    
if __name__ == '__main__':
    main()
    