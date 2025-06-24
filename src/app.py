"""
Conversational RAG on PDF Input

Usage:
    streamlit run app.py
"""

import uuid
import requests
import streamlit as st


# FastAPI server base URL
BASE_URL = "http://127.0.0.1:8199"


class RAGApp:
    """
    A Streamlit application for interacting with an LLM (Language Model) that
    can answer questions based on uploaded documents or directly from user input.

    The application supports both retrieval-augmented generation (RAG) when documents
    are provided and direct LLM invocation when no documents are uploaded.
    """

    def __init__(self):
        self.session_id = uuid.uuid4()
        self.model_name = None
        self.base_config = self.get_config()

        self.initialize_ui()

    def get_config(self):
        response = requests.post(f"{BASE_URL}/get-config/", timeout=10)
        response_json = response.json()
        return response_json["config"]

    def initialize_ui(self):
        """
        Initialize the user interface (UI) of the Streamlit application, including the sidebar,
        main body, chat input, and message display. Sets up the LLM and, if documents are uploaded,
        prepares the retrieval chain for RAG.
        """
        st.set_page_config(
            page_title="BuildYourRAG",
            layout="wide",
            initial_sidebar_state="auto",
        )
        self.setup_sidebar()

        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "user_context" not in st.session_state:
            st.session_state["user_context"] = []

        self.display_chat_history()

        if prompt_user := st.chat_input("Type your message..."):
            self.handle_user_input(prompt_user)

        with st.expander("Show document"):
            if "context" in st.session_state and st.session_state["context"]:
                context_text = "\n\n".join(st.session_state["context"])
                st.write(f"« {context_text} »")

        self.apply_styles()

    def setup_sidebar(self):
        """
        Set up the sidebar in the Streamlit application, including the file uploader for PDFs.
        If PDFs are uploaded, the documents are loaded and stored.
        """
        with st.sidebar:
            response = requests.post(
                f"{BASE_URL}/get-names/",
                json={},
                timeout=10,
            )
            response_json = response.json()

            self.model_name = st.selectbox(
                "Select a model",
                response_json["model_names"],
                index=response_json["model_index"],
            )

            with st.sidebar.expander("Hyperparameters", expanded=False):
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.base_config["generation"]["TEMPERATURE"],
                    step=0.1,
                    format="%.1f",
                )
                layout = st.checkbox(
                    "Layout", value=self.base_config["processing"]["LAYOUT"]
                )
                sub_chunking = st.checkbox(
                    "Sub-chunking", value=self.base_config["processing"]["SUB_CHUNKING"]
                )
                figure_extraction = st.checkbox(
                    "Figure Extraction",
                    value=self.base_config["processing"]["FIGURE_EXTRACTION"],
                )
                top_k = st.slider(
                    "Top K",
                    min_value=1,
                    max_value=10,
                    value=self.base_config["retrieval"]["TOP_K"],
                    step=1,
                )

                requests.post(
                    f"{BASE_URL}/update-config/",
                    json={
                        "temperature": temperature,
                        "layout": layout,
                        "sub_chunking": sub_chunking,
                        "figure_extraction": figure_extraction,
                        "top_k": top_k,
                    },
                    timeout=10,
                )

            self.collection_name = st.selectbox(
                "Select a collection",
                response_json["collection_list"] + ["Create new collection"],
                index=response_json["collection_index"],
                disabled=False,
            )

            if self.collection_name == "Create new collection":
                new_collection_name = st.text_input(
                    "Enter the name for the new collection"
                )
                if new_collection_name:
                    requests.post(
                        f"{BASE_URL}/create-collection/",
                        json={
                            "collection_name": new_collection_name,
                        },
                        timeout=100,
                    )
                    st.success(
                        f"Collection '{new_collection_name}' created successfully!"
                    )
                    st.rerun()
                    self.collection_name = new_collection_name

            else:
                if st.button("Delete collection"):
                    requests.post(
                        f"{BASE_URL}/delete-collection/",
                        json={"files": [], "collection_name": self.collection_name},
                        timeout=100,
                    )
                    st.rerun()

                with st.form("upload-form", clear_on_submit=True, border=False):
                    uploaded_files = st.file_uploader(
                        "Choose a PDF file", accept_multiple_files=True, type="pdf"
                    )
                    submitted = st.form_submit_button("Add files")

                if submitted and uploaded_files is not None:
                    files = [
                        ("files", (file.name, file.getvalue(), file.type))
                        for file in uploaded_files
                    ]
                    with st.spinner("Processing files..."):
                        requests.post(
                            f"{BASE_URL}/upload/",
                            files=files,
                            data={"collection_name": self.collection_name},
                            timeout=1000,
                        )
                        st.success("Files successfully added !")

                st.header("Collection Files")
                response = requests.post(
                    f"{BASE_URL}/list-files/",
                    json={
                        "collection_name": self.collection_name,
                    },
                    timeout=10,
                )
                response_json = response.json()
                checked_files = [
                    file
                    for file in response_json["files_list"]
                    if st.checkbox(file, key=file)
                ]

                if st.button("Delete selected files"):
                    requests.post(
                        f"{BASE_URL}/delete-files/",
                        json={
                            "files": checked_files,
                            "collection_name": self.collection_name,
                        },
                        timeout=100,
                    )
                    st.rerun()

    def generate_response(self, prompt_user):
        """
        Generate a response to the user's input. If documents are uploaded and the RAG chain is available,
        the response is generated using retrieval-augmented generation. Otherwise, the response is generated
        directly by invoking the LLM.

        Args:
            prompt_user (str): The user's input prompt.

        Returns:
            tuple: A tuple containing the response text and token counts (which are placeholders here).
        """
        history = st.session_state["messages"][:-1]
        user_context = st.session_state["user_context"]
        # if len(history) % 2 == 1:
        #     history.pop()  # remove last message

        response = requests.post(
            f"{BASE_URL}/generate-response/",
            json={
                "model_name": self.model_name,
                "collection_name": self.collection_name,
                "prompt_user": prompt_user,
                "history": history,
                "user_context": user_context,
            },
            timeout=1000,
        )
        response_json = response.json()

        return (
            response_json["output"],
            response_json["context"],
            response_json["user_context"],
        )

    def display_chat_history(self):
        """
        Display the chat history stored in the session state, rendering each message in the chat UI.
        """
        if not st.session_state["messages"]:
            response = requests.post(
                f"{BASE_URL}/initial-message/",
                timeout=10,
            )
            initial_message = response.json()["initial_message"]
            if initial_message:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": initial_message}
                )

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self, prompt_user):
        """
        Handle the user's input by generating a response from the LLM (or RAG chain) and updating
        the chat history.

        Args:
            prompt_user (str): The user's input prompt.
        """
        if hasattr(self, "collection_name"):
            st.session_state["messages"].append(
                {"role": "user", "content": prompt_user}
            )
            st.chat_message("user").write(prompt_user)

            with st.spinner("Thinking..."):
                full_response, context, user_context = self.generate_response(
                    prompt_user
                )

            st.chat_message("assistant").write(full_response)
            st.session_state["messages"].append(
                {"role": "assistant", "content": full_response}
            )
            st.session_state["context"] = context
            st.session_state["user_context"].append(user_context)
        else:
            st.error("Please select or create a collection first", icon="🚨")

    def apply_styles(self):
        """
        Apply custom CSS styles to the Streamlit app to enhance the UI/UX.
        """
        with open("src/styles.css", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


if __name__ == "__main__":
    RAGApp()
