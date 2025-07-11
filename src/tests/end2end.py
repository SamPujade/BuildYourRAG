import unittest
import os
import yaml
import shutil
from dotenv import load_dotenv
import chromadb
from streamlit.testing.v1 import AppTest

import sys

sys.path.append("./src/")

from rag.embedding_model import Multilingual
from database.doc_processing import process


TEST_FILE = "data/test/unit/unit_paper.pdf"


class RAGTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coll_name = "unit"
        with open("src/configs/config.yaml", "r") as config_file:
            cls.config = yaml.safe_load(config_file)
            cls.config["generation"]["LLM"] = "Gemini 1.5 Flash"
        cls.chroma_data_path = cls.config["dataset"]["CHROMA_DATA_PATH"]
        cls.upload_folder = os.path.join(cls.chroma_data_path, "upload/")
        cls.google_api_key = os.getenv("GOOGLE_API_KEY")
        cls.client = chromadb.PersistentClient(path=cls.chroma_data_path)
        cls.embedding_function = Multilingual().embedding_function

    def testApp(self):
        at = AppTest.from_file("app_frontend.py", default_timeout=100).run()
        collection_name = "unit_test"
        if collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(name=collection_name)

        # # Select new model
        # at.selectbox[0].select("Llama 3 8b")
        # self.assertEqual(at.selectbox[0].value, "Llama 3 8b")

        # Create new collection
        at.selectbox[1].select("Create new collection").run()
        at.text_input[0].input(collection_name).run()
        at.selectbox[1].select(collection_name).run()

        # Chat without files
        at.chat_input[0].set_value("Hi !").run()
        self.assertEqual(at.session_state["messages"][0]["content"], "Hi !")
        self.assertEqual(len(at.session_state["messages"]), 2)

        # Add file to new collection (at.file_uploader does not exist yet)
        collection = self.client.get_collection(
            name=collection_name, embedding_function=self.embedding_function
        )
        process(collection, TEST_FILE, self.config)
        shutil.copy(TEST_FILE, os.path.join(self.upload_folder, collection_name))
        at.run()
        self.assertEqual(at.checkbox[0].label, "unit_paper.pdf")

        # Chat with new file
        # at.selectbox[0].select("Mistral Nemo")
        at.chat_input[0].set_value("What is the title of the paper ?").run()
        self.assertIn(
            "Attention Is All You Need", at.session_state["messages"][-1]["content"]
        )

        # # Use API
        # at.toggle[0].set_value(True).run()
        # self.assertEqual(at.text_input[0].label, "Enter API key 👇")
        # at.text_input[0].input(self.google_api_key).run()
        # at.selectbox[0].select("Gemini 1.5 Flash").run()
        # at.chat_input[0].set_value("Hi !").run()
        # self.assertEqual(len(at.session_state["messages"]), 6)

        # Delete file
        at.checkbox[0].check().run()
        self.assertEqual(at.button[1].label, "Delete selected files")
        at.button[1].click().run()
        self.assertEqual(len(at.checkbox), 0)

        # Delete collection
        self.assertEqual(at.button[0].label, "Delete collection")
        self.assertEqual(at.selectbox[1].value, collection_name)
        self.assertIn(collection_name, [c.name for c in self.client.list_collections()])
        at.button[0].click().run()
        self.assertNotIn(
            collection_name, [c.name for c in self.client.list_collections()]
        )

        assert not at.exception


if __name__ == "__main__":
    load_dotenv()

    unittest.main()
