"""
Unit Testing Script

Description:

This script runs unit tests using the `unittest` framework.

Usage :

Run all unit tests : 

python unit.py

-------------------------------------------

Run onlt the app test :

python unit.py RAGTest.testApp


"""

import unittest
import os
import sys
import yaml
import shutil
from dotenv import load_dotenv

import chromadb

sys.path.append("./src/")

from rag.embedding_model import Multilingual
from rag.pipeline import RAGPipeline
from database.doc_processing import process
from rag.generation_model import MistralNemo, GeminiFlash


TEST_FILE = "data/test/unit/unit_paper.pdf"


class RAGTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coll_name = "unit"
        with open("src/configs/config.yaml", "r") as config_file:
            cls.config = yaml.safe_load(config_file)
            cls.config["generation"]["LLM"] = "Mistral Nemo"
        cls.chroma_data_path = cls.config["dataset"]["CHROMA_DATA_PATH"]
        cls.upload_folder = os.path.join(cls.chroma_data_path, "upload/")
        cls.google_api_key = os.getenv("GOOGLE_API_KEY")
        cls.client = chromadb.PersistentClient(path=cls.chroma_data_path)
        cls.embedding_function = Multilingual().embedding_function

    def testCreateCollection(self):
        if self.coll_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(name=self.coll_name)

        collection = self.client.create_collection(
            name=self.coll_name, embedding_function=self.embedding_function
        )

        process(collection, TEST_FILE, self.config)
        db = collection.get()

        self.assertTrue(len(db["documents"]) > 0)
        self.assertEqual(db["metadatas"][0]["chunk"], 0)


    def testCreatePipeline(self):
        rag_pipeline = RAGPipeline(self.coll_name, config=self.config)

        self.assertEqual(rag_pipeline.retriever.collection.name, self.coll_name)


    def testQuery(self):
        rag_pipeline = RAGPipeline(self.coll_name, config=self.config)
        query = "What is the title of the paper ?"
        output, context = rag_pipeline.predict(query, [])

        self.assertIsInstance(output, str)
        self.assertIsInstance(context, list)
        self.assertTrue("Attention Is All You Need" in output)


    def testGeminiQuery(self):
        model = GeminiFlash(api_key=self.google_api_key)
        query = "Hi how are you"
        output = model.predict(query, [])

        self.assertIsInstance(output, str)


    def testLlamaCPPQuery(self):
        model = MistralNemo()
        query = "Hi how are you"
        output = model.predict(query, [])

        self.assertIsInstance(output, str)

    def testRouter(self):
        model = GeminiFlash(api_key=self.google_api_key)
        rag_pipeline = RAGPipeline(self.coll_name, config=self.config, model=model)
        query = "Hi how are you"
        output = rag_pipeline.router.query_routing(query)

        self.assertIsInstance(output, str)
        self.assertTrue(output == "Other")

        query_2 = "Hi, what is the title of this document ?"
        output_2 = rag_pipeline.router.query_routing(query_2)

        self.assertIsInstance(output_2, str)
        self.assertTrue(output_2 == "Context")


    def testMultipleDocs(self):
        coll_name_mult = self.coll_name+  "_mult"
        if coll_name_mult in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(name=coll_name_mult)
        collection = self.client.create_collection(
            name=coll_name_mult, embedding_function=self.embedding_function
        )
        collection.add(
            documents=["First document"],
            ids=["id1"],
            metadatas=[{"from": "first_doc.pdf", "chunk": 0}],
        )
        process(collection, TEST_FILE, self.config)
        metadatas_doc_2 = collection.get(where={"from": "unit_paper.pdf"})["metadatas"]
        
        self.assertTrue(1 in list(set([m['chunk'] for m in metadatas_doc_2])))
        self.assertFalse(0 in list(set([m['chunk'] for m in metadatas_doc_2])))


if __name__ == "__main__":
    load_dotenv()

    unittest.main()

    # script_path = os.path.abspath("app.py")
    # at = AppTest.from_file(script_path, default_timeout=30).run()
    # assert not at.exception
