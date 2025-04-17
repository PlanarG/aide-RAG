import os
import json
import logging

import numpy as np
from typing import List
from pathlib import Path
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from aide.utils.config import Config, _load_cfg

logger = logging.getLogger("aide")
embed_model = None
tokenizer = None

class Retriever:
    def _load_data_dir(self, doc_dir: Path):
        if not doc_dir.exists() or not doc_dir.is_dir():
            raise ValueError(f"Document directory {doc_dir} does not exist or is not a directory.")

        doc_info_path = doc_dir / "info.json"
        if not doc_info_path.exists():
            raise ValueError(f"Metadata {doc_info_path} does not exist in the document directory.")
        
        with open(doc_info_path, "r") as f:
            self.info = json.load(f)
        
        config_keys = list(self.info.keys())
        doc_names = doc_dir.glob("*.txt")

        self.documents = []
        self.documents_by_id = {}
        for doc in doc_names:
            if doc.stem not in config_keys:
                raise ValueError(f"Document {doc.stem} is not listed in the info.json file. Perhaps the document folder is corrupted.")
            if "votes" not in self.info[doc.stem]:
                raise ValueError(f"Document {doc.stem} does not have a 'votes' key in the configuration file. ")

            with open(doc_dir / doc.name, "r") as f:
                text = f.read()
            if len(text) == 0:
                continue
            self.documents_by_id[doc.stem] = text
            self.documents.append(Document(
                page_content=text, 
                metadata={
                    "votes": self.info[doc.stem]["votes"],
                    "title": self.info[doc.stem]["title"],
                    "id": doc.stem
                }
            ))
    
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def get_embedding(self, inputs: List[str], batch_size=32) -> Tensor:
        """Get the embedding for a list of inputs."""
        if len(inputs) > batch_size:
            results = []
            for i in tqdm(range(0, len(inputs), batch_size), desc="Getting embeddings"):
                results.append(self.get_embedding(inputs[i:i + batch_size]))
            return torch.cat(results, dim=0)
        
        global embed_model, tokenizer
        batch_dict = tokenizer(inputs, max_length=512, padding=True, truncation=True, return_tensors="pt")
        batch_dict = { k: v.to(self.device) for k, v in batch_dict.items() }
        with torch.no_grad():
            outputs = embed_model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)

    def __init__(self, cfg: Config, doc_dir: Path):
        self.cfg = cfg
        self.doc_dir = doc_dir

        self._load_data_dir(doc_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        global embed_model, tokenizer

        if embed_model is None:
            embed_model = AutoModel.from_pretrained(cfg.retriever.embed_model)
            tokenizer = AutoTokenizer.from_pretrained(cfg.retriever.embed_model)
            embed_model.to(self.device)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.retriever.max_chunk_size,
            chunk_overlap=cfg.retriever.chunk_overlap,
        )

        self.split_docs = []
        for doc in self.documents:
            for chunk in splitter.split_documents([doc]):
                chunk.metadata["votes"] = doc.metadata["votes"]
                self.split_docs.append(chunk)

        content = [split_doc.page_content for split_doc in self.split_docs]
        self.embeddings = self.get_embedding(content)

        logger.info(f"Document directory {doc_dir} loaded with {len(self.documents)} documents.")
    
    def get_detailed_instruct(self, query: str) -> str:
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        return f'Instruct: {task}\nQuery: {query}'
    
    def get_hotest_docs(self, k = 10) -> List[tuple[str, str, int]]:
        """Get the top k hottest documents based on votes."""
        if k > len(self.documents):
            raise ValueError(f"Requested {k} documents, but only {len(self.documents)} are available.")

        sorted_docs = sorted(
            self.documents, 
            key=lambda x: x.metadata["votes"], 
            reverse=True
        )
        
        results = []
        for doc in sorted_docs[:k]:
            results.append((doc.metadata["title"], doc.metadata["id"], doc.metadata["votes"]))
        
        return results


    def _calc_score(self, raw_score, vote):
        return raw_score * np.clip(np.log(np.log(vote + 1) + 1), 0.5, 1.5)
    
    
    def get_relevant_docs(self, query: str, by: str="content") -> List[str]:
        if by not in ["content", "id"]:
            raise ValueError(f"Invalid value for 'by': {by}. Expected 'content' or 'id'.")

        if by == "content":
            query = self.get_detailed_instruct(query)
            q_embeddings = self.get_embedding([query])

            scores = (q_embeddings @ self.embeddings.T)[0]
            raw_results = [(doc, self._calc_score(scores[id_], doc.metadata["votes"])) for id_, doc in enumerate(self.split_docs)]

            scored_results = sorted(
                raw_results, 
                key=lambda x: x[1],
                reverse=True
            )[:self.cfg.retriever.k]

            results, doc_ids = [], []
            for doc, _ in scored_results:
                if doc.metadata["id"] not in doc_ids:
                    results.append(self.documents_by_id[doc.metadata["id"]])
                    doc_ids.append(doc.metadata["id"])
                if len(results) >= self.cfg.retriever.k:
                    break
            
        else:
            ids = query.split("\n")
            results = []
            for id in ids:
                if id in self.documents_by_id:
                    results.append(self.documents_by_id[id])

        return results

if __name__ == '__main__':
    cfg = _load_cfg()
    cfg.doc_base_dir = Path("/home/planarg/aideml/aide/example_tasks/aerial-cactus-identification/docs")
    print(cfg.retriever)
    retriever = Retriever(cfg, cfg.doc_base_dir / "discussions")
    print(retriever.get_hotest_docs(k=5))
    print(retriever.get_relevant_docs("How to use fastai", by="content"))
    print(retriever.get_relevant_docs("93575", by="id"))







    
