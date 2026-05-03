import os
import json
import numpy as np
import torch
import re
from sentence_transformers import SentenceTransformer
import h5py
from src.utils import tokenCounter
import faiss

# 强制设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class database():
    def __init__(self, db_path, embedding_model) -> None:
        print(f"[*] Initializing Database (Local Mode)...")
        
        # 1. 检测设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"    > Device: {self.device}")

        # 2. 加载模型
        print(f"    > Loading Model: {embedding_model}...")
        try:
            self._prepare_local_nomic_remote_code(embedding_model)
            self.embedding_model = SentenceTransformer(
                embedding_model, 
                trust_remote_code=True,
                device=str(self.device)
            )
            print("    > [Success] Model loaded successfully!")
        except Exception as e:
            print(f"\n[!!! Error !!!] Model load failed: {e}")
            raise e

        # 3. 加载 JSON 数据库
        print(f"    > Loading JSON Database...")
        self.paper_index = {}
        self.normalized_title_to_id = {}
        json_path = f'{db_path}/arxiv_paper_db.json'
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    data = raw_data.get('cs_paper_info', raw_data.get('_default', raw_data))
                    for k, v in data.items():
                        if isinstance(v, dict) and 'id' in v:
                            self.paper_index[v['id']] = v
                            norm_title = self._normalize_title(v.get('title', ''))
                            if norm_title and norm_title not in self.normalized_title_to_id:
                                self.normalized_title_to_id[norm_title] = v['id']
                print(f"    > Loaded {len(self.paper_index)} papers.")
            except Exception as e:
                print(f"    [Error] JSON corrupted: {e}")
        
        self.token_counter = tokenCounter()

        # 4. 加载 FAISS
        print(f"    > Loading FAISS Indices...")
        self.has_faiss = False
        try:
            if os.path.exists(f'{db_path}/faiss_paper_title_embeddings.bin'):
                self.title_loaded_index = faiss.read_index(f'{db_path}/faiss_paper_title_embeddings.bin')
                self.abs_loaded_index = faiss.read_index(f'{db_path}/faiss_paper_abs_embeddings.bin')
                self.id_to_index, self.index_to_id = self.load_index_arxivid(db_path)
                self.has_faiss = True
                print("    > [Success] FAISS ready.")
            else:
                print("    [Warning] FAISS files missing.")
        except Exception as e:
            print(f"    [Error] FAISS load failed: {e}")

    def _prepare_local_nomic_remote_code(self, embedding_model):
        if not os.path.isdir(embedding_model):
            return

        required_files = [
            "configuration_hf_nomic_bert.py",
            "modeling_hf_nomic_bert.py",
        ]
        missing = [name for name in required_files if not os.path.exists(os.path.join(embedding_model, name))]
        if not missing:
            return

        cache_root = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/nomic-ai/nomic-bert-2048")
        if not os.path.isdir(cache_root):
            return

        for root, _, files in os.walk(cache_root):
            for name in missing[:]:
                if name in files:
                    src = os.path.join(root, name)
                    dst = os.path.join(embedding_model, name)
                    try:
                        with open(src, "r", encoding="utf-8") as fsrc:
                            content = fsrc.read()
                        with open(dst, "w", encoding="utf-8") as fdst:
                            fdst.write(content)
                        missing.remove(name)
                    except Exception:
                        pass
            if not missing:
                break

    def load_index_arxivid(self, db_path):
        with open(f'{db_path}/arxivid_to_index_abs.json','r') as f:
            mapping = json.load(f)
        return {k: int(v) for k, v in mapping.items()}, {int(v): k for k, v in mapping.items()}

    def _normalize_title(self, title):
        title = str(title or "").lower().strip()
        title = title.replace("&", " and ")
        title = re.sub(r'[\'"`“”‘’]', '', title)
        title = title.replace('$', ' ')
        title = title.replace('\\', ' ')
        title = re.sub(r'[\-_:;,./()\[\]{}]+', ' ', title)
        title = re.sub(r'\s+', ' ', title)
        return title.strip()

    def _resolve_short_title_alias(self, normalized_title):
        if not normalized_title:
            return None
        token_num = len(normalized_title.split())
        if token_num == 0 or token_num > 5:
            return None

        prefix = normalized_title + ' '
        matched_id = None
        for title, pid in self.normalized_title_to_id.items():
            if title == normalized_title or title.startswith(prefix):
                if matched_id is not None:
                    return None
                matched_id = pid
        return matched_id

    # === 核心检索功能 ===
    def get_ids_from_query(self, query, num, shuffle=False):
        if not self.has_faiss: return []
        vector = self.embedding_model.encode(['search_query: ' + query], show_progress_bar=False)[0]
        return self.search_index(vector, num, self.abs_loaded_index)

    def get_ids_from_queries(self, queries, num, shuffle=False):
        if not self.has_faiss: return []
        vectors = self.embedding_model.encode(['search_query: ' + q for q in queries], show_progress_bar=False)
        return [self.search_index(vec, num, self.abs_loaded_index) for vec in vectors]

    def rank_papers_by_query(self, query, candidate_ids, num=20):
        if not candidate_ids:
            return []

        candidate_ids = list(dict.fromkeys([pid for pid in candidate_ids if pid in self.paper_index]))
        if not candidate_ids:
            return []

        query_vec = self.embedding_model.encode(['search_query: ' + query], show_progress_bar=False)[0]
        candidate_infos = self.get_paper_info_from_ids(candidate_ids)
        candidate_texts = []
        valid_ids = []
        for info in candidate_infos:
            if not info:
                continue
            valid_ids.append(info['id'])
            title = info.get('title', '')
            abs_text = info.get('abs', '')
            candidate_texts.append(f"{title}\n{abs_text}")

        if not valid_ids:
            return []

        doc_vecs = self.embedding_model.encode(
            ['search_document: ' + text for text in candidate_texts],
            show_progress_bar=False,
        )
        query_norm = np.linalg.norm(query_vec) + 1e-12
        doc_norms = np.linalg.norm(doc_vecs, axis=1) + 1e-12
        scores = np.dot(doc_vecs, query_vec) / (doc_norms * query_norm)
        order = np.argsort(-scores)[:num]
        return [valid_ids[idx] for idx in order]

    def search_index(self, vector, k, index):
        vector = np.array([vector]).astype('float32')
        distances, indices = index.search(vector, k)
        return [self.index_to_id[idx] for idx in indices[0] if idx != -1]
    
    def batch_search(self, query_vectors, top_k=1, title=False):
        if not self.has_faiss: return []
        query_vectors = np.array(query_vectors).astype('float32')
        if title:
            distances, indices = self.title_loaded_index.search(query_vectors, top_k)
        else:
            distances, indices = self.abs_loaded_index.search(query_vectors, top_k)
        results = []
        for i, query in enumerate(query_vectors):
            result = [(self.index_to_id[idx], distances[i][j]) for j, idx in enumerate(indices[i]) if idx != -1]
            results.append([_[0] for _ in result])
        return results

    # === [关键修复] 恢复 citation 搜索功能 ===
    def get_titles_from_citations(self, citations):
        if not self.has_faiss:
            return [""] * len(citations)

        flat_ids = ["Unknown"] * len(citations)
        unresolved_indices = []
        unresolved_citations = []

        for idx, citation in enumerate(citations):
            norm_title = self._normalize_title(citation)
            if norm_title in self.normalized_title_to_id:
                flat_ids[idx] = self.normalized_title_to_id[norm_title]
                continue

            alias_id = self._resolve_short_title_alias(norm_title)
            if alias_id:
                flat_ids[idx] = alias_id
            else:
                unresolved_indices.append(idx)
                unresolved_citations.append(citation)

        if unresolved_citations:
            q = self.get_embeddings_queries(unresolved_citations)
            batch_results = self.batch_search(q, 3, title=True)
            for idx, citation, res in zip(unresolved_indices, unresolved_citations, batch_results):
                if res:
                    citation_norm = self._normalize_title(citation)
                    chosen = None
                    for pid in res:
                        title = self.paper_index.get(pid, {}).get('title', '')
                        title_norm = self._normalize_title(title)
                        if title_norm == citation_norm or title_norm.startswith(citation_norm + ' '):
                            chosen = pid
                            break
                    flat_ids[idx] = chosen or res[0]
        return flat_ids

    def get_embeddings_documents(self, batch_text):
        # 对应 model.encode
        batch_text = ['search_document: ' + _ for _ in batch_text]
        return self.embedding_model.encode(batch_text, show_progress_bar=False)

    def get_embeddings_queries(self, batch_text):
        batch_text = ['search_query: ' + _ for _ in batch_text]
        return self.embedding_model.encode(batch_text, show_progress_bar=False)

    # === 数据获取 ===
    def get_paper_info_from_ids(self, ids):
        return [self.paper_index.get(pid) for pid in ids if pid in self.paper_index]

    def get_title_from_ids(self, ids):
        return [self.paper_index.get(pid, {}).get('title', 'Unknown') for pid in ids]
    
    def get_abs_from_ids(self, ids):
        return [self.paper_index.get(pid, {}).get('abs', '') for pid in ids]

    def get_paper_from_ids(self, ids, max_len=1500):
        h5_path = './paper_content.h5'
        h5_data = {}
        if os.path.exists(h5_path):
            try:
                with h5py.File(h5_path, 'r') as f:
                    for pid in ids:
                        if pid in f: h5_data[pid] = str(f[pid][()])
            except: pass
        results = []
        for pid in ids:
            content = h5_data.get(pid, self.paper_index.get(pid, {}).get('abs', ''))
            results.append(self.token_counter.text_truncation(content, max_len))
        return results

    def get_date_from_ids(self, ids): return []
