import json
import re
from dataclasses import dataclass, asdict

from src.model import APIModel


@dataclass
class RAGContext:
    topic: str
    keywords: list
    candidate_ids: list
    selected_ids: list
    keyword_to_ids: dict
    selected_target: int
    keep_strategy: str

    def to_dict(self):
        return asdict(self)


class TopicRAGPlanner:
    def __init__(
        self,
        model: str,
        api_key: str,
        api_url: str,
        database,
        keyword_num: int = 5,
        candidate_per_keyword: int = 12,
        keep_num: int = 0,
        filter_chunk_size: int = 12,
    ) -> None:
        self.api_model = APIModel(model, api_key, api_url)
        self.db = database
        self.keyword_num = keyword_num
        self.candidate_per_keyword = candidate_per_keyword
        self.keep_num = keep_num
        self.filter_chunk_size = filter_chunk_size

    def resolve_keep_num(self, candidate_ids):
        if self.keep_num and self.keep_num > 0:
            return min(self.keep_num, len(candidate_ids)) if candidate_ids else self.keep_num
        if not candidate_ids:
            return max(180, self.keyword_num * 18)
        auto_keep = max(180, int(len(candidate_ids) * 0.78))
        auto_keep = max(auto_keep, self.keyword_num * 18)
        return min(len(candidate_ids), auto_keep)

    def _generate_prompt(self, template, paras):
        prompt = template
        for k, v in paras.items():
            prompt = prompt.replace(f'[{k}]', v)
        return prompt

    def _parse_json_list(self, text):
        if not text:
            return []
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
            if isinstance(data, dict):
                for key in ('keywords', 'selected_ids', 'items'):
                    if key in data and isinstance(data[key], list):
                        return [str(x).strip() for x in data[key] if str(x).strip()]
        except Exception:
            pass

        bracket = re.search(r'\[(.*?)\]', text, re.DOTALL)
        if bracket:
            inner = bracket.group(1)
            items = [x.strip().strip('"').strip("'") for x in inner.split(',')]
            return [x for x in items if x]

        items = [line.strip("-* \t") for line in text.splitlines()]
        items = [x for x in items if x]
        return items

    def generate_keywords(self, topic):
        prompt = f"""
You are preparing a literature retrieval plan for an academic survey.
Topic: {topic}

Task:
1. Propose {self.keyword_num} concise search keywords or subtopics.
2. Each keyword should be short, specific, and useful for retrieval.
3. Avoid duplicates and overly broad wording.
4. Cover different literature axes when possible, such as core methods, architectures, training or inference strategies, evaluation settings, applications, and limitations.

Return strict JSON only in the form:
["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
"""
        response = self.api_model.chat(prompt, temperature=0.2)
        keywords = self._parse_json_list(response)
        keywords = [k for k in keywords if k]
        if not keywords:
            keywords = [topic]
        return keywords[:self.keyword_num]

    def retrieve_candidates(self, topic, keywords):
        keyword_to_ids = {}
        candidate_ids = []
        for keyword in keywords + [topic]:
            ids = self.db.get_ids_from_query(keyword, num=self.candidate_per_keyword, shuffle=False)
            keyword_to_ids[keyword] = ids
            candidate_ids.extend(ids)
        candidate_ids = list(dict.fromkeys(candidate_ids))
        return candidate_ids, keyword_to_ids

    def filter_candidates(self, topic, keywords, candidate_ids):
        if not candidate_ids:
            return [], self.resolve_keep_num(candidate_ids)

        keep_num = self.resolve_keep_num(candidate_ids)

        candidate_infos = self.db.get_paper_info_from_ids(candidate_ids)
        prompts = []
        chunks = []
        for start in range(0, len(candidate_infos), self.filter_chunk_size):
            chunk_infos = [info for info in candidate_infos[start:start + self.filter_chunk_size] if info]
            if not chunk_infos:
                continue
            chunks.append(chunk_infos)
            paper_text = []
            for idx, info in enumerate(chunk_infos, 1):
                abs_text = info.get('abs', '')
                abs_text = abs_text[:700]
                paper_text.append(
                    f"[ID={info['id']}]\nTitle: {info.get('title', '')}\nAbstract: {abs_text}\n"
                )
            prompt = f"""
You are screening candidate papers for an academic survey.
Topic: {topic}
Keywords: {", ".join(keywords)}

Select the most relevant papers from the candidate list below.
Keep papers that are central, complementary, and useful for building the survey.
Prefer a balanced pool covering seminal works, representative methods, strong empirical studies, benchmarks, applications, and challenge-oriented papers when they are relevant to the topic.
Avoid weakly related, duplicate, or tangential papers.
Return strict JSON only:
{{"selected_ids": ["id1", "id2"]}}

Candidate papers:
---
{chr(10).join(paper_text)}
---
"""
            prompts.append(prompt)

        responses = self.api_model.batch_chat(prompts, temperature=0.1)
        selected_ids = []
        for response, chunk_infos in zip(responses, chunks):
            ids = self._parse_json_list(response)
            if ids:
                valid = {info['id'] for info in chunk_infos}
                selected_ids.extend([pid for pid in ids if pid in valid])

        selected_ids = list(dict.fromkeys(selected_ids))
        ranked_fallback = self.db.rank_papers_by_query(
            topic + ' ' + ' '.join(keywords),
            candidate_ids,
            num=keep_num,
        )

        merged = []
        for pid in selected_ids + ranked_fallback:
            if pid not in merged:
                merged.append(pid)

        if not merged:
            merged = ranked_fallback or candidate_ids
        return merged[:keep_num], keep_num

    def build(self, topic):
        keywords = self.generate_keywords(topic)
        candidate_ids, keyword_to_ids = self.retrieve_candidates(topic, keywords)
        selected_ids, keep_num = self.filter_candidates(topic, keywords, candidate_ids)
        if not selected_ids:
            selected_ids = candidate_ids[:keep_num]
        return RAGContext(
            topic=topic,
            keywords=keywords,
            candidate_ids=candidate_ids,
            selected_ids=selected_ids,
            keyword_to_ids=keyword_to_ids,
            selected_target=keep_num,
            keep_strategy='fixed' if self.keep_num and self.keep_num > 0 else 'auto',
        )
