import copy
import json
import math
import os
import re
import threading
import time
from collections import Counter

from src.model import APIModel
from src.utils import tokenCounter
from src.prompt import (
    SUBSECTION_WRITING_PROMPT,
    LCE_PROMPT,
    CHECK_CITATION_PROMPT,
    CITATION_ENRICH_PROMPT,
    CITATION_REBALANCE_PROMPT,
    GLOBAL_CITATION_EXPANSION_PROMPT,
)


class subsectionWriter():
    
    def __init__(self, model:str, api_key:str, api_url:str, database, image_api_key:str="") -> None:
        self.model, self.api_key, self.api_url = model, api_key, api_url
        self.image_api_key = image_api_key if image_api_key else api_key
        self.api_model = APIModel(self.model, self.api_key, self.api_url)
        self.db = database
        self.token_counter = tokenCounter()
        self.input_token_usage, self.output_token_usage = 0, 0

    def clean_model_text(self, text, strip_tables=False):
        cleaned = (text or "").replace('<format>', '').replace('</format>', '').strip()
        cleaned = re.sub(r'^\s*```(?:markdown|md)?\s*\n', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n```+\s*$', '', cleaned)
        cleaned = cleaned.strip()
        if strip_tables:
            cleaned = self.remove_markdown_tables(cleaned)
        return cleaned

    def count_citation_spans(self, text):
        span_count = 0
        for match in re.finditer(r'\[([^\]]+)\]', text or ''):
            bracket = match.group(1).strip()
            if not bracket or bracket == 'Mechanism Diagram':
                continue
            span_count += 1
        return span_count

    def citation_stats(self, text):
        title_counts = self.collect_citation_titles(text)
        return {
            "span_count": self.count_citation_spans(text),
            "unique_title_count": len(title_counts),
            "max_repeat": max(title_counts.values()) if title_counts else 0,
            "title_counts": title_counts,
        }

    def _normalize_title(self, title):
        if hasattr(self.db, '_normalize_title'):
            return self.db._normalize_title(title)
        title = str(title or "").lower().strip()
        title = re.sub(r'\s+', ' ', title)
        return title

    def collect_cited_ids(self, section_content, title_map):
        normalized_title_to_id = {}
        for pid, title in title_map.items():
            norm_title = self._normalize_title(title)
            if norm_title and norm_title not in normalized_title_to_id:
                normalized_title_to_id[norm_title] = pid

        cited_id_counts = Counter()
        for subsections in section_content:
            for subsection_text in subsections:
                for title, count in self.collect_citation_titles(subsection_text).items():
                    pid = normalized_title_to_id.get(self._normalize_title(title))
                    if pid:
                        cited_id_counts[pid] += count
        return cited_id_counts

    def build_priority_title_list(self, reference_ids, title_map, max_titles=12):
        titles = []
        for pid in reference_ids:
            title = title_map.get(pid)
            if title and title not in titles:
                titles.append(title)
            if len(titles) >= max_titles:
                break
        return "\n".join(titles) if titles else "None"

    def write_runtime_checkpoint(self, topic, saving_path, stage, survey_text="", references=None, extra=None):
        if not saving_path:
            return

        os.makedirs(saving_path, exist_ok=True)
        preview_path = os.path.join(saving_path, "_running_preview.md")
        status_path = os.path.join(saving_path, "_running_status.json")

        if survey_text:
            with open(preview_path, "w", encoding="utf-8") as f:
                f.write(survey_text.strip() + "\n")

        payload = {
            "topic": topic,
            "stage": stage,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        if references is not None:
            payload["reference_count"] = len(references)
        if survey_text:
            stats = self.citation_stats(survey_text.split("## References")[0])
            payload["citation_spans"] = stats["span_count"]
            payload["unique_citation_titles"] = stats["unique_title_count"]
        if extra:
            payload.update(extra)

        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def compute_subsection_reference_plan(self, subsection_descriptions, allowed_ids, rag_num, global_used_ids=None):
        candidate_pool = list(allowed_ids) if allowed_ids else []
        plans = []
        used_ids = set(global_used_ids or set())
        if candidate_pool:
            total_subsections = max(1, len(subsection_descriptions))
            target_unique_budget = min(
                len(candidate_pool),
                max(math.ceil(rag_num * total_subsections * 1.7), rag_num + 180),
            )
            per_subsection_budget = max(rag_num + 12, math.ceil(target_unique_budget / total_subsections) + 6)
        else:
            per_subsection_budget = rag_num

        for idx, description in enumerate(subsection_descriptions):
            if candidate_pool:
                ranked_ids = self.db.rank_papers_by_query(
                    description,
                    candidate_pool,
                    num=max(per_subsection_budget * 5, rag_num * 4),
                )
                fresh_ids = [pid for pid in ranked_ids if pid not in used_ids]
                chosen_ids = fresh_ids[:per_subsection_budget]
                if len(chosen_ids) < per_subsection_budget:
                    for pid in ranked_ids:
                        if pid not in chosen_ids:
                            chosen_ids.append(pid)
                        if len(chosen_ids) >= per_subsection_budget:
                            break
                if idx == len(subsection_descriptions) - 1 and len(used_ids) < min(len(candidate_pool), 180):
                    remaining_ids = [pid for pid in ranked_ids if pid not in chosen_ids]
                    for pid in remaining_ids:
                        chosen_ids.append(pid)
                        if len(chosen_ids) >= max(per_subsection_budget + 10, rag_num + 12):
                            break
                chosen_ids = chosen_ids[:max(per_subsection_budget + 10, rag_num + 12)]
            else:
                chosen_ids = self.db.get_ids_from_query(description, num=per_subsection_budget, shuffle=False)

            used_ids.update(chosen_ids)
            plans.append(chosen_ids)
        return plans, used_ids

    def write(self, topic, outline, rag_num=30, subsection_len=500, refining=True, reflection=True, saving_path="./output/", illustrator_agent=None, rag_context=None):
        parsed_outline = self.parse_outline(outline=outline)
        section_content = [[] for _ in range(len(parsed_outline['sections']))]

        print(f"[*] Writing survey body for {len(parsed_outline['sections'])} sections...")

        section_paper_texts = [[] for _ in range(len(parsed_outline['sections']))]
        total_ids = []
        section_references_ids = [[] for _ in range(len(parsed_outline['sections']))]
        section_citation_targets = [[] for _ in range(len(parsed_outline['sections']))]
        section_unique_paper_targets = [[] for _ in range(len(parsed_outline['sections']))]
        allowed_ids = set(getattr(rag_context, 'selected_ids', []) or [])
        global_used_ids = set()

        for i in range(len(parsed_outline['sections'])):
            descriptions = parsed_outline['subsection_descriptions'][i]
            subsection_reference_plan, global_used_ids = self.compute_subsection_reference_plan(
                descriptions,
                allowed_ids,
                rag_num,
                global_used_ids=global_used_ids,
            )
            for d, references_ids in zip(descriptions, subsection_reference_plan):
                total_ids += references_ids
                section_references_ids[i].append(references_ids)
                citation_target = self.estimate_citation_target(subsection_len, len(references_ids))
                section_citation_targets[i].append(citation_target)
                section_unique_paper_targets[i].append(
                    self.estimate_unique_paper_target(len(references_ids), citation_target)
                )

        total_references_infos = self.db.get_paper_info_from_ids(list(set(total_ids)))
        temp_title_dic = {p['id']: p['title'] for p in total_references_infos}
        temp_abs_dic = {p['id']: p['abs'] for p in total_references_infos}

        for i in range(len(parsed_outline['sections'])):
            for references_ids in section_references_ids[i]:
                references_titles = [temp_title_dic[_] for _ in references_ids if _ in temp_title_dic]
                references_papers = [temp_abs_dic[_] for _ in references_ids if _ in temp_abs_dic]
                paper_texts = ''
                for t, p in zip(references_titles, references_papers):
                    paper_texts += f'---\n\npaper_title: {t}\n\npaper_content:\n\n{p}\n'
                paper_texts += '---\n'
                section_paper_texts[i].append(paper_texts)

        thread_l = []
        for i in range(len(parsed_outline['sections'])):
            thread = threading.Thread(
                target=self.write_subsection_with_reflection,
                args=(
                    section_paper_texts[i],
                    topic,
                    outline,
                    parsed_outline['sections'][i],
                    parsed_outline['subsections'][i],
                    parsed_outline['subsection_descriptions'][i],
                    section_content,
                    i,
                    rag_num,
                    subsection_len,
                    section_citation_targets[i],
                    section_unique_paper_targets[i],
                ),
            )
            thread_l.append(thread)
            thread.start()
            time.sleep(0.1)
        for thread in thread_l:
            thread.join()

        section_content = self.strip_uncontrolled_tables_from_sections(section_content)
        raw_survey = self.generate_document(parsed_outline, section_content)
        raw_survey_with_references, raw_references = self.process_references(raw_survey)
        self.write_runtime_checkpoint(
            topic,
            saving_path,
            "raw_draft_ready",
            raw_survey_with_references,
            raw_references,
            extra={"section_count": len(parsed_outline['sections'])},
        )

        if refining:
            print("[*] Refining subsection coherence...")
            final_section_content = self.refine_subsections(topic, outline, section_content)
            final_section_content = self.strip_uncontrolled_tables_from_sections(final_section_content)
            refined_stage_survey = self.generate_document(parsed_outline, final_section_content)
            refined_stage_with_refs, refined_stage_refs = self.process_references(refined_stage_survey)
            self.write_runtime_checkpoint(
                topic,
                saving_path,
                "coherence_refined",
                refined_stage_with_refs,
                refined_stage_refs,
            )

            print("[*] Rebalancing subsection citations...")
            final_section_content = self.rebalance_citations_after_refinement(
                topic,
                parsed_outline,
                final_section_content,
                section_references_ids,
                temp_title_dic,
                temp_abs_dic,
                section_citation_targets,
                section_unique_paper_targets,
            )
            rebalanced_survey = self.generate_document(parsed_outline, final_section_content)
            rebalanced_with_refs, rebalanced_refs = self.process_references(rebalanced_survey)
            self.write_runtime_checkpoint(
                topic,
                saving_path,
                "citation_rebalanced",
                rebalanced_with_refs,
                rebalanced_refs,
            )

            print("[*] Expanding citations only for weak subsections...")
            final_section_content = self.expand_global_citation_coverage(
                topic,
                parsed_outline,
                final_section_content,
                section_references_ids,
                temp_title_dic,
                temp_abs_dic,
                section_citation_targets,
                section_unique_paper_targets,
            )

            print("[*] Inserting controlled survey tables...")
            final_section_content = self.insert_section_tables(
                topic,
                parsed_outline,
                final_section_content,
                section_paper_texts,
            )
            table_ready_survey = self.generate_document(parsed_outline, final_section_content)
            table_ready_with_refs, table_ready_refs = self.process_references(table_ready_survey)
            self.write_runtime_checkpoint(
                topic,
                saving_path,
                "tables_inserted",
                table_ready_with_refs,
                table_ready_refs,
            )
            if illustrator_agent:
                print("[*] Generating and benchmarking figures...")
                final_section_content = illustrator_agent.enrich_sections_with_diagrams(
                    topic,
                    parsed_outline,
                    final_section_content,
                    saving_path,
                )
            refined_survey = self.generate_document(parsed_outline, final_section_content)
            refined_survey_with_references, refined_references = self.process_references(refined_survey)
            self.write_runtime_checkpoint(
                topic,
                saving_path,
                "final_refined",
                refined_survey_with_references,
                refined_references,
            )
            return (
                raw_survey + '\n',
                raw_survey_with_references + '\n',
                raw_references,
                refined_survey + '\n',
                refined_survey_with_references + '\n',
                refined_references,
            )
        return raw_survey + '\n', raw_survey_with_references + '\n', raw_references

    def strip_uncontrolled_tables_from_sections(self, section_content):
        cleaned = []
        for subsections in section_content:
            cleaned.append([self.remove_markdown_tables(content) for content in subsections])
        return cleaned

    def remove_markdown_tables(self, text):
        if not text:
            return text

        separator_pattern = re.compile(r'^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$')
        lines = text.splitlines()
        cleaned = []
        i = 0

        while i < len(lines):
            if i + 1 < len(lines) and '|' in lines[i] and separator_pattern.match(lines[i + 1].strip()):
                i += 2
                while i < len(lines) and lines[i].strip().startswith('|'):
                    i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if cleaned and cleaned[-1].strip():
                    cleaned.append('')
                continue

            cleaned.append(lines[i])
            i += 1

        return re.sub(r'\n{3,}', '\n\n', '\n'.join(cleaned)).strip()

    def insert_block_after_first_paragraph(self, text, block):
        block = (block or '').strip()
        body = (text or '').strip()
        if not block:
            return body
        if not body:
            return block

        parts = re.split(r'\n\s*\n', body, maxsplit=1)
        if len(parts) == 1:
            return body + "\n\n" + block
        return parts[0].strip() + "\n\n" + block + "\n\n" + parts[1].strip()

    def score_section_for_table(self, section_name, subsection_titles, subsection_contents):
        text = " ".join([section_name] + list(subsection_titles) + [c[:1200] for c in subsection_contents]).lower()
        weights = {
            'comparison': 4,
            'compare': 4,
            'benchmark': 4,
            'evaluation': 4,
            'dataset': 3,
            'architecture': 3,
            'method': 3,
            'approach': 3,
            'application': 2,
            'challenge': 2,
            'limitation': 2,
            'taxonomy': 3,
            'framework': 3,
            'pipeline': 3,
            'empirical': 3,
        }
        penalties = {
            'introduction': 4,
            'background': 1,
            'future': 5,
            'outlook': 5,
            'conclusion': 5,
        }

        score = 0
        for key, weight in weights.items():
            if key in text:
                score += weight
        lowered_name = section_name.lower()
        for key, weight in penalties.items():
            if key in lowered_name:
                score -= weight

        citation_density = len(re.findall(r'\[[^\]]+\]', " ".join(subsection_contents)))
        score += min(4, citation_density // 10)
        if len(subsection_contents) >= 3:
            score += 1
        return score

    def pick_table_anchor_subsection(self, subsection_titles, subsection_contents):
        best_idx = 0
        best_score = -10**9
        weights = {
            'comparison': 5,
            'compare': 5,
            'benchmark': 5,
            'evaluation': 4,
            'dataset': 4,
            'architecture': 3,
            'method': 3,
            'application': 2,
            'challenge': 2,
            'limitation': 2,
            'taxonomy': 3,
        }
        penalties = {'introduction': 4, 'future': 5, 'outlook': 5, 'conclusion': 5}

        for idx, (title, content) in enumerate(zip(subsection_titles, subsection_contents)):
            text = f"{title}\n{content[:1800]}".lower()
            score = 0
            for key, weight in weights.items():
                if key in text:
                    score += weight
            for key, weight in penalties.items():
                if key in title.lower():
                    score -= weight
            score += min(3, len(re.findall(r'\[[^\]]+\]', content)) // 6)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def refine_subsections(self, topic, outline, section_content):
        section_content_even = copy.deepcopy(section_content)

        thread_l = []
        for i in range(len(section_content)):
            for j in range(len(section_content[i])):
                if j % 2 == 0:
                    if j == 0:
                        contents = [''] + section_content[i][:2]
                    elif j == (len(section_content[i]) - 1):
                        contents = section_content[i][-2:] + ['']
                    else:
                        contents = section_content[i][j-1:j+2]
                    thread = threading.Thread(target=self.lce, args=(topic, outline, contents, section_content_even[i], j))
                    thread_l.append(thread)
                    thread.start()
        for thread in thread_l:
            thread.join()

        final_section_content = copy.deepcopy(section_content_even)

        thread_l = []
        for i in range(len(section_content_even)):
            for j in range(len(section_content_even[i])):
                if j % 2 == 1:
                    if j == (len(section_content_even[i]) - 1):
                        contents = section_content_even[i][-2:] + ['']
                    else:
                        contents = section_content_even[i][j-1:j+2]
                    thread = threading.Thread(target=self.lce, args=(topic, outline, contents, final_section_content[i], j))
                    thread_l.append(thread)
                    thread.start()
        for thread in thread_l:
            thread.join()

        return final_section_content

    def insert_section_tables(self, topic, parsed_outline, section_content, section_paper_texts):
        prompts = []
        prompt_meta = []
        max_tables = max(3, min(5, round(len(parsed_outline['sections']) * 0.5)))
        section_rankings = []

        for i, section_name in enumerate(parsed_outline['sections']):
            score = self.score_section_for_table(
                section_name,
                parsed_outline['subsections'][i],
                section_content[i],
            )
            section_rankings.append((score, i))

        candidate_indices = [
            idx for score, idx in sorted(section_rankings, reverse=True)
            if score > 0
        ][: min(len(parsed_outline['sections']), max_tables + 2)]

        for i in candidate_indices:
            section_name = parsed_outline['sections'][i]
            subsections = parsed_outline['subsections'][i]
            subsection_text = "\n\n".join(
                f"### {sub}\n{content}" for sub, content in zip(subsections, section_content[i])
            )
            paper_context = "\n".join(section_paper_texts[i][: min(3, len(section_paper_texts[i]))])
            if not subsection_text.strip() or not paper_context.strip():
                continue
            default_insert_idx = self.pick_table_anchor_subsection(subsections, section_content[i])
            prompt = f"""
You are revising an academic survey section about {topic}.
Section title: {section_name}

Current section text:
---
{subsection_text[:9000]}
---

Reference papers:
---
{paper_context[:9000]}
---

Task:
1. Produce at most one concise markdown table for this section, and only if a table clearly adds value.
2. Prefer survey-style "main comparison tables" rather than many small local tables.
3. The table should compare representative methods, datasets, mechanisms, capabilities, limitations, or evaluation settings that are genuinely discussed in the section.
4. Keep the table to 4-6 columns and 4-7 rows.
5. Each row should be content-dense and academically useful.
6. When a row contains a concrete claim about a method, benchmark, limitation, or dataset, include supporting citations in the relevant cell using [paper_title] format and only cite papers from the reference papers above.
7. Insert the table after the single best subsection for comparison or synthesis. The default best subsection index is {default_insert_idx + 1}, but you may choose a different one if the section text clearly suggests a better anchor.
8. Do not add any prose before or after the table.
9. If the section is mainly conceptual, introductory, or narrative and a table would feel redundant, return [NO_TABLE].

Return exactly one of the following formats:
[NO_TABLE]
or
[INSERT_AFTER_SUBSECTION]
{default_insert_idx + 1}
[/INSERT_AFTER_SUBSECTION]
[TABLE]
| Column A | Column B |
| --- | --- |
| ... | ... |
[/TABLE]
"""
            prompts.append(prompt)
            prompt_meta.append((i, default_insert_idx))

        if not prompts:
            return section_content

        tables = self.api_model.batch_chat(prompts, temperature=0.2)
        inserted = 0
        for (i, default_insert_idx), table_text in zip(prompt_meta, tables):
            if inserted >= max_tables:
                break
            cleaned = table_text.strip().replace('<format>', '').replace('</format>', '')
            if '[NO_TABLE]' in cleaned:
                continue
            table_match = re.search(r'\[TABLE\](.*?)\[/TABLE\]', cleaned, re.DOTALL | re.IGNORECASE)
            insert_match = re.search(r'\[INSERT_AFTER_SUBSECTION\](.*?)\[/INSERT_AFTER_SUBSECTION\]', cleaned, re.DOTALL | re.IGNORECASE)
            table_block = table_match.group(1).strip() if table_match else cleaned
            table_block = re.sub(r'^```(?:markdown)?\s*', '', table_block, flags=re.IGNORECASE)
            table_block = re.sub(r'\s*```$', '', table_block)
            if '|' not in table_block or table_block.count('|') < 6:
                continue

            insert_idx = default_insert_idx
            if insert_match:
                match = re.search(r'\d+', insert_match.group(1))
                if match:
                    insert_idx = int(match.group(0)) - 1

            if not section_content[i]:
                continue
            insert_idx = max(0, min(insert_idx, len(section_content[i]) - 1))
            section_content[i][insert_idx] = self.insert_block_after_first_paragraph(
                section_content[i][insert_idx],
                table_block,
            )
            inserted += 1
        return section_content

    def estimate_citation_target(self, subsection_len, available_papers):
        try:
            word_num = int(subsection_len)
        except Exception:
            word_num = 500
        soft_target = max(10, round(word_num / 42))
        if available_papers > 0:
            soft_target = min(soft_target, max(10, math.ceil(available_papers * 0.8)))
        return soft_target

    def estimate_unique_paper_target(self, available_papers, citation_target):
        if available_papers <= 0:
            return 0
        return min(available_papers, max(12, min(citation_target + 8, math.ceil(available_papers * 0.72))))

    def write_subsection_with_reflection(self, paper_texts_l, topic, outline, section, subsections, subdescriptions, res_l, idx, rag_num=20, subsection_len=1000, citation_targets=None, unique_paper_targets=None):
        prompts = []
        for j in range(len(subsections)):
            subsection = subsections[j]
            description = subdescriptions[j] if j < len(subdescriptions) else subsection
            citation_target = 6
            unique_paper_target = 8
            if citation_targets and j < len(citation_targets):
                citation_target = citation_targets[j]
            if unique_paper_targets and j < len(unique_paper_targets):
                unique_paper_target = unique_paper_targets[j]
            prompt = self.__generate_prompt(
                SUBSECTION_WRITING_PROMPT,
                paras={
                    'OVERALL OUTLINE': outline,
                    'SUBSECTION NAME': subsection,
                    'DESCRIPTION': description,
                    'TOPIC': topic,
                    'PAPER LIST': paper_texts_l[j],
                    'SECTION NAME': section,
                    'WORD NUM': str(subsection_len),
                    'CITATION NUM': str(citation_target),
                    'UNIQUE PAPER NUM': str(unique_paper_target),
                },
            )
            prompts.append(prompt)

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)
        contents = self.api_model.batch_chat(prompts, temperature=1)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(contents)
        contents = [
            self.clean_model_text(c, strip_tables=True)
            for c in contents
        ]

        enrich_prompts = []
        for j, (content, paper_texts) in enumerate(zip(contents, paper_texts_l)):
            citation_target = 6
            unique_paper_target = 8
            if citation_targets and j < len(citation_targets):
                citation_target = citation_targets[j] + 5
            if unique_paper_targets and j < len(unique_paper_targets):
                unique_paper_target = unique_paper_targets[j] + 3
            enrich_prompts.append(
                self.__generate_prompt(
                    CITATION_ENRICH_PROMPT,
                    paras={
                        'SUBSECTION': content,
                        'TOPIC': topic,
                        'PAPER LIST': paper_texts,
                        'CITATION NUM': str(citation_target),
                        'UNIQUE CITATION NUM': str(unique_paper_target + 2),
                    },
                )
            )
        self.input_token_usage += self.token_counter.num_tokens_from_list_string(enrich_prompts)
        contents = self.api_model.batch_chat(enrich_prompts, temperature=0.3)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(contents)
        contents = [
            self.clean_model_text(c, strip_tables=True)
            for c in contents
        ]

        prompts = []
        for content, paper_texts in zip(contents, paper_texts_l):
            prompts.append(
                self.__generate_prompt(
                    CHECK_CITATION_PROMPT,
                    paras={'SUBSECTION': content, 'TOPIC': topic, 'PAPER LIST': paper_texts},
                )
            )
        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)
        contents = self.api_model.batch_chat(prompts, temperature=1)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(contents)
        contents = [
            self.clean_model_text(c, strip_tables=True)
            for c in contents
        ]

        res_l[idx] = contents
        return contents

    def flatten_unique_ids(self, nested_ids):
        merged = []
        for ids in nested_ids:
            for pid in ids:
                if pid not in merged:
                    merged.append(pid)
        return merged

    def collect_citation_titles(self, text):
        counts = Counter()
        for match in re.finditer(r'\[([^\]]+)\]', text or ''):
            bracket = match.group(1).strip()
            if not bracket or re.fullmatch(r'[\d\s;,]+', bracket):
                continue
            for title in bracket.split(';'):
                cleaned = title.strip()
                if cleaned and cleaned != 'Mechanism Diagram':
                    counts[cleaned] += 1
        return counts

    def build_paper_text(self, reference_ids, title_map, abs_map, max_papers=None, max_abs_chars=900):
        paper_texts = []
        dedup_ids = []
        for pid in reference_ids:
            if pid in title_map and pid in abs_map and pid not in dedup_ids:
                dedup_ids.append(pid)
        if max_papers is not None:
            dedup_ids = dedup_ids[:max_papers]

        for pid in dedup_ids:
            title = title_map[pid]
            abs_text = (abs_map[pid] or '')[:max_abs_chars]
            paper_texts.append(
                f"---\n\npaper_title: {title}\n\npaper_content:\n\n{abs_text}\n"
            )
        if not paper_texts:
            return "---\n"
        return "".join(paper_texts) + "---\n"

    def select_rebalance_reference_ids(self, description, local_ids, section_ids, max_refs=68):
        local_ids = [pid for pid in local_ids if pid]
        section_ids = [pid for pid in section_ids if pid]
        if not section_ids:
            return local_ids[:max_refs]

        ranked_ids = self.db.rank_papers_by_query(
            description,
            section_ids,
            num=min(len(section_ids), max(max_refs * 3, len(local_ids) * 4, 96)),
        )
        expanded_ids = list(local_ids)
        for pid in ranked_ids:
            if pid not in expanded_ids:
                expanded_ids.append(pid)
            if len(expanded_ids) >= max_refs:
                break
        return expanded_ids[:max_refs]

    def rebalance_citations_after_refinement(
        self,
        topic,
        parsed_outline,
        section_content,
        section_references_ids,
        title_map,
        abs_map,
        section_citation_targets,
        section_unique_paper_targets,
    ):
        prompts = []
        prompt_meta = []
        validation_prompts = []

        for i, section_name in enumerate(parsed_outline['sections']):
            section_ids = self.flatten_unique_ids(section_references_ids[i])
            subsection_titles = parsed_outline['subsections'][i]
            subsection_descriptions = parsed_outline['subsection_descriptions'][i]
            for j, subsection_text in enumerate(section_content[i]):
                description = subsection_descriptions[j] if j < len(subsection_descriptions) else subsection_titles[j]
                local_ids = section_references_ids[i][j]
                rebalance_ids = self.select_rebalance_reference_ids(
                    f"{section_name}. {subsection_titles[j]}. {description}",
                    local_ids,
                    section_ids,
                    max_refs=max(58, min(82, len(section_ids))),
                )
                paper_text = self.build_paper_text(rebalance_ids, title_map, abs_map, max_papers=72)
                overused_titles = self.collect_citation_titles(subsection_text)
                overused = [
                    f"{title} ({count} times)"
                    for title, count in overused_titles.most_common(8)
                    if count >= 3
                ]
                if not overused:
                    overused = ["None detected; still diversify where valid."]

                citation_target = section_citation_targets[i][j] if j < len(section_citation_targets[i]) else 10
                unique_target = section_unique_paper_targets[i][j] if j < len(section_unique_paper_targets[i]) else 12
                prompt = self.__generate_prompt(
                    CITATION_REBALANCE_PROMPT,
                    paras={
                        'TOPIC': topic,
                        'PAPER LIST': paper_text,
                        'SUBSECTION': subsection_text,
                        'DESCRIPTION': description,
                        'OVERUSED TITLES': "\n".join(overused),
                        'CITATION NUM': str(citation_target + 4),
                        'UNIQUE CITATION NUM': str(unique_target + 4),
                    },
                )
                prompts.append(prompt)
                prompt_meta.append((i, j, paper_text))

        if not prompts:
            return section_content

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)
        revised_contents = self.api_model.batch_chat(prompts, temperature=0.25)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(revised_contents)
        cleaned_contents = [
            self.remove_markdown_tables(c.replace('<format>', '').replace('</format>', ''))
            for c in revised_contents
        ]

        for (i, j, paper_text), content in zip(prompt_meta, cleaned_contents):
            validation_prompts.append(
                self.__generate_prompt(
                    CHECK_CITATION_PROMPT,
                    paras={'SUBSECTION': content, 'TOPIC': topic, 'PAPER LIST': paper_text},
                )
            )

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(validation_prompts)
        validated_contents = self.api_model.batch_chat(validation_prompts, temperature=0.6)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(validated_contents)
        validated_contents = [
            self.remove_markdown_tables(c.replace('<format>', '').replace('</format>', ''))
            for c in validated_contents
        ]

        for (i, j, _), content in zip(prompt_meta, validated_contents):
            section_content[i][j] = content
        return section_content

    def expand_global_citation_coverage(
        self,
        topic,
        parsed_outline,
        section_content,
        section_references_ids,
        title_map,
        abs_map,
        section_citation_targets,
        section_unique_paper_targets,
    ):
        prompts = []
        prompt_meta = []
        validation_prompts = []
        global_cited_ids = self.collect_cited_ids(section_content, title_map)
        all_reference_ids = []
        for section_ids in section_references_ids:
            for subsection_ids in section_ids:
                for pid in subsection_ids:
                    if pid not in all_reference_ids:
                        all_reference_ids.append(pid)

        globally_underused_ids = [pid for pid in all_reference_ids if global_cited_ids.get(pid, 0) == 0]

        weak_targets = []
        for i, section_name in enumerate(parsed_outline['sections']):
            subsection_titles = parsed_outline['subsections'][i]
            subsection_descriptions = parsed_outline['subsection_descriptions'][i]
            for j, subsection_text in enumerate(section_content[i]):
                stats = self.citation_stats(subsection_text)
                citation_target = section_citation_targets[i][j] if j < len(section_citation_targets[i]) else 10
                unique_target = section_unique_paper_targets[i][j] if j < len(section_unique_paper_targets[i]) else 12
                citation_gap = max(0, citation_target - stats["span_count"])
                unique_gap = max(0, unique_target - stats["unique_title_count"])
                repeated_penalty = max(0, stats["max_repeat"] - 3)
                if citation_gap <= 1 and unique_gap <= 2 and repeated_penalty <= 1:
                    continue
                weak_targets.append(
                    (
                        citation_gap * 3 + unique_gap * 2 + repeated_penalty,
                        i,
                        j,
                        subsection_titles[j],
                        subsection_descriptions[j] if j < len(subsection_descriptions) else subsection_titles[j],
                        stats,
                    )
                )

        if not weak_targets:
            return section_content

        weak_targets = sorted(weak_targets, reverse=True)
        max_targets = max(8, min(16, len(weak_targets)))
        weak_targets = weak_targets[:max_targets]

        for _, i, j, subsection_title, description, stats in weak_targets:
            section_name = parsed_outline['sections'][i]
            subsection_text = section_content[i][j]
            section_ids = self.flatten_unique_ids(section_references_ids[i])
            local_ids = list(section_references_ids[i][j])
            expansion_seed_ids = []
            for pid in globally_underused_ids:
                if pid in section_ids and pid not in expansion_seed_ids:
                    expansion_seed_ids.append(pid)
                if len(expansion_seed_ids) >= 18:
                    break

            expanded_ids = self.select_rebalance_reference_ids(
                f"{topic}. {section_name}. {subsection_title}. {description}",
                local_ids + expansion_seed_ids,
                section_ids,
                max_refs=max(64, min(88, len(section_ids))),
            )
            paper_text = self.build_paper_text(expanded_ids, title_map, abs_map, max_papers=56, max_abs_chars=720)
            prompt = self.__generate_prompt(
                GLOBAL_CITATION_EXPANSION_PROMPT,
                paras={
                    'TOPIC': topic,
                    'PAPER LIST': paper_text,
                    'SUBSECTION': subsection_text,
                    'DESCRIPTION': description,
                    'CURRENT CITATION NUM': str(stats["span_count"]),
                    'CURRENT UNIQUE CITATION NUM': str(stats["unique_title_count"]),
                    'TARGET CITATION NUM': str(section_citation_targets[i][j] + 3 if j < len(section_citation_targets[i]) else 13),
                    'TARGET UNIQUE CITATION NUM': str(section_unique_paper_targets[i][j] + 3 if j < len(section_unique_paper_targets[i]) else 15),
                    'PRIORITY PAPERS': self.build_priority_title_list(expansion_seed_ids, title_map, max_titles=10),
                },
            )
            prompts.append(prompt)
            prompt_meta.append((i, j, paper_text))

        if not prompts:
            return section_content

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)
        revised_contents = self.api_model.batch_chat(prompts, temperature=0.2)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(revised_contents)
        cleaned_contents = [
            self.clean_model_text(c, strip_tables=True)
            for c in revised_contents
        ]

        for (i, j, paper_text), content in zip(prompt_meta, cleaned_contents):
            validation_prompts.append(
                self.__generate_prompt(
                    CHECK_CITATION_PROMPT,
                    paras={'SUBSECTION': content, 'TOPIC': topic, 'PAPER LIST': paper_text},
                )
            )

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(validation_prompts)
        validated_contents = self.api_model.batch_chat(validation_prompts, temperature=0.55)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(validated_contents)
        validated_contents = [
            self.clean_model_text(c, strip_tables=True)
            for c in validated_contents
        ]

        for (i, j, _), content in zip(prompt_meta, validated_contents):
            section_content[i][j] = content
        return section_content

    def __generate_prompt(self, template, paras):
        prompt = template
        for k in paras.keys():
            prompt = prompt.replace(f'[{k}]', paras[k])
        return prompt

    def lce(self, topic, outline, contents, res_l, idx):
        prompt = self.__generate_prompt(
            LCE_PROMPT,
            paras={
                'OVERALL OUTLINE': outline,
                'PREVIOUS': contents[0],
                'FOLLOWING': contents[2],
                'TOPIC': topic,
                'SUBSECTION': contents[1],
            },
        )
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
        refined_content = self.clean_model_text(self.api_model.chat(prompt, temperature=1), strip_tables=False)
        self.output_token_usage += self.token_counter.num_tokens_from_string(refined_content)
        res_l[idx] = refined_content
        return refined_content.replace('Here is the refined subsection:\n', '')

    def parse_outline(self, outline):
        result = {
            "title": "",
            "sections": [],
            "section_descriptions": [],
            "subsections": [],
            "subsection_descriptions": [],
        }

        lines = outline.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('# '):
                result["title"] = line[2:].strip()
            elif line.startswith('## '):
                result["sections"].append(line[3:].strip())
                if i + 1 < len(lines) and lines[i + 1].startswith('Description:'):
                    result["section_descriptions"].append(lines[i + 1].split('Description:', 1)[1].strip())
                    result["subsections"].append([])
                    result["subsection_descriptions"].append([])
            elif line.startswith('### '):
                if result["subsections"]:
                    result["subsections"][-1].append(line[4:].strip())
                    if i + 1 < len(lines) and lines[i + 1].startswith('Description:'):
                        result["subsection_descriptions"][-1].append(lines[i + 1].split('Description:', 1)[1].strip())

        return result

    def process_references(self, survey):
        survey = self.unwrap_markdown_fences(survey)
        protected_survey, placeholders = self.protect_non_citation_blocks(survey)
        citations = self.extract_citations(protected_survey)
        updated_text, references = self.replace_citations_with_numbers(citations, protected_survey)
        return self.restore_non_citation_blocks(updated_text, placeholders), references

    def unwrap_markdown_fences(self, text):
        if not text:
            return text

        def repl(match):
            content = match.group(1).strip('\n')
            if any(token in content for token in ('## ', '### ', '|', '![')):
                return match.group(0)
            return content

        text = re.sub(r'```markdown\s*\n(.*?)\n```', repl, text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'```\s*\n(.*?)\n```', repl, text, flags=re.DOTALL)
        return text

    def strip_internal_headings(self, text):
        cleaned_lines = []
        for line in text.splitlines():
            if re.match(r'^\s*#{2,6}\s+', line):
                continue
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines).strip()

    def generate_document(self, parsed_outline, subsection_contents):
        document = []
        document.append(f"# {parsed_outline['title']}\n")

        for i, section in enumerate(parsed_outline['sections']):
            document.append(f"## {section}\n")
            for j, subsection in enumerate(parsed_outline['subsections'][i]):
                document.append(f"### {subsection}\n")
                if i < len(subsection_contents) and j < len(subsection_contents[i]):
                    document.append(self.strip_internal_headings(subsection_contents[i][j]) + "\n")

        return "\n".join(document)

    def extract_citations(self, markdown_text):
        pattern = re.compile(r'\[(.*?)\]', re.DOTALL)
        matches = pattern.findall(markdown_text)
        citations = []
        for match in matches:
            parts = match.split(';')
            for part in parts:
                cit = part.strip()
                if cit not in citations:
                    citations.append(cit)
        return citations

    def protect_non_citation_blocks(self, text):
        placeholders = {}

        def repl(match):
            key = f"__PLACEHOLDER_{len(placeholders)}__"
            placeholders[key] = match.group(0)
            return key

        text = re.sub(r'```.*?```', repl, text, flags=re.DOTALL)
        text = re.sub(r'!\[.*?\]\(.*?\)', repl, text)
        text = re.sub(r'\[.*?\]\(.*?\)', repl, text)
        return text, placeholders

    def restore_non_citation_blocks(self, text, placeholders):
        for key, value in placeholders.items():
            text = text.replace(key, value)
        return text

    def replace_citations_with_numbers(self, citations, markdown_text):
        ids = self.db.get_titles_from_citations(citations)
        citation_to_ids = {citation: idx for citation, idx in zip(citations, ids)}

        paper_infos = self.db.get_paper_info_from_ids(ids)
        temp_dic = {p['id']: p['title'] for p in paper_infos}
        ids_to_titles = {idx: temp_dic[idx] for idx in ids if idx in temp_dic}
        titles_to_ids = {title: idx for idx, title in ids_to_titles.items()}

        titles = list(dict.fromkeys(ids_to_titles.values()))
        title_to_number = {title: num + 1 for num, title in enumerate(titles)}
        number_to_title = {num: title for title, num in title_to_number.items()}
        number_to_title_sorted = {key: number_to_title[key] for key in sorted(number_to_title)}

        def replace_match(match):
            citation_text = match.group(1)
            individual_citations = citation_text.split(';')
            numbered_citations = []
            for citation in individual_citations:
                citation = citation.strip()
                if not citation:
                    continue

                citation_id = citation_to_ids.get(citation)
                if citation_id is None:
                    normalized_citation = re.sub(r'\s+', ' ', citation.replace('\n', ' ')).strip()
                    citation_id = citation_to_ids.get(normalized_citation)
                if citation_id is None:
                    for known_citation, known_id in citation_to_ids.items():
                        normalized_known = re.sub(r'\s+', ' ', known_citation.replace('\n', ' ')).strip()
                        if normalized_known == re.sub(r'\s+', ' ', citation.replace('\n', ' ')).strip():
                            citation_id = known_id
                            break
                if citation_id is not None:
                    if citation_id in ids_to_titles:
                        numbered_citations.append(str(title_to_number[ids_to_titles[citation_id]]))
            return '[' + '; '.join(numbered_citations) + ']' if numbered_citations else match.group(0)

        updated_text = re.sub(r'\[(.*?)\]', replace_match, markdown_text, flags=re.DOTALL)
        references_section = "\n\n## References\n\n"

        references = {num: titles_to_ids[title] for num, title in number_to_title_sorted.items()}
        for idx, title in number_to_title_sorted.items():
            t = title.replace('\n', '')
            references_section += f"[{idx}] {t}\n\n"

        return updated_text + references_section, references
