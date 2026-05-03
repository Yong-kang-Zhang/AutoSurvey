import re
import threading
import time
import copy

from src.model import APIModel
from src.utils import tokenCounter
from src.prompt import SUBSECTION_WRITING_PROMPT, LCE_PROMPT, CHECK_CITATION_PROMPT


class subsectionWriter():
    
    def __init__(self, model:str, api_key:str, api_url:str, database, image_api_key:str="") -> None:
        self.model, self.api_key, self.api_url = model, api_key, api_url
        self.image_api_key = image_api_key if image_api_key else api_key
        self.api_model = APIModel(self.model, self.api_key, self.api_url)
        self.db = database
        self.token_counter = tokenCounter()
        self.input_token_usage, self.output_token_usage = 0, 0

    def write(self, topic, outline, rag_num=30, subsection_len=500, refining=True, reflection=True, saving_path="./output/", illustrator_agent=None):
        parsed_outline = self.parse_outline(outline=outline)
        section_content = [[] for _ in range(len(parsed_outline['sections']))]

        section_paper_texts = [[] for _ in range(len(parsed_outline['sections']))]
        total_ids = []
        section_references_ids = [[] for _ in range(len(parsed_outline['sections']))]

        for i in range(len(parsed_outline['sections'])):
            descriptions = parsed_outline['subsection_descriptions'][i]
            for d in descriptions:
                references_ids = self.db.get_ids_from_query(d, num=rag_num, shuffle=False)
                total_ids += references_ids
                section_references_ids[i].append(references_ids)

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
                    str(subsection_len),
                ),
            )
            thread_l.append(thread)
            thread.start()
            time.sleep(0.1)
        for thread in thread_l:
            thread.join()

        raw_survey = self.generate_document(parsed_outline, section_content)
        raw_survey_with_references, raw_references = self.process_references(raw_survey)

        if refining:
            final_section_content = self.refine_subsections(topic, outline, section_content)
            if illustrator_agent:
                final_section_content = illustrator_agent.enrich_sections_with_diagrams(
                    topic,
                    parsed_outline,
                    final_section_content,
                    saving_path,
                )
            refined_survey = self.generate_document(parsed_outline, final_section_content)
            refined_survey_with_references, refined_references = self.process_references(refined_survey)
            return (
                raw_survey + '\n',
                raw_survey_with_references + '\n',
                raw_references,
                refined_survey + '\n',
                refined_survey_with_references + '\n',
                refined_references,
            )
        return raw_survey + '\n', raw_survey_with_references + '\n', raw_references

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

    def write_subsection_with_reflection(self, paper_texts_l, topic, outline, section, subsections, subdescriptions, res_l, idx, rag_num=20, subsection_len=1000, citation_num=8):
        prompts = []
        for j in range(len(subsections)):
            subsection = subsections[j]
            description = subdescriptions[j]
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
                    'CITATION NUM': str(citation_num),
                },
            )
            prompts.append(prompt)

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)
        contents = self.api_model.batch_chat(prompts, temperature=1)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(contents)
        contents = [c.replace('<format>', '').replace('</format>', '') for c in contents]

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
        contents = [c.replace('<format>', '').replace('</format>', '') for c in contents]

        res_l[idx] = contents
        return contents

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
        refined_content = self.api_model.chat(prompt, temperature=1).replace('<format>', '').replace('</format>', '')
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
        protected_survey, placeholders = self.protect_non_citation_blocks(survey)
        citations = self.extract_citations(protected_survey)
        updated_text, references = self.replace_citations_with_numbers(citations, protected_survey)
        return self.restore_non_citation_blocks(updated_text, placeholders), references

    def generate_document(self, parsed_outline, subsection_contents):
        document = []
        document.append(f"# {parsed_outline['title']}\n")

        for i, section in enumerate(parsed_outline['sections']):
            document.append(f"## {section}\n")
            for j, subsection in enumerate(parsed_outline['subsections'][i]):
                document.append(f"### {subsection}\n")
                if i < len(subsection_contents) and j < len(subsection_contents[i]):
                    document.append(subsection_contents[i][j] + "\n")

        return "\n".join(document)

    def extract_citations(self, markdown_text):
        pattern = re.compile(r'\[(.*?)\]')
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
                if citation in citation_to_ids:
                    citation_id = citation_to_ids[citation]
                    if citation_id in ids_to_titles:
                        numbered_citations.append(str(title_to_number[ids_to_titles[citation_id]]))
            return '[' + '; '.join(numbered_citations) + ']' if numbered_citations else match.group(0)

        updated_text = re.sub(r'\[(.*?)\]', replace_match, markdown_text)
        references_section = "\n\n## References\n\n"

        references = {num: titles_to_ids[title] for num, title in number_to_title_sorted.items()}
        for idx, title in number_to_title_sorted.items():
            t = title.replace('\n', '')
            references_section += f"[{idx}] {t}\n\n"

        return updated_text + references_section, references
