import os
import json
import argparse
import re

from src.agents.outline_writer import outlineWriter
from src.agents.writer import subsectionWriter
from src.agents.illustrator import IllustratorAgent
from src.database import database
from src.model import APIModel
from src.latex_converter import MD2LatexConverter
from src.rag_planner import TopicRAGPlanner
from src.image_benchmark import ImageBenchmarkAgent


def remove_descriptions(text):
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not line.strip().startswith("Description")]
    return '\n'.join(filtered_lines)


def clean_generated_survey(text):
    cleaned_lines = []
    in_mermaid_block = False

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        if stripped.startswith("```mermaid"):
            in_mermaid_block = True
            continue

        if in_mermaid_block:
            if stripped == "```":
                in_mermaid_block = False
            continue

        if stripped.startswith(">"):
            continue

        if re.match(r'^\*\(View\s+\d+:.*\)\*$', stripped):
            continue

        cleaned_lines.append(raw_line.rstrip())

    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    return cleaned_text.strip() + '\n'


def write_outline(topic, model, section_num, outline_reference_num, db, api_key, api_url, rag_context=None):
    outline_writer = outlineWriter(model=model, api_key=api_key, api_url=api_url, database=db)
    outline = outline_writer.draft_outline(topic, outline_reference_num, 30000, section_num, rag_context=rag_context)
    return outline, remove_descriptions(outline)


def write_subsection(
    topic,
    model,
    outline,
    subsection_len,
    rag_num,
    db,
    api_key,
    api_url,
    refinement=True,
    saving_path="./output/",
    image_api_key="",
    image_model="",
    image_api_url="",
    rag_context=None,
    enable_image_benchmark=True,
):
    illustrator = None
    if image_api_key:
        api_model = APIModel(model=model, api_key=api_key, api_url=api_url)
        benchmark_agent = ImageBenchmarkAgent(api_model) if enable_image_benchmark else None
        illustrator = IllustratorAgent(
            api_model,
            image_api_key=image_api_key,
            image_model=image_model,
            image_api_url=image_api_url,
            benchmark_agent=benchmark_agent,
        )

    subsection_writer = subsectionWriter(model=model, api_key=api_key, api_url=api_url, database=db)
    if refinement:
        return subsection_writer.write(
            topic,
            outline,
            subsection_len=subsection_len,
            rag_num=rag_num,
            refining=True,
            saving_path=saving_path,
            illustrator_agent=illustrator,
            rag_context=rag_context,
        )
    return subsection_writer.write(
        topic,
        outline,
        subsection_len=subsection_len,
        rag_num=rag_num,
        refining=False,
        rag_context=rag_context,
    )


def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path', default='./output/', type=str, help='Directory to save the output survey')
    parser.add_argument('--model', default='gpt-4o-2024-05-13', type=str, help='Model to use')
    parser.add_argument('--topic', default='', type=str, help='Topic to generate survey for')
    parser.add_argument('--topic_txt', default='', type=str, help='Text file containing the topic to generate survey for')
    parser.add_argument('--section_num', default=7, type=int, help='Number of sections in the outline')
    parser.add_argument('--subsection_len', default=320, type=int, help='Length of each subsection')
    parser.add_argument('--outline_reference_num', default=2000, type=int, help='Number of references for outline generation')
    parser.add_argument('--rag_num', default=30, type=int, help='Number of references passed to each subsection writer')
    parser.add_argument('--keyword_num', default=12, type=int, help='Number of keywords for topic decomposition')
    parser.add_argument('--candidate_per_keyword', default=40, type=int, help='Number of candidates retrieved per keyword')
    parser.add_argument('--rag_keep_num', default=0, type=int, help='Number of references kept after candidate filtering; 0 means auto')
    parser.add_argument('--rag_filter_chunk_size', default=12, type=int, help='Chunk size for LLM-based candidate filtering')
    parser.add_argument('--api_url', default='https://api.openai.com/v1/chat/completions', type=str, help='url for API request')
    parser.add_argument('--api_key', default='', type=str, help='API key for the model')
    parser.add_argument('--image_api_key', default='', type=str, help='API key for the image model')
    parser.add_argument('--image_model', default='gpt-image-2-all', type=str, help='Image model to use')
    parser.add_argument('--image_api_url', default='', type=str, help='Image API endpoint; defaults based on image model')
    parser.add_argument('--disable_image_benchmark', action='store_true', help='Disable image suitability benchmark')
    parser.add_argument('--db_path', default='./database', type=str, help='Directory of the database.')
    parser.add_argument('--embedding_model', default='nomic-ai/nomic-embed-text-v1', type=str, help='Embedding model for retrieval.')
    args = parser.parse_args()
    return args


def resolve_topic(args):
    if args.topic_txt:
        with open(args.topic_txt, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return args.topic.strip()


def main(args):
    topic = resolve_topic(args)
    if not topic:
        raise ValueError("A topic is required. Please provide --topic or --topic_txt.")

    db = database(db_path=args.db_path, embedding_model=args.embedding_model)

    rag_planner = TopicRAGPlanner(
        model=args.model,
        api_key=args.api_key,
        api_url=args.api_url,
        database=db,
        keyword_num=args.keyword_num,
        candidate_per_keyword=args.candidate_per_keyword,
        keep_num=args.rag_keep_num,
        filter_chunk_size=args.rag_filter_chunk_size,
    )
    rag_context = rag_planner.build(topic)

    if not os.path.exists(args.saving_path):
        os.makedirs(args.saving_path)

    outline_with_description, _ = write_outline(
        topic,
        args.model,
        args.section_num,
        args.outline_reference_num,
        db,
        args.api_key,
        args.api_url,
        rag_context=rag_context,
    )

    raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = write_subsection(
        topic,
        args.model,
        outline_with_description,
        args.subsection_len,
        args.rag_num,
        db,
        args.api_key,
        args.api_url,
        refinement=True,
        saving_path=args.saving_path,
        image_api_key=args.image_api_key,
        image_model=args.image_model,
        image_api_url=args.image_api_url,
        rag_context=rag_context,
        enable_image_benchmark=not args.disable_image_benchmark,
    )

    refined_survey_with_references = clean_generated_survey(refined_survey_with_references)

    md_filename = f'{args.saving_path}/{topic}.md'
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(refined_survey_with_references)

    json_filename = f'{args.saving_path}/{topic}.json'
    save_dic = {
        'survey': refined_survey_with_references,
        'reference': refined_references,
        'rag_plan': rag_context.to_dict(),
    }
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(save_dic, f, indent=4, ensure_ascii=False)

    rag_filename = f'{args.saving_path}/rag_plan.json'
    with open(rag_filename, 'w', encoding='utf-8') as f:
        json.dump(rag_context.to_dict(), f, indent=4, ensure_ascii=False)

    print(f"[*] Done! Files saved to {md_filename}, {json_filename}, and {rag_filename}")

    print("[*] Converting Survey to LaTeX format...")
    latex_converter = MD2LatexConverter(md_filename)
    latex_converter.convert()


if __name__ == '__main__':
    args = paras_args()
    main(args)
