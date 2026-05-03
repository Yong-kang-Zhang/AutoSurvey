import os
import json
import argparse
import numpy as np
from tqdm import trange,tqdm
import threading
from src.model import APIModel
from src.utils import tokenCounter
from src.database import database
from src.agents.judge import Judge
from tqdm import tqdm
import time

def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output/', type=str, help='Directory containing the output survey')
    parser.add_argument('--model',default='gpt-4o-2024-05-13', type=str, help='Model for evaluation')
    parser.add_argument('--topic',default='', type=str, help='Topic of the survey')
    parser.add_argument('--api_url',default='https://api.openai.com/v1/chat/completions', type=str, help='url for API request')
    parser.add_argument('--api_key',default='', type=str, help='API key for the model')
    parser.add_argument('--db_path',default='./database', type=str, help='Directory of the database.')
    parser.add_argument('--embedding_model',default='nomic-ai/nomic-embed-text-v1', type=str, help='Embedding model for retrieval.')
    args = parser.parse_args()

    return args

def read_survey(path, topic):
    with open(f'{path}/{topic}.json', 'r') as f:
        dic = json.loads(f.read())
    return dic['survey'], dic['reference']


def map_five_to_hundred(score):
    try:
        return float(score) * 20.0
    except Exception:
        return score

def evaluate(args):

    db = database(db_path = args.db_path, embedding_model = args.embedding_model)

    if not os.path.exists(args.saving_path):
        os.mkdir(args.saving_path)

    judge = Judge(args.model, args.api_key, args.api_url, db)

    survey, references = read_survey(args.saving_path, args.topic)

    criterion = ['Coverage', 'Structure', 'Relevance']

    scores = judge.batch_criteria_based_judging(survey, args.topic, criterion)

    recall, precision = judge.citation_quality(survey, references)

    image_benchmark_summary = ""
    image_benchmark_score = None
    image_benchmark_path = os.path.join(args.saving_path, 'image_benchmark.txt')
    if os.path.exists(image_benchmark_path):
        with open(image_benchmark_path, 'r', encoding='utf-8') as f:
            image_benchmark_summary = f.read().strip()
        try:
            image_benchmark_score = json.loads(image_benchmark_summary).get("summary_score")
        except Exception:
            image_benchmark_score = None

    with open(f'{args.saving_path}/{args.topic}_evaluation.txt', 'w', encoding='utf-8') as f:
        result = f'Judged by {args.model}:\n'
        for c, s in zip(criterion, scores):
            mapped = map_five_to_hundred(s)
            result += f'{c} = {mapped:.1f} ({float(s):.1f}/5.0)\n'
        result += f'Citation Recall = {recall:.4f}\nCitation Precision = {precision:.4f}\n'
        if image_benchmark_score is not None:
            result += f'Image Benchmark Score = {image_benchmark_score:.2f}\n'
        if image_benchmark_summary:
            result += f'Image Benchmark = {image_benchmark_summary}\n'
        f.write(result)

if __name__ == '__main__':

    args = paras_args()

    evaluate(args)
