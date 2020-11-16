"""
This script creates monoT5 input files by taking an index,
queries and the retrieval run file for the queries and then
creating files for monoT5 input. Each line in the monoT5 input
file follows the format:
    f'Query: {query} Document: {document} Relevant:\n')
"""
import collections
import argparse
import json
import logging
import spacy

from pyserini.index import IndexReader
from tqdm import tqdm
from pathlib import Path
from os import path

def load_queries(path: str):
    """
    Loads queries into a dictionary of query_id -> (query, question)
    """
    queries = collections.OrderedDict()
    with open(path) as f:
        raw_json = f.read()
        parsed_json = json.loads(raw_json)
        for topic in parsed_json:
            # Question ID is the integer question ID prefixed with "CQ"
            # or "EQ" depending on whether it's a consumer or expert question.
            # This has to be parsed out so it can be joined against the run
            # file.
            question_id_string = topic["question_id"]
            # The prefix is of length 2.
            question_id = int(question_id_string[2:])
            question = topic["question"]
            query = topic["query"]
            if args.use_question:
                queries[question_id] = question
            else:
                queries[question_id] = query

    return queries


def load_run(path):
    """Loads run into a dict of key: query_id, value: list of candidate doc
    ids."""

    print('Loading run...')
    run = collections.OrderedDict()
    with open(path) as f:
        for line in tqdm(f):
            query_id, _, doc_id, rank, _, _ = line.split()
            query_id = int(query_id)
            if query_id not in run:
                run[query_id] = []
            run[query_id].append((doc_id, int(rank)))

    # Sort candidate docs by rank.
    sorted_run = collections.OrderedDict()
    for query_id, doc_ids_ranks in run.items():
        sorted(doc_ids_ranks, key=lambda x: x[1])
        doc_ids = [doc_id for doc_id, _ in doc_ids_ranks]
        sorted_run[query_id] = doc_ids

    return sorted_run

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, required=True,
                        help="tsv file with two columns, <query_id> and <query_text>")
    parser.add_argument("--run", type=str, required=True,
                        help="tsv file with three columns <query_id>, <doc_id> and <rank>")
    parser.add_argument('--index', required=True, default='',
                        help='index path')
    parser.add_argument("--t5_input", type=str, required=True,
                        help="path to store t5_input, txt format")
    parser.add_argument("--t5_input_ids", type=str, required=True,
                        help="path to store the query-doc ids of t5_input, tsv format")
    parser.add_argument('--use_question_and_query', action='store_true',
                        help="If true, the t5 input query is a concatenation of \
                            question and query.  Otherwise, it's just the question.")
    parser.add_argument('--stride', type=int, default=4, help='')
    parser.add_argument('--max_length', type=int, default=8, help='')
    
    args = parser.parse_args()

    queries = load_queries(path=args.queries)
    run = load_run(path=args.run)
    
    logging.info(args)
    nlp = spacy.blank("en")
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    index_utils = IndexReader(args.index)

    # Metrics to report at end
    num_docs = 0
    num_segments = 0
    num_no_segments = 0
    num_no_content = 0

    print("Writing t5 input and ids")
    with open(args.t5_input, 'w') as fout_t5_texts, open(args.t5_input_ids, 'w') as fout_t5_ids:
        for num_examples, (query_id, candidate_doc_ids) in enumerate(
                tqdm(run.items(), total=len(run))):
            query = queries[query_id]
            seen_doc_ids = set()
            for candidate_doc_id in candidate_doc_ids:
                if candidate_doc_id in seen_doc_ids:
                    continue # Should this ever happen?
                contents = index_utils.doc_contents(candidate_doc_id)
                if not contents:
                    print(f'Doc id not found: {candidate_doc_id}')
                    num_no_content += 1
                    continue
                sections = contents.split('\n')
                if len(sections) == 0:
                    num_no_content += 1
                    continue
                doc_title = sections[0] # TODO Check title is in content.
                num_docs += 1
                if len(sections) < 2:
                    num_no_content += 1
                    continue
                
                # Get passage text excluding title.
                passage_text = ' '.join(sections[1:])
                # Remove duplicate spaces and line breaks.
                passage_text = ' '.join(passage_text.split())
                
                # TODO Why do we only consider the first 10K chars?  May
                # have to change this: [:10000]
                doc = nlp(passage_text)
                sentences = [sent.string.strip() for sent in doc.sents]

                if not sentences:
                    num_no_segments += 1
                    sentences = ['']
                
                for i in range(0, len(sentences), args.stride):
                    segment = ' '.join(
                        sentences[i:(i + args.max_length)]).strip()

                    if doc_title:
                        if doc_title.startswith('.'):
                            doc_title = doc_title[1:]
                        segment = '. '.join([doc_title, segment])

                    fout_t5_ids.write(f'{query_id}\t{candidate_doc_id}\t{i}\n')
                    fout_t5_texts.write(
                        f'Query: {query} Document: {segment} '
                        'Relevant:\n')
                    n_segments += 1
                    if i + args.max_length >= len(sentences):
                        break
                seen_doc_ids.add(candidate_doc_id)
    print(f'{num_no_content} examples with only title')
    print(f'Wrote {n_segments} segments from {num_docs} docs.')
    print(f'There were {num_no_segments} docs without segments/sentences.')
    print('Done.')
