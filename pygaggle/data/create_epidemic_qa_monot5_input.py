"""
This script creates monoT5 input files by taking corpus,
queries and the retrieval run file for the queries and then
create files for monoT5 input. Each line in the monoT5 input
file follows the format:
    f'Query: {query} Document: {document} Relevant:\n')
"""
import collections
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from os import path

parser = argparse.ArgumentParser()
parser.add_argument("--queries", type=str, required=True,
                    help="tsv file with two columns, <query_id> and <query_text>")
parser.add_argument("--run", type=str, required=True,
                    help="tsv file with three columns <query_id>, <doc_id> and <rank>")
parser.add_argument('--corpus', required=True, type=str,
                    help='The path to the directory containing \
                          the document JSON files')
parser.add_argument("--t5_input", type=str, required=True,
                    help="path to store t5_input, txt format")
parser.add_argument("--t5_input_ids", type=str, required=True,
                    help="path to store the query-doc ids of t5_input, tsv format")
parser.add_argument('--use-question-and-query', action='store_true',
                    help="If true, the t5 input query is a concatenation of \
                        question and query.  Otherwise, it's just the question.")
args = parser.parse_args()

def get_document_text(path_to_toplevel_docs_directory, filename):
    """
    Returns a string containing document text prepended with the title.
    """
    if path.exists(path_to_toplevel_docs_directory + "/" + filename +".json"):
        filepath = path_to_toplevel_docs_directory + "/" + filename +".json"
    else:
    # The final-round consumer primary corpus has nested subdirectories
    # breaking up documents by search.  Also, some of the consumer documents 
    # are suffixed with an additional GUID.  Therefore, we need to do a global
    # search for the file if the initial search doesn't work.
        filepaths = list(Path(path_to_toplevel_docs_directory).rglob(filename+"*"))
        if len(filepaths) == 0:
            print("Unable to find document named " + filename)
            return ""
        elif len(filepaths) > 1:
            print("Multiple paths found for document named " + filename)

        filepath = filepaths[0]

    with open(filepath) as f:
        raw_json = f.read()
        parsed_json = json.loads(raw_json)
    metadata = parsed_json["metadata"]
    # Some consumer documents don't have titles.
    title=""
    if metadata["title"]:
        title=metadata["title"]

    document_text = ""
    for context in parsed_json["contexts"]:
        document_text += context["text"]
        document_text += "\n"

    return title + " " + document_text


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
            if args.use_question_and_query:
                queries[question_id] = question + "? " + query
            else:
                queries[question_id] = question + " " + query

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

queries = load_queries(path=args.queries)
run = load_run(path=args.run)

print("Writing t5 input and ids")
with open(args.t5_input, 'w') as fout_t5, open(args.t5_input_ids, 'w') as fout_tsv:
    for num_examples, (query_id, candidate_doc_ids) in enumerate(
            tqdm(run.items(), total=len(run))):
        query = queries[query_id]
        for candidate_doc_id in candidate_doc_ids:
            document_text = get_document_text(args.corpus, candidate_doc_id)
            fout_t5.write(
                f'Query: {query} Document: {document_text} Relevant:\n')
            fout_tsv.write(f'{query_id}\t{candidate_doc_id}\n')
