import re
import csv
import time
from os import listdir, path, cpu_count
from collections import Counter
from multiprocessing import Pool
import pandas as pd
import liwc
from preprocess import preprocess_gov2
from configs import CSV_PATH, EXPERIMENT_FP


def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)


def calculate_gendered_count(input_fp):
    """

    :param document:
    :return:
    """
    parse, _ = liwc.load_token_parser('../data/LIWC2015Dictionary.dic')
    query_df = pd.read_csv(input_fp, names=["qid", "terms", "weight"])
    # df_unbiased = pd.read_csv(UNBIASED_EXPANSION_FP, names=["qid", "terms", "weight"])
    #
    qids = list(pd.unique(query_df['qid']))
    total_fm_terms = 0
    total_male_terms = 0
    for query_id in qids:
        female_bias = 0
        male_bias = 0
        query_terms = query_df[query_df["qid"] == query_id]
        query = query_terms["terms"].tolist()
        query_str = " ".join(query)
        doc_tokens = tokenize(query_str)
        # liwc_counts = Counter(category for token in doc_tokens for category in parse(token))
        categories = []
        # token_counter = 0
        for token in doc_tokens:
            token = token.lower()
            # token_counter += 1
            for category in parse(token):
                categories.append(category)
        liwc_counts = Counter(categories)
        if "female" in liwc_counts.keys():
            female_bias = liwc_counts["female"]
        if "male" in liwc_counts.keys():
            male_bias = liwc_counts["male"]
        total_fm_terms += female_bias
        total_male_terms += male_bias
    return total_fm_terms, total_male_terms


def calculate_doc_score(doc_str, parse):
    """

    :param document:
    :return:
    """
    doc_tokens = tokenize(doc_str)
    # liwc_counts = Counter(category for token in doc_tokens for category in parse(token))
    categories = []
    token_counter = 0
    for token in doc_tokens:
        token = token.lower()
        token_counter += 1
        for category in parse(token):
            categories.append(category)
    liwc_counts = Counter(categories)
    if "female" in liwc_counts.keys():
        female_bias = liwc_counts["female"] / token_counter
    else:
        female_bias = 0
    if "male" in liwc_counts.keys():
        male_bias = liwc_counts["male"] / token_counter
    else:
        male_bias = 0
    return female_bias, male_bias


def calculate_file_score(extracted_docs_fp, csv_path):
    """
    writes liwc scores of different files in a csv file
    :return:
    """
    print(extracted_docs_fp)
    parse, _ = liwc.load_token_parser('../data/LIWC2015Dictionary.dic')
    with open(csv_path, 'w') as liwc_csv_results:
        writer = csv.writer(liwc_csv_results)
        for row_number, file in enumerate(listdir(extracted_docs_fp)):
            row = [file]
            with open(path.join(extracted_docs_fp, file)) as document:
                doc_str = document.read()
                doc_str = preprocess_gov2(doc_str)
                female_bias, male_bias = calculate_doc_score(doc_str, parse)
                row.append(female_bias)
                row.append(male_bias)
                writer.writerow(row)
                # if row_number % 1000 == 0:
                #     print(row_number)


def calculate_liwc_lamdas():
    """
    writes liwc scores of different runs of lambda
    in a csv file
    :return:
    """
    lamdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for my_lambda in lamdas:
        print(my_lambda)
        my_csv_path = CSV_PATH + "_" +str(my_lambda)+".csv"
        my_docs = EXPERIMENT_FP + "/expanded_landa_{}".format(my_lambda)
        calculate_file_score(my_docs, my_csv_path)


def multiprocess_liwc_lambdas():
    lamdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    data = []
    for my_lambda in lamdas:
        print(my_lambda)
        my_csv_path = CSV_PATH + "_" +str(my_lambda)+".csv"
        my_docs = EXPERIMENT_FP + "/expanded_landa_{}".format(my_lambda)
        data.append((my_docs, my_csv_path))
    process_pool = Pool(cpu_count())
    process_pool.starmap(calculate_file_score, data)


def multiprocess_file_score():
    experiments = [".bm25", ".bm25+rm3"]
    datasets = ["cw09", "cw12"]
    data = []
    for dataset in datasets:
        for experiment in experiments:
            my_experiment = "run." +dataset + experiment
            experiment_fp = "../data/" + dataset + "/" + my_experiment
            csv_path = "../data/" + dataset + "/liwc_"+my_experiment+".csv"
            data.append((experiment_fp, csv_path))
    process_pool = Pool(cpu_count())
    process_pool.starmap(calculate_file_score, data)


if __name__ == "__main__":
    TIME_START = time.time()
    calculate_liwc_lamdas()
    calculate_file_score(EXPERIMENT_FP, CSV_PATH)
    multiprocess_liwc_lambdas()
    multiprocess_file_score()
    TIME_END = time.time()
    print("total time = {}".format(TIME_END-TIME_START))

    # dataset = "cw12"
    # file_path = "../data/expanded_queries/"+dataset+"/cleaned_queries_0.5.txt"
    # total_fm_terms, total_male_terms = calculate_gendered_count(file_path)
    # print("total number of gendered terms in robust04 expanded_queries_0.5 = {}"
    #       .format(total_fm_terms+total_male_terms))
