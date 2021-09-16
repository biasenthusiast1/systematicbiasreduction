from pyserini.search import SimpleSearcher
import os
# import numpy as np
# from configs import DATASET


def extract_documents():
    experiments = ["run.cw12.bm25+rm3", "run.cw12.bm25"]
    # searcher = SimpleSearcher('/data/anserini/lucene-index.gov2.pos+docvectors+rawdocs')
    # searcher = SimpleSearcher.from_prebuilt_index('robust04')
    searcher = SimpleSearcher('/data/anserini/lucene-index.cw12b13.pos+docvectors+rawdocs')

    for experiment in experiments:
        file_address = "../data/cw12/"+experiment+".txt"
        with open(file_address, "r") as index_file:
            if not os.path.exists("../data/cw12/"+experiment):
                os.makedirs("../data/cw12/"+experiment)
            for line_number, line in enumerate(index_file):
                # print(line.split(" ")[3])
                idx = line.split(" ")[2]
                write_address = "../data/cw12/"+experiment+"/"+idx+".txt"
                doc = searcher.doc(idx)
                with open(write_address, "w") as file_to_write:
                    file_to_write.write(doc.raw())
                if line_number % 1000 == 0:
                    print(line_number)


def extract_expanded_documents():
    experiment = "unbiased_expansions"
    searcher = SimpleSearcher('/data/anserini/lucene-index.cw12b13.pos+docvectors+rawdocs')

    # searcher = SimpleSearcher.from_prebuilt_index('robust04')
    lamdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for my_lambda in lamdas:
        print(my_lambda)
        my_directory = "../data/cw12/"+experiment +\
                      "/expanded_landa_"+str(my_lambda)
        file_address = my_directory +".txt"
        with open(file_address, "r") as index_file:
            if not os.path.exists(my_directory):
                os.makedirs(my_directory)
            for line_number, line in enumerate(index_file):
                # print(line.split(" ")[3])
                idx = line.split(" ")[2]
                write_address = my_directory +"/"+idx+".txt"
                doc = searcher.doc(idx)
                with open(write_address, "w") as file_to_write:
                    file_to_write.write(doc.raw())
                # if line_number % 1000 == 0:
                #     print(line_number)


if __name__ == "__main__":
    extract_documents()