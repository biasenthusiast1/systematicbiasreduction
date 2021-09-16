"""
16.2.2021
Author: Shirin
"""
import csv
from configs import CSV_PATH,\
    EXPERIMENT_FP, RESULTS_SAVE_PATH


def creat_fm_dictionary(csv_path):
    """
    creates a dictionary of {"doc_id": [femail_affiliation, mail_affiliation, femail-mail]}
    :param my_csv: the resulting csv file of liwc
    :return: fm_dictionary
    """
    fm_dictionary = {}
    with open(csv_path, "r") as csv_file:
        my_csv = csv.reader(csv_file)
        for line_number, line in enumerate(my_csv):
            # print(line_number)
            doc_idx = line[0].replace(".txt", "")
            femail_aff = float(line[1])
            mail_aff = float(line[2])
            subtract = femail_aff - mail_aff
            fm_dictionary[doc_idx] = [femail_aff, mail_aff, subtract]
    return fm_dictionary


def find_top_n_docs(run_file_path, cutoff):
    """

    :param run_file:
    :return: top_n_docs: stores a dictionary of query_id and top_n doc_ids:
                {"query_id":[docic_1, docid_2, ..., docid_cutoff]}
    """
    top_n_docs = {}
    doc_ids = []
    with open(run_file_path, 'r') as run_file:
        for doc_counter, line in enumerate(run_file):
            list_line = line.split(" ")
            if int(list_line[3]) < cutoff+1:
                query_id = list_line[0]
                # print("finding top-{} docs of query id = {}".format(cutoff, query_id))
                doc_ids.append(list_line[2])
                top_n_docs[query_id] = doc_ids
            else:
                doc_ids = []
    return top_n_docs


def calculate_query_score_cutoff(doc_ids, fm_dictionary):
    """
    calculate the mean fm_score of each query
    based on its top_n retreived docs
    score = abs(mean(femail_score - mail_score))
    :return:
    """
    query_score = [0, 0, 0]
    for doc_idx in doc_ids:
        # query_score += dictionary[doc_idx]
        query_score = [(query_score[i] + fm_dictionary[doc_idx][i]) for i in range(len(query_score))]
    query_score = [abs(score/len(doc_ids))*100 for score in query_score]
    return query_score


def calculate_score_cutoff(topn_docs_dict, fm_dictionary):

    total_score = [0, 0, 0]
    for counter, doc_ids in enumerate(topn_docs_dict.values()):
        # print("calculating score for query number {}".format(counter))
        query_score = calculate_query_score_cutoff(doc_ids, fm_dictionary)
        total_score[0] += query_score[0]
        total_score[1] += query_score[1]
        total_score[2] += query_score[2]
    total_score[0] = total_score[0] / len(topn_docs_dict)
    total_score[1] = total_score[1] / len(topn_docs_dict)
    total_score[2] = total_score[2] / len(topn_docs_dict)
    return total_score


def write_score_cutoffs(csv_path, run_file_path, save_path):
    print("stat creating and saving the fm_dictionary of liwc")
    fm_dict = creat_fm_dictionary(csv_path)
    cutoff_list = [5,10,20,30,50,100]
    resulting_csv_row = []
    with open(save_path, "w") as result_file:
        for cutoff in cutoff_list:
            print("cutoff = {}".format(cutoff))
            print("creat topn docs dictionary")
            topn_docs_dict = find_top_n_docs(run_file_path, cutoff)
            fm_score = calculate_score_cutoff(topn_docs_dict, fm_dict)
            resulting_csv_row += [round(fm_score[2], 2), round(fm_score[0], 2),
                                        round(fm_score[1], 2), round(fm_score[0]-fm_score[1], 2)]
        writer = csv.writer(result_file)
        writer.writerow(resulting_csv_row)
            # log = "cutoff {}-> total score: femail ={}, mail={}, femail - mail ={} \n"\
            #     .format(cutoff, fm_score[0], fm_score[1], fm_score[2])
            # result_file.write(log)


def calculate_total_score_lambdas():
    """

    :return:
    """
    lamdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for my_lambda in lamdas:
        print(my_lambda)
        csv_path = CSV_PATH + "_" +str(my_lambda)+".csv"
        run_file_path = EXPERIMENT_FP + "/expanded_landa_{}".format(my_lambda)+".txt"
        save_path = RESULTS_SAVE_PATH + "expanded_landa_{}".format(my_lambda)+".csv"
        write_score_cutoffs(csv_path, run_file_path, save_path)


def calculate_total_score():
    csv_path = CSV_PATH + ".csv"
    run_file_path = EXPERIMENT_FP +".txt"
    save_path = RESULTS_SAVE_PATH + ".csv"
    write_score_cutoffs(csv_path, run_file_path, save_path)


if __name__ == "__main__":
    calculate_total_score()
    # calculate_total_score_lambdas()
