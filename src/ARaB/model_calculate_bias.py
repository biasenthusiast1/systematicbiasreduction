import collections
import numpy as np
import pickle
import pandas as pd 

experiments = ['bert-mini',
               'betr-mini_bias',
               'bert-tiny',
               'bert-tiny_bias',
               'electra-small',
               'electra-small_bias',
               'distilroberta',
               'distilroberta_bias'
               ]

metrics = ['ARaB']
methods = ['tf', 'bool']

qry_bias_paths = {}
for metric in metrics:
    qry_bias_paths[metric] = {}
    for exp_name in experiments:
        qry_bias_paths[metric][exp_name] = {}
        for _method in methods:
            qry_bias_paths[metric][exp_name][_method] = 'data/ARaB/ecir_neutrals/%s_run_bias_%s_%s.pkl' \
                                                        % (exp_name, _method, metric)

queries_gender_annotated_path = "resources/queries_gender_annotated.csv"

at_ranklist = [5, 10, 20, 30, 50, 100]

qry_bias_perqry = {}

for metric in metrics:
    qry_bias_perqry[metric] = {}
    for exp_name in experiments:
        qry_bias_perqry[metric][exp_name] = {}
        for _method in methods:
            _path = qry_bias_paths[metric][exp_name][_method]
            print (_path)
            with open(_path, 'rb') as fr:
                qry_bias_perqry[metric][exp_name][_method] = pickle.load(fr)


df = pd.read_csv("results/bias_inferences/ecir_neutrals/inference_bert-mini.trec", sep = " ", names = ['qid','q0',"docid","r","s","a"])
query_ids = pd.unique(df['qid']).tolist()

eval_results_bias = {}
eval_results_feml = {}
eval_results_male = {}

for metric in metrics:
    eval_results_bias[metric] = {}
    eval_results_feml[metric] = {}
    eval_results_male[metric] = {}
    for exp_name in experiments:
        eval_results_bias[metric][exp_name] = {}
        eval_results_feml[metric][exp_name] = {}
        eval_results_male[metric][exp_name] = {}
        for _method in methods:
            eval_results_bias[metric][exp_name][_method] = {}
            eval_results_feml[metric][exp_name][_method] = {}
            eval_results_male[metric][exp_name][_method] = {}
            for at_rank in at_ranklist:
                _bias_list = []
                _feml_list = []
                _male_list = []

                for qryid in query_ids:
                    _bias_list.append(qry_bias_perqry[metric][exp_name][_method][at_rank][qryid][0])
                    _feml_list.append(qry_bias_perqry[metric][exp_name][_method][at_rank][qryid][1])
                    _male_list.append(qry_bias_perqry[metric][exp_name][_method][at_rank][qryid][2])
                    
                eval_results_bias[metric][exp_name][_method][at_rank] = np.mean([(_male_x-_feml_x) for _male_x, _feml_x in zip(_male_list, _feml_list)])
                eval_results_feml[metric][exp_name][_method][at_rank] = np.mean(_feml_list)
                eval_results_male[metric][exp_name][_method][at_rank] = np.mean(_male_list)


result = []  
for metric in metrics:
    print (metric)  
    for at_rank in at_ranklist:
        tmp = []
        tmp.append(at_rank)
        for _method in methods:
            tmp.append(_method)
            for exp_name in experiments:
                print ("%25s\t%2d %5s\t%f\t%f\t%f" % (exp_name, at_rank, _method, eval_results_bias[metric][exp_name][_method][at_rank], eval_results_feml[metric][exp_name][_method][at_rank], eval_results_male[metric][exp_name][_method][at_rank]))
                tmp.append(eval_results_bias[metric][exp_name][_method][at_rank])
        result.append(tmp)
        # print()
        # print ("==========")

print(result)
# df = pd.DataFrame(result, columns = ["cut_off", "TF", "bert_biased", "bert_mixed_5","bert_mixed_10", "bert_mixed_15", "bert_mixed_20", "bert_mixed_25", "bool","bert_biased", "bert_mixed_5","bert_mixed_10", "bert_mixed_15", "bert_mixed_20", "bert_mixed_25"])  
df = pd.DataFrame(result, columns = ["cut_off",
                                     "TF",
                                       'bert-mini',
                                       'betr-mini_bias',
                                       'bert-tiny',
                                       'bert-tiny_bias',
                                       'electra-small',
                                       'electra-small_bias',
                                       'distilroberta',
                                       'distilroberta_bias',
                                       "bool",
                                       'bert-mini',
                                       'betr-mini_bias',
                                       'bert-tiny',
                                       'bert-tiny_bias',
                                       'electra-small',
                                       'electra-small_bias',
                                       'distilroberta',
                                       'distilroberta_bias'])
df.to_csv("./data/ARaB/ecir_neutrals/ARaB.csv")












