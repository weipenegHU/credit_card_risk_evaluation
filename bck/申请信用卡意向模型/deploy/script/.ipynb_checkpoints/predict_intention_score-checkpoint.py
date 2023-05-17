import pandas as pd
import numpy as np
import warnings
import joblib
import sys

warnings.filterwarnings("ignore", module="lightgbm")


def preprocess(file):
    data = pd.read_csv(file, header=None, low_memory=False)
    data.columns = ['customer_name',
                    'send_date_key',
                    'product_key',
                    'customer_key',
                    'customer_phone',
                    'customer_source',
                    'age',
                    'gender',
                    'mob',
                    'new_customer',
                    'product_factor',
                    'product_name',
                    'non_desensitization',
                    'apply_credit_card_average_month',
                    'apply_loan_average_month',
                    'apply_credit_card_online_average_month',
                    'apply_loan_online_average_month',
                    'credit_card_apply_count',
                    'loan_product_apply_count',
                    'credit_card_audit_count',
                    'credit_card_reject_count',
                    'credit_card_audit_rate',
                    'credit_card_reject_rate',
                    'last_apply_product_type_key',
                    'credit_card_online_apply_count',
                    'loan_product_online_apply_count',
                    'days_from_last_credit_card_apply',
                    'history_apply_specific_bank',
                    'audit_specific_bank',
                    'reject_specific_bank',
                    'apply_same_bank_as_last_apply',
                    'line_key',
                    'push_times_in_thirty_dats',
                    'degree',
                    'have_credit_cards',
                    'bank',
                    'need_loan',
                    'career',
                    'credit_cards_num',
                    'pos_merchant']
    return data


# 通过转换类型，减少数据所占内存
def reduce_memory_usage(matrix):
    matrix['age'] = matrix['age'].astype("int8")
    matrix['gender'] = matrix['gender'].astype("int8")
    matrix['mob'] = matrix['mob'].astype("int16")
    matrix['credit_card_apply_count'] = matrix['credit_card_apply_count'].astype("int8")
    matrix['loan_product_apply_count'] = matrix['loan_product_apply_count'].astype("int8")
    matrix['credit_card_audit_count'] = matrix['credit_card_audit_count'].astype("float32")
    matrix['credit_card_reject_count'] = matrix['credit_card_reject_count'].astype("float32")
    matrix['credit_card_audit_rate'] = matrix['credit_card_audit_rate'].astype("float32")
    matrix['credit_card_reject_rate'] = matrix['credit_card_reject_rate'].astype('float32')
    matrix['credit_card_online_apply_count'] = matrix['credit_card_online_apply_count'].astype("int8")
    matrix['loan_product_online_apply_count'] = matrix['loan_product_online_apply_count'].astype("int8")
    matrix['days_from_last_credit_card_apply'] = matrix['days_from_last_credit_card_apply'].astype("int16")
    matrix['push_times_in_thirty_dats'] = matrix['push_times_in_thirty_dats'].fillna(0).astype("int16")
    matrix['apply_credit_card_average_month'] = matrix['apply_credit_card_average_month'].astype("float16")
    matrix['apply_loan_average_month'] = matrix['apply_loan_average_month'].astype("float16")
    matrix['apply_credit_card_online_average_month'] = matrix['apply_credit_card_online_average_month'].astype("float16")
    matrix['apply_loan_online_average_month'] = matrix['apply_loan_online_average_month'].astype("float16")
    matrix['new_customer'] = matrix['new_customer'].astype("int8")
    matrix['last_apply_product_type_key'] = matrix['last_apply_product_type_key'].astype("int8")
    matrix['history_apply_specific_bank'] = matrix['history_apply_specific_bank'].astype("int8")
    matrix['audit_specific_bank'] = matrix['audit_specific_bank'].astype("int8")
    matrix['reject_specific_bank'] = matrix['reject_specific_bank'].astype("int8")
    matrix['apply_same_bank_as_last_apply'] = matrix['apply_same_bank_as_last_apply'].astype("int8")
    matrix['line_key'] = matrix['line_key'].astype("int8")
    matrix['pos_merchant'] = matrix['pos_merchant'].astype("int8")
    matrix['need_loan'] = matrix['need_loan'].astype("int8")
    matrix['degree'] = matrix['degree'].astype("int8")
    matrix['have_credit_cards'] = matrix['have_credit_cards'].astype("int8")
    matrix['credit_cards_num'] = matrix['credit_cards_num'].astype("int8")
    matrix['career'] = matrix['career'].astype("int8")
    return matrix


def feature_engineering(matrix):
    bank_encoder = joblib.load(BANK_ENCODER_PATH)
    major_banks = ['招商银行', '中信银行', '交通银行', '光大银行', '工商银行', '平安银行', '渤海银行', '广发银行',
                   '民生银行', '华夏银行', '浦发银行', '建设银行']
    matrix['bank_processed'] = matrix['bank'].copy().astype('str')
    matrix.loc[~matrix['bank'].isin(major_banks), 'bank_processed'] = "rare"
    matrix['bank_processed'] = bank_encoder.transform(matrix['bank_processed']).astype("int8")
    matrix = matrix.drop(columns=['bank'])

    kmeans = joblib.load(KMEANS_PATH)
    cluster_features = ['credit_card_apply_count', 'loan_product_apply_count', 'credit_card_online_apply_count',
                        'loan_product_online_apply_count', 'apply_credit_card_average_month', 'apply_loan_average_month',
                        'apply_credit_card_online_average_month',
                        'apply_loan_online_average_month']
    matrix['cluster_n_3'] = kmeans.predict(matrix[cluster_features])

    # 因为lgbm是按列的index，而不是按名称抽取特征，因此要对列重新排序，以保持和训练集的一致
    matrix = matrix[['customer_key', 'age', 'gender', 'mob','credit_card_apply_count', 'loan_product_apply_count', 'credit_card_audit_count',
                     'credit_card_reject_count', 'credit_card_audit_rate', 'credit_card_reject_rate', 'last_apply_product_type_key',
                     'credit_card_online_apply_count', 'loan_product_online_apply_count', 'days_from_last_credit_card_apply',
                     'history_apply_specific_bank', 'audit_specific_bank', 'reject_specific_bank',
                     'apply_same_bank_as_last_apply', 'push_times_in_thirty_dats', 'apply_credit_card_average_month',
                     'apply_loan_average_month', 'apply_credit_card_online_average_month',
                     'apply_loan_online_average_month', 'new_customer', 'bank_processed',
                     'line_key', 'cluster_n_3', 'pos_merchant', 'need_loan', 'degree', 'have_credit_cards',
                     'credit_cards_num', 'career']]

    return matrix


def predict_apply_score(file_path):
    matrix = preprocess(file_path)
    matrix = feature_engineering(reduce_memory_usage(matrix))
    results = np.array(matrix['customer_key'])
    models = []
    for num_model in range(0, 2):
        model = f'model_{num_model}'
        lgbm = joblib.load(f'{LGBM_PATH}gbm{num_model:1d}.pkl')
        pred_score = lgbm.predict(matrix.drop(columns=['customer_key'], errors='ignore'))
        results = np.column_stack((results, pred_score))
        models.append(model)

    results = pd.DataFrame(results, columns=['customer_key'] + models)
    results = results.melt(id_vars=['customer_key'], value_vars=models, var_name='model',
                              value_name='score')

    return results


if __name__ == "__main__":
    # 修改模型所在路径
    KMEANS_PATH = "/sda/share/notebooks/notebooks/credit_card_intention_model/deploy/model/utils/kmeans.pkl"
    BANK_ENCODER_PATH = "/sda/share/notebooks/notebooks/credit_card_intention_model/deploy/model/bank_encoder.pkl"
    POS_MERCHANT_PATH = "/sda/share/notebooks/notebooks/credit_card_intention_model/deploy/data/pos_merchant.csv"
    LGBM_PATH = "/sda/share/notebooks/notebooks/credit_card_intention_model/deploy/model/gbm/"

    input = sys.argv[1]
    # low_bound = sys.argv[2]
    # up_bound = sys.argv[3]
    output = sys.argv[2]
    predictions = predict_apply_score(input)
    # print(predictions)
    predictions.to_csv(output, index=False)
