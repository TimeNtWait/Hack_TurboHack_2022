import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

GT_RAW = "./nikita/gt_raw.csv" # это файл с датасетом, который предоставлен участникам
GT_FILTERED = "./nikita/gt_filtered.csv" # это датасет, из которого эксперты отфильтровали строки
SUBM_PATH = "./nikita/turbo_pred_1.csv" # это файл участников, датасет из которого они отфильтровали строки


GT_RAW = "./turbohack-filtering-master/turbohack-filtering-master/data/public_test.csv" # это файл с датасетом, который предоставлен участникам
GT_FILTERED = "./turbohack-filtering-master/turbohack-filtering-master/data/submission.csv" # это датасет, из которого эксперты отфильтровали строки
SUBM_PATH = "./turbohack-filtering-master/turbohack-filtering-master/data/submission.csv" # это файл участников, датасет из которого они отфильтровали строки

def metric(y_true, y_pred):
    return f1_score(y_true, y_pred)

def prepare_df(path):
    df = pd.read_csv(path)
    df["Параметр"] = pd.to_datetime(df["Параметр"])
    df.set_index("Параметр", inplace=True)
    return df

def main():
    gt_raw = prepare_df(GT_RAW)
    gt_filtered = prepare_df(GT_FILTERED)
    subm_df = prepare_df(SUBM_PATH)

    gt_raw["label_true"] = 1 # 1, значит строка должна быть удалена
    gt_raw["label_pred"] = 1
    gt_raw.loc[gt_filtered.index, "label_true"] = 0 # 0, эти строки остаются
    gt_raw.loc[subm_df.index, "label_pred"] = 0
    
    true = gt_raw["label_true"]
    pred = gt_raw["label_pred"]
    
    return metric(true, pred)

if __name__ == "__main__":
    print(main())