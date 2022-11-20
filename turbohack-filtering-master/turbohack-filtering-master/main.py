import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from collections import Counter

pd.options.mode.chained_assignment = None  # default='warn'

# Путь до неразмеченных данных. Является константой, нельзя поменять маршрут монтирования данных.
# Тренировочные / тестовые данные для локальной работы следует размещать по этому маршруту
GT_RAW = "./data/public_test.csv"

# Путь, по которому должен генерироваться итоговый сабмит.
SUBM_PATH = "./data/submission.csv"


def baseline_subm(gt_raw):
    subm_df = gt_raw
    drop_ind = (np.abs(stats.zscore(subm_df)) > 3).any(axis=1)
    subm_df = subm_df.loc[~drop_ind]
    subm_df.to_csv(SUBM_PATH)

# Применим изоляционные деревья
def calc_anomaly_isolation_forest(df, use_columns, contamination=0.1, outliers_value = 1):
    # аргумент contamination - вызывает warning, вроде это проблема в sklearn, надо будет еще разобраться
    i_forest = IsolationForest(n_estimators=100, contamination=contamination, max_samples=df.shape[0], \
                            max_features=len(use_columns), bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
    i_forest.fit(df[use_columns].to_numpy())
    pred = i_forest.predict(df[use_columns].to_numpy())
    anomalys = np.where(pred == -1, outliers_value, (1-outliers_value))
    return anomalys

# Используем модель регрессии для поиска аномалий LinearRegression
def calc_regression(x_train, x_test, y_train, y_test, std_porog = 3):
    reg = LinearRegression().fit(x_train, y_train)
    y_predict = reg.predict(x_test)
    delta_y_test_predict = y_predict - np.array(y_test)
    max(delta_y_test_predict), min(delta_y_test_predict), np.std(delta_y_test_predict)
    std_delta = np.std(delta_y_test_predict)
    mask_anomaly = np.where(abs(delta_y_test_predict) > std_delta*std_porog, True, False)
    return x_test[mask_anomaly].index


# Отбрасываем те значения которые превышают (стандартное отклоенние)*4 - 4 вместо 3 выбраны, т.к. необходимо 100% быть увереным в аномалии
def search_anomalies(data, cnt_std = 2.5):
    data_std = np.std(data)
    data_mean = np.mean(data)
    # data_mean = np.median(data)
    limit_3_std = data_std * cnt_std
    lower_limit  = data_mean - limit_3_std
    upper_limit = data_mean + limit_3_std
    find_outliers = data.where((data < lower_limit) | (data>upper_limit) )
    return find_outliers

# Отбрасываем те значения которые
# 1. рассчитывем квартили (% распределение данных)
# 2. по каждому признаку определяем данные которые выходят
# за границы квартильного размаха (по умолчанию полтора размаха)
# 3. Определяем те объекты, которые выходят за границы квартильного
# размаха по нескольким признакам (по умполчанию 2).
# Другими словами: если у рассматриваемого объекта по двум и более признакам
# выявлен выход за полуторный квартильный размах, то этот объект считаем аномалией
def detect_outliers(df, features, n_features=2, k_outlier=1.5):
    outlier_indices = []
    # проходим по каждому признаку
    for col in features:
        # Определяем Q1 - 1ый квартиль (25% данных будут меньше Q1, а 75% больше чем значение Q1)
        Q1 = np.percentile(df[col], 25)
        # Определяем Q3 - 3ый квартиль (25% данных будут больше Q3, а 75% меньше чем значение Q3)
        Q3 = np.percentile(df[col], 75)
        # IQR межквартильный размах, описывает 50% данных
        IQR = Q3 - Q1
        # Определяем порог для засчитывания значения как аномального
        # по умолчанию 1.5*межквартильного размаха!
        outlier_step = k_outlier * IQR
        # Определяем аномальные данные, которые выходят за порог outlier_step
        outlier_list_col = df[(df[col] < Q1 - outlier_step) |     (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    # Определяем те объекты, которые выходят за границы квартильного размаха по нескольким признакам (по умполчанию 2).
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n_features)
    return multiple_outliers

# Определение выбросов через резкий скачок значений при условии,
# что по дате небыло скачка, т.е. разница между данными 10 минут
def find_gap_anomaly(df, metrics_columns, std_porog=1.5, n_counter=3):
    find_anomaly_rows = []
    for column in metrics_columns:
        current_values = df[column]
        next_values = df[column].shift(-1)
        std_values = df[column].std()
        delta_values = next_values - current_values
        delta_values.iloc[-1] = 0
        df["TempCalc"] = pd.DataFrame(delta_values)
        mask_delta_values = np.where((abs(df["TempCalc"]) > std_values*std_porog)&(df["DeltaTime"] == np.timedelta64(10,'m')), True, False )
        find_anomaly_rows.extend(df[mask_delta_values].index)
    outlier_rows = list(k for k, v in Counter(find_anomaly_rows).items() if v >= n_counter)
    return outlier_rows


def main_clear_app(train_df_step1, metrics):
    # добавляем информацию о соблюдении последовательности
    # т.е. если временная последовательность соблюдена то 1
    # если был скачок по времени (перерыв в данных), то 0

    current_time = train_df_step1["Time"]
    next_time = train_df_step1["Time"].shift(-1)
    # delta_time_series = pd.concat(next_time-current_time,pd.Series([np.timedelta64(10,'m')]) )
    delta_time_series = next_time - current_time
    delta_time_series = pd.DataFrame({"DeltaTime": delta_time_series})
    delta_time_series.iloc[-1] = np.timedelta64(10, 'm')
    # train_df_step1.loc[:,"DeltaTime"] = delta_time_series.loc[:,"DeltaTime"]
    train_df_step1["DeltaTime"] = delta_time_series["DeltaTime"]
    train_df_step1.shape

    # Проходим по каждому признаку и определяем аномалии
    anomaly_rows = []
    for column in metrics:
        anomaly = search_anomalies(train_df_step1[column], cnt_std=3)
        anomaly_rows.extend(list(anomaly.dropna().index))
    # Удаляем дубликаты строк
    anomaly_rows = list(set(anomaly_rows))

    train_df_step1.loc[:,'anomaly_std'] = 0
    train_df_step1.loc[anomaly_rows, ['anomaly_std']] = 1

    # Создаем train_df_step2 в котором уже выкинутые найденные 182 аномалии (Всего осталось аномалий 523->341)
    train_df_step2 = train_df_step1.drop(anomaly_rows)

    # Проходим по каждому признаку и определяем аномалии
    anomaly_rows = []
    for column in metrics:
        anomaly = search_anomalies(train_df_step2[column], cnt_std=3)
        anomaly_rows.extend(list(anomaly.dropna().index))
    # Удаляем дубликаты строк
    anomaly_rows = list(set(anomaly_rows))

    train_df_step2['anomaly_std'] = 0
    train_df_step2.loc[anomaly_rows, ['anomaly_std']] = 1

    # Создаем train_df_step2 в котором уже выкинутые найденные 182 аномалии (Всего осталось аномалий 523->341)
    train_df_step2 = train_df_step2.drop(anomaly_rows)

    # Еще раз проверяем отклонение среднеквадратическое отклонение, но с менее жесткими
    # требованиями, в связи с чем выявлется большое кол-во хороших данных, для их исключения проверятся
    # кол-во попаданий каждого объекта как аномальный по разным признакам. Если отклонение наблюдается
    # по двум и более признакам, то считаем, что объект аномальный

    # Проходим по каждому признаку и определяем аномалии
    anomaly_list = []
    for column in metrics:
        anomaly_series = search_anomalies(train_df_step2[column], cnt_std=2.5)
        anomaly_list.extend(list(anomaly_series.dropna().index))

    anomaly_rows = list(k for k, v in Counter(anomaly_list).items() if v > 10)

    train_df_step2['anomaly_std_multirow'] = 0
    train_df_step2.loc[anomaly_rows, ['anomaly_std_multirow']] = 1

    # обновляем train_df_step2 в котором выкинутые дополнитеьно найденные аномалии
    train_df_step3 = train_df_step2.drop(anomaly_rows)

    train_df_step4 = train_df_step3.copy()

    # Ищем аномалии у которых был резкий скачок (gap > std*1.5) и такие объекты были замечены более чем по 3м признакам
    gap_outlier_rows = find_gap_anomaly(train_df_step4, metrics, std_porog=0.7, n_counter=2)
    train_df_step4['anomaly_gap'] = 0
    train_df_step4.loc[gap_outlier_rows, ['anomaly_gap']] = 1

    # обновляем train_df_step2 в котором выкинутые дополнитеьно найденные аномалии
    train_df_step5 = train_df_step4.drop(gap_outlier_rows)

    train_df_step6 = train_df_step5.copy()

    # Настроим параметры для изоляционных деревьев
    # contamination = 0.003, а в качестве рассамтриваемых колонок используются все признаки
    train_df_step6['anomaly_iforest'] = calc_anomaly_isolation_forest(train_df_step6, metrics, contamination=0.002)
    anomaly_rows = train_df_step6[train_df_step6['anomaly_iforest'] == 1].index

    # # обновляем train_df_step6
    train_df_step7 = train_df_step6.drop(anomaly_rows)

    find_anomaly_regression = []
    for column in metrics:
        x = train_df_step7[metrics].drop(columns=[column])
        y = train_df_step7[column]
        find_outlier_rows = calc_regression(x, x, y, y)
        find_anomaly_regression.extend(find_outlier_rows)

    outlier_rows = list(k for k, v in Counter(find_anomaly_regression).items() if v >= 10)
    train_df_step7['anomaly_dl'] = 0
    train_df_step7.loc[outlier_rows, ['anomaly_dl']] = 1
    train_df_step7['anomaly_dl'].value_counts()

    train_df_step8 = train_df_step7.drop(outlier_rows)

    return train_df_step8

def main():
    train_df = pd.read_csv(GT_RAW).rename(columns={"Параметр": "Time"})
    time_loc_index = train_df.columns.get_loc("Time")
    train_df['Time'] = pd.to_datetime(train_df['Time'])
    train_df_step1 = train_df.copy()

    time_column = train_df_step1.columns[time_loc_index]
    metrics_columns = train_df_step1.columns[(time_loc_index+1):]
    # Группируем коррелирующие параметры. Группы хранятся в groups
    # Последним массивом параметров входящем в groups явл. плохокоррелированные
    corr_columns = train_df_step1[metrics_columns].corr()
    groups = []
    select_columns = []
    for i, col in enumerate(corr_columns.columns):
        if col in select_columns:
            continue
        select_corr_columns = corr_columns[corr_columns[col].abs() > 0.7].index
        if len(select_corr_columns) < 10:
            continue
        select_columns.extend(select_corr_columns)
        groups.append(list(select_corr_columns))
    small_corr_columns = list(set(corr_columns.columns) - set(select_columns))
    if len(small_corr_columns) > 0:
        groups.append(small_corr_columns)
    train_df_step9_all = main_clear_app(train_df_step1[[time_column] + list(metrics_columns)],
                                        metrics_columns)
    train_df_step10_all = train_df_step9_all.copy()
    for i, group in enumerate(groups):
        train_df_step9_group = main_clear_app(train_df_step10_all[[time_column] + group], group)
        train_df_step10_all = train_df_step10_all.loc[train_df_step9_group.index]
    train_df_step10_all[[time_column] + list(metrics_columns)].rename(columns={"Time": "Параметр"}).set_index("Параметр").to_csv(SUBM_PATH)

if __name__ == "__main__":
    main()
