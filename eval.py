import sklearn.metrics
import pandas as pd
import numpy as np
import os


def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    try:
         mape_val = np.mean(np.abs((actual - pred) / actual)) * 100
    except:
        mape_val = np.nan
    return mape_val

def mpe(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    try:
        mpe_val = np.mean((actual - pred) / actual) * 100
    except:
        mpe_val = np.nan
    return mpe_val


ct = pd.read_csv("./census_tracker.csv")
ct_vars = ct['Variable'].to_list()

for gpu in range(2, 8):

    preds_dir = os.path.join("gpu" + str(gpu), "final", "preds")
    
    print(preds_dir)

    # maes, r2s, mapes, mpes, eabssums, esums = [], [], [], [], [], []

    for preds_df in os.listdir(preds_dir):

        variable = preds_df.split(".")[0][:-8]
        cur_kfold = int(preds_df.split("_")[-2]) + 1

        preds_df = pd.read_csv(os.path.join(preds_dir, preds_df))
        pred_df = preds_df[preds_df["tv"] == "val"]

        cur_mae = sklearn.metrics.mean_absolute_error(preds_df['true'], preds_df['pred'])
        cur_r2 = sklearn.metrics.r2_score(preds_df['true'], preds_df['pred'])
        cur_mape = mape(preds_df['true'], preds_df['pred'])
        cur_mpe = mpe(preds_df['true'], preds_df['pred'])
        cur_abssume = np.sum(np.abs(preds_df['true'] - preds_df['pred']))
        cur_sume = np.sum(preds_df['true'] - preds_df['pred'])

        stat_names = ['MAE', 'R2', 'MAPE', 'MPE', "Sum E", "Abs Sum E"]
        stat_vals = [cur_mae, cur_r2, cur_mape, cur_mpe, cur_sume, cur_abssume]

        for stat in range(0, len(stat_names)):
            col = "kfold " + str(cur_kfold) + " " + stat_names[stat]
            row = ct_vars.index(variable)
            ct.at[row, col] = stat_vals[stat]

    ct.to_csv("./census_tracker.csv")



