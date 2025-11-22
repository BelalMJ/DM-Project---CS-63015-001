import numpy as np
from scipy import stats

def paired_t_test(scores_model_1, scores_model_2, alpha=0.05):
    scores_model_1 = np.array(scores_model_1)
    scores_model_2 = np.array(scores_model_2)
    t_stat, p_value = stats.ttest_rel(scores_model_1, scores_model_2, nan_policy='omit')
    significant = p_value < alpha
    return t_stat, p_value, significant
