import rouge
import numpy as np


def macro_average(raw_rouges):
    """Calculate maco average along raw rouge scores
       i.e. with 'ref', 'hyp', 'overlap'

    Args:
        raw_rouges(list): raw rouge scores, of the following format
            raw_rouges=[$rouge_scores,]
            rouge_scores={$metric: {'hyp': int,
                                    'ref': int,
                                    'overlap': int}
            metric in ["rouge-1", "rouge-2", "rouge-l"]

        Returns:
            macro_average(dict):
                {
                    $metric: {
                        'f': float,
                        'r': float,
                        'p': float
                    }
                }
    """
    rouge_frp = [
        {
            m: rouge.rouge_score.f_r_p_rouge_n(_[m]["hyp"],
                                               _[m]["ref"],
                                               _[m]["overlap"])
            for m in _.keys()
        } for _ in raw_rouges
    ]

    macro_average = {
        m: {
            s: np.mean([_[m][s] for _ in rouge_frp])
            for s in rouge_frp[0][m].keys()
        } for m in rouge_frp[0].keys()
    }
    return macro_average, rouge_frp


def micro_average(raw_rouges):
    rouge_sums = {
        m: {
            s: sum([_[m][s] for _ in raw_rouges])
            for s in raw_rouges[0][m].keys()
        } for m in raw_rouges[0].keys()
    }

    rouge_micro = {
        m: rouge.rouge_score.f_r_p_rouge_n(rouge_sums[m]["hyp"],
                                           rouge_sums[m]["ref"],
                                           rouge_sums[m]["overlap"])
        for m in rouge_sums.keys()
    }

    return rouge_micro, rouge_sums
