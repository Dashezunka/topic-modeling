import math
import numpy as np

def calc_cosine_similarity(p, q):
    sum_of_products = 0
    p_sqr_sum = 0
    q_sqr_sum = 0
    for i in range(len(p)):
        sum_of_products += p[i] * q[i]
        p_sqr_sum += p[i] ** 2
        q_sqr_sum += q[i] ** 2
    return sum_of_products / (math.sqrt(p_sqr_sum) * math.sqrt(q_sqr_sum))


def norm_components(vec):
    comp_sum = np.sum(vec)
    return [i/comp_sum for i in vec]


def calc_Kullback_Leibler_divergence(p, q):
    result = 0
    p_k_norm = norm_components(p)
    q_k_norm = norm_components(q)
    for k in range(len(p)):
        if p_k_norm[k] == 0 and q_k_norm[k] == 0:
            continue
        if p_k_norm[k] == 0 or q_k_norm[k] == 0:
            result += math.log(2**30)
            continue
        result += p_k_norm[k] * math.log(p_k_norm[k] / q_k_norm[k])

    return result


def calc_Jensen_Shannon_divergence(p, q):
    m = []
    for k in range(len(p)):
        m.append((p[k] + q[k]) / 2)
    result = calc_Kullback_Leibler_divergence(p,m) + calc_Kullback_Leibler_divergence(q,m)
    return result