

#TP: True Positive, FN: False Negetice, FP: False Positive
def calcMops(TP, FN, FP):

    s = dict()
    s['recall']            = TP / (TP + FN)
    s['precision']         = TP / (TP + FP)
    s['completeness']      = s['recall']
    s['correctness']       = s['precision']
    s['fscore']            = (2 * s['recall']  * s['precision'] ) / ( s['recall'] + s['precision'])
    s['jaccardIndex']      = TP / (TP + FN + FP)
    s['branchingFactor']   = FP / TP
    s['missFactor']        = FN / TP

    return s