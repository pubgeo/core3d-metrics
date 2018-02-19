

def calcMops(true_positives, false_negatives, false_positives):
    s = {
        'recall': true_positives / (true_positives + false_negatives),
        'precision': true_positives / (true_positives + false_positives),
        'jaccardIndex': true_positives / (true_positives + false_negatives + false_positives),
        'branchingFactor': false_positives / true_positives,
        'missFactor': false_negatives / true_positives,
    }
    s['completeness'] = s['recall']
    s['correctness'] = s['precision']
    s['fscore'] = (2 * s['recall'] * s['precision']) / (s['recall'] + s['precision'])

    return s


def getUnitArea(tform):
    return abs(tform[1] * tform[5])


def getUnitHeight(tform):
    return (abs(tform[1]) + abs(tform[5])) / 2
