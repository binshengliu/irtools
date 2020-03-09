def wtl_seq(base, compare, threshold=0.1):
    win, tie, loss = 0, 0, 0
    for e1, e2 in zip(base, compare):
        if e2 > e1 and (e2 - e1) > e1 * threshold:
            win += 1
        elif e1 > e2 and (e1 - e2) > e1 * threshold:
            loss += 1
        else:
            tie += 1

    return win, tie, loss


def wtl_dict(base, compare, threshold=0.1):
    common = list(base.keys() & compare.keys())
    base = [base[x] for x in common]
    compare = [compare[x] for x in common]
    return wtl_seq(base, compare, threshold)
