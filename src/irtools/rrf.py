from operator import itemgetter


class FuseRrf:
    def __init__(self, k=60, depth=1000):
        self.k = k
        self.depth = depth

    def fuse(self, sorted_runs, weights=None):
        """Implements a reciprocal rank fusion as define in

            ``Reciprocal Rank fusion outperforms Condorcet and
            individual Rank Learning Methods`` by Cormack, Clarke and
            Buettcher.

            Parameters: k: term to avoid vanishing importance of
                lower-ranked documents. Default value is 60 (default
                value used in their paper).

        """

        doc_scores = {}
        if weights is None:
            weights = [1.0] * len(sorted_runs)
        assert len(sorted_runs) == len(weights)
        for sorted_dnos, weight in zip(sorted_runs, weights):
            for pos, dno in enumerate(sorted_dnos[:self.depth], start=1):
                score = (1.0 / (self.k + pos)) * weight
                doc_scores[dno] = doc_scores.get(dno, 0.0) + score

        # Writes out information for this topic
        fused = sorted(
            doc_scores.items(), key=itemgetter(1), reverse=True)[:self.depth]

        return fused

    def __call__(self, sorted_runs, weights=None):
        return self.fuse(sorted_runs, weights)
