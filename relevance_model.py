from krovetzstemmer import Stemmer
from collections import Counter
from operator import itemgetter
import math
from stopwords import stopwords


def normalize(input):
    if isinstance(input, list):
        assert all(v >= 0 for v in input)
        s = sum(input)
        input = [v / s for v in input]
        return input
    elif isinstance(input, dict):
        assert all(v >= 0 for v in input.values())
        s = sum(input.values())
        input = {k: v / s for k, v in input.items()}
        return input
    else:
        raise Exception('Unknown input')


class RelevanceModel:
    def __init__(self,
                 *,
                 terms,
                 scores=None,
                 probs=None,
                 stem=False,
                 remove_stop=False):
        """scores: a list of scores, can be log value or probabilities
        terms: [['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h']] or
               ['a b c d', 'e f g h'] or
               [[['a', 0.4], ['b', 0.3], ['c', 0.2]], [['e', 0.5], ['f', 0.5]]]
        """
        if (scores and probs) or (not scores and not probs):
            raise Exception('One of scores and probs must be specified.')

        if scores and not probs:
            probs = [math.exp(s - scores[0]) for s in scores]

        probs = normalize(probs)

        stemmer = Stemmer()

        if isinstance(terms[0], str):
            terms = [s.split() for s in terms]
        else:
            assert hasattr(terms[0], '__iter__')

        if stem:
            terms = [list(map(stemmer.stem, s)) for s in terms]

        rm = {}
        for doc_prob, term_list in zip(probs, terms):
            length = len(term_list)
            for term, occur in Counter(term_list).items():
                rm.setdefault(term, 0.0)
                rm[term] += doc_prob * (occur / length)

        # Removing stop words must be after generating the
        # distribution because it changes document length.
        if remove_stop:
            rm = {t: p for t, p in rm.items() if t not in stopwords}
            rm = normalize(rm)

        self._rm = rm

    def indri_rm1(self, nterms=None):
        rm = list(sorted(self._rm.items(), reverse=True, key=itemgetter(1)))
        rm = rm[:nterms]
        rm = ['{:f} {}'.format(pair[1], pair[0]) for pair in rm]
        rm = '#weight({})'.format(' '.join(rm))
        return rm

    def bag_of_words(self, nterms=None):
        rm = list(sorted(self._rm.keys(), reverse=True, key=self._rm.get))
        rm = ' '.join(rm[:nterms])
        return rm
