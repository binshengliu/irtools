from krovetzstemmer import Stemmer
from collections import Counter, OrderedDict
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

    def indri_rm1(self, nterms=None, whitelist=None):
        rm = list(sorted(self._rm.items(), reverse=True, key=itemgetter(1)))
        if whitelist:
            rm = list(filter(lambda x: x[0] in whitelist, rm))
        rm = rm[:nterms]
        rm = ['{:f} {}'.format(pair[1], pair[0]) for pair in rm]
        rm = '#weight({})'.format(' '.join(rm))
        return rm

    def bag_of_words(self, nterms=None, whitelist=None):
        rm = list(sorted(self._rm.keys(), reverse=True, key=self._rm.get))
        if whitelist:
            rm = list(filter(whitelist.__contains__, rm))
        rm = ' '.join(rm[:nterms])
        return rm


class TrecRelevanceModels:
    def __init__(self,
                 *,
                 trec_output,
                 doc_texts,
                 stem=False,
                 remove_stop=False):
        """scores: a list of scores, can be log value or probabilities
        terms: [['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h']] or
               ['a b c d', 'e f g h'] or
               [[['a', 0.4], ['b', 0.3], ['c', 0.2]], [['e', 0.5], ['f', 0.5]]]
        """
        self._stem = stem
        self._remove_stop = remove_stop
        self._doc_texts = {}
        self._qno_docs = OrderedDict()
        self._qno_scores = OrderedDict()
        for line in trec_output.splitlines():
            qno, _, docno, _, score, _ = line.split()
            self._qno_docs.setdefault(qno, []).append(docno)
            self._qno_scores.setdefault(qno, []).append(float(score))
            self._doc_texts[docno] = doc_texts[docno]

    def _build_rms(self, ndocs):
        ndocs_dict = self._build_qno_map(ndocs)
        qno_docs = OrderedDict(
            map(lambda k, v: (k, v[:ndocs_dict[k]]), self._qno_docs.keys(),
                self._qno_docs.values()))
        qno_scores = OrderedDict(
            map(lambda k, v: (k, v[:ndocs_dict[k]]), self._qno_scores.keys(),
                self._qno_scores.values()))

        rms = OrderedDict()
        for qno, docs in qno_docs.items():
            terms = [self._doc_texts[d] for d in docs]
            scores = qno_scores[qno]

            rm = RelevanceModel(terms=terms,
                                scores=scores,
                                stem=self._stem,
                                remove_stop=self._remove_stop)
            rms[qno] = rm
        return rms

    def _build_qno_map(self, to_build):
        if isinstance(to_build, dict):
            qno_map = to_build
        elif isinstance(to_build, list):
            qno_map = dict(zip(self._qno_docs.keys(), to_build))
        else:
            qno_map = dict(
                zip(self._qno_docs.keys(),
                    [to_build] * len(self._qno_docs.keys())))
        return qno_map

    def indri_rm1(self, *, ndocs=None, nterms=None, whitelist=None):
        rms = self._build_rms(ndocs)
        terms_dict = self._build_qno_map(nterms)
        white_dict = self._build_qno_map(whitelist)
        queries = [
            rm.indri_rm1(terms_dict[qno], white_dict[qno])
            for qno, rm in rms.items()
        ]
        return list(rms.keys()), queries

    def bag_of_words(self, *, ndocs=None, nterms=None, whitelist=None):
        rms = self._build_rms(ndocs)
        terms_dict = self._build_qno_map(nterms)
        white_dict = self._build_qno_map(whitelist)
        queries = [
            rm.bag_of_words(terms_dict[qno], white_dict[qno])
            for qno, rm in rms.items()
        ]
        return list(rms.keys()), queries
