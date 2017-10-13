import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_score = float('Inf')
        best_n = self.n_constant

        num_data = sum(self.lengths)
        num_features = len(self.X[0])

        for n in range(self.min_n_components, self.max_n_components):
            num_parameters = n*(n-1) + n + n*num_features
            hmm_model = self.base_model(n)
            try:
                if hmm_model:
                    logL = hmm_model.score(self.X, self.lengths)
                    BIC = -2*logL+num_parameters*math.log(num_data)
            except:
                BIC = float('Inf')

            if BIC < best_score:
                best_score = BIC
                best_n = n

        return self.base_model(best_n)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_score = float('-Inf')
        best_n = self.n_constant

        for n in range(self.min_n_components, self.max_n_components):
            hmm_model = self.base_model(n)
            
            if hmm_model:
                try: 
                    logL = hmm_model.score(self.X, self.lengths)
                except:
                    logL = float('-Inf')

                logL_y = []
                for key in self.hwords.keys():
                    if key != self.this_word:
                        X, sequences = self.hwords[key]
                        try:
                            logL_y.append(hmm_model.score(X, sequences))
                        except:
                            continue
            else:
                logL = float('-Inf')
            if len(logL_y) > 0:
                score = logL - np.mean(logL_y)
            else:
                score = logL
            if score > best_score:
                best_score = score
                best_n = n

        return self.base_model(best_n)



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = -float('Inf')
        best_n = self.n_constant
        n_split = 3
        split_method = KFold(n_splits = n_split)

        for n in range(self.min_n_components, self.max_n_components):
            score = 0
            if len(self.sequences) <= n_split:
                hmm_model = self.base_model(n)
                try:
                    if hmm_model:
                        score = hmm_model.score(self.X, self.lengths)
                except:
                    score = -float('Inf')
            else:
                cv_score = []
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_X, train_seqs = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_seqs = combine_sequences(cv_test_idx, self.sequences)
                    try:
                        hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(train_X, train_seqs)
                    except:
                        hmm_model = None
                    try:
                        if hmm_model:
                            cv_score.append(hmm_model.score(test_X, test_seqs))
                    except:
                        continue

                score = np.mean(cv_score)

            

            if score > best_score:
                best_score = score
                best_n = n
        return self.base_model(best_n)
                
