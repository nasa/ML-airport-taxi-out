import hyperopt as hp
import numpy as np
from sklearn.model_selection import cross_val_score
import logging
from sklearn.metrics import make_scorer

class OptimizerHOP:
    def __init__(self, pipeline, param_grid, trials = hp.Trials(), cv = 3, scoring = None) :
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.trials = trials
        self.cv = cv
        self.scoring = scoring
        
        

    def _negmad(self, yt, yp):
        """
        default optimization metrics if the scoring = None
        """
        log = logging.getLogger(__name__)
        res = yt-yp
        log.info(f"HP Default score function - percentage of NaN residual : {np.isnan(res).sum()/100./len(res)}%")
        return -1*np.nanmedian(np.abs(res-np.nanmedian(res)))*1.4826
        
        
    def _objective(self, param):
        """
        Run a K-fold cross validation
        """
        self.pipeline.set_params(**param)
        #self.pipeline.fit(self.X, self.y)
        #yp = self.pipeline.predict(self.X)
        scores = cross_val_score(self.pipeline, self.X, self.y, cv = self.cv, scoring = self.scoring)
        # store the std of the score since I don't think I can pass that to hyperopt
        self.std_cv.append(np.nanstd(scores))
        # need to return negative since it tries to minimize whereas score are maximum
        return -1.0*np.nanmean(scores)

    def _get_space(self):
        self.space = {}
        for keyi, valuei in self.param_grid.items() :
            self.space[keyi] = hp.hp.choice(keyi, valuei)

    def _conv_trials_to_gridsearch(self):
        ntrials = len(self.trials.trials)
        self.cv_results_ = {'params':[],'mean_test_score':[],'std_test_score':[]}
        param_list = [ x['misc']['vals'] for x in self.trials.trials ]
        self.cv_results_['mean_test_score'] = [ -1*x['result']['loss'] for x in self.trials.trials]
        # std stored separately for now, hopefully no snafu in ordering
        self.cv_results_['std_test_score'] = self.std_cv
        # modify param_list to have values instead of indexes :
        keysp = param_list[0].keys()
        for k in range(ntrials) :
            for keyspi in keysp :
                param_list[k][keyspi] = self.param_grid[keyspi][param_list[k][keyspi][0]]
        ### fill the structure
        self.cv_results_['params']=param_list
        ### get the best params
        self.best_params_ = self.results.copy()
        for keyspi in keysp :
                self.best_params_[keyspi] = self.param_grid[keyspi][self.best_params_[keyspi]]
                

    def fit(self, X, y):
        self.X = X
        self.y = y
        self._get_space()
        self.std_cv = []
        if (self.scoring == None) :
            self.scoring = make_scorer(self._negmad)
        self.results = hp.fmin(self._objective, self.space, algo = hp.tpe.suggest, max_evals = 100, trials = self.trials)
        self._conv_trials_to_gridsearch()
        return self
