from taxi_out.pipelines.data_science.nodes import *
import numpy as np

# TESTs don't work if there are some .. in the parameters.yml
# or other optimization (not tested yet)

# Tests performed :
# On function define/train_unimp_???_model :
#  if Dataset has a constant target and defined features :
#  - Test function returns a FilterPipeline Object
#  - Test prediction array same lenght as input array
#  - Test that prediction is the same constant
#  if Dataset has one feature with bad_percent (40%) of [Nan,'', None]:
#  - Test function returns a FilterPipeline Object
#  - Test prediction array same lenght as input array
#  - Test rows without bad data return the target constant
#  - Test rows with bad core featuress return NaN
#  - Test rows with bad non-core feature return target constant
#  if Dataset with each feature with bad_percent (10%) of [Nan,'', None]:
#  - Test function returns a FilterPipeline Object
#  - Test prediction array same lenght as input array
#  - Test rows without bad data return the target constant
#  - Test rows with bad core featuress return NaN
#  - Test rows with bad non-core feature return target constant
#
# On function predict (w models from full, AMA, ramp) :
#   if Dataset has a constant target and defined features :
#   - Test predict returns a DataFrame Object
#   - Test output DataFrame has same number of rows and 2 additional columns
#   - Test the name of the new columns
#   - Test missing_core_features column is set properly
#
#


# data_fit, config, parameters defined in conftest.py


def _test_train_any_model(data, params, train_func, model_params, filter_pip) :
    """
      Test any train_unimp_???_model with data, this function
      is called by each specific tests (ramp, AMA, full)
    """
    # how many data set to test on :
    ndata = len(data)
    
    for k  in range(ndata) :
        # Checking if predictions have proper size and value
        data_i = data[k].copy()
        data_i[model_params['target']] = data_i['target']
        # select only the main pipelines, not the CV ones
        pip = train_func(data_i, model_params, params['globals'],
                         None, params['RANDOM_SEED'], filter_pip)[0]
        assert type(pip) == FilterPipeline
        # Predict the test expect target value :
        data_preds = data_i.loc[data_i.group == 'test',:].copy()
        preds = pip.predict(data_preds)
        # Output of the model are rounded
        assert (len(preds) == len(data_preds))
        assert (preds == np.round(data_i.loc[0,'target'])).all()

        
        # Checking for bad input in the training data 
        bad_inputs = [np.nan, "", None]
        bad_percent = 0.4
        indx = data_i.index
        nrow = len(indx)
        # Checking each features one at a time:
        core_feats = model_params['features_core']
        feats = model_params['features']
        for feati in feats :
            for bad_inputi in bad_inputs :
                data_i = data[k].copy()
                data_i[model_params['target']] = data_i['target']
                bd_indx = np.random.choice(indx,int(bad_percent*nrow),replace=False)
                data_i.loc[bd_indx,feati] = bad_inputi
                data_i['bad_input'] = False
                data_i.loc[bd_indx,'bad_input'] = True
                # select only the main training pipelines not the CV ones
                pip = train_func(data_i, model_params, params['globals'],
                                 None, params['RANDOM_SEED'], filter_pip)[0]
                assert type(pip) == FilterPipeline
                # Predict the test expect target value :
                data_preds = data_i.loc[data_i.group == 'test',:].copy()
                preds = pip.predict(data_preds)
                # Output of the model are rounded
                assert (len(preds) == len(data_preds))
                # input with good feature should return target
                bd_test = data_preds['bad_input'] 
                assert (preds[~bd_test] == np.round(data_i.loc[0,'target'])).all()                
                # input with bad feature should return NaN (for now) for core
                # and it should return target for non-core
                # WARNING WE MIGHT CHANGE THIS !!!!!!!!!!! FOR CORE AT LEAST
                if (feati in core_feats) :
                    # now it is the default response
                    if (np.isfinite(pip.default_response)) :
                        assert (preds[bd_test] == pip.default_response).all()
                    else :
                        assert (np.isfinite(preds[bd_test]) == False).all()
                else :
                    assert (preds[bd_test] == np.round(data_i.loc[0,'target'])).all()                

        # Checking a mix of bad inputs for different sets
        data_i = data[k].copy()
        data_i[model_params['target']] = data_i['target']
        data_i['bad_input'] = False
        data_i['bad_input_core'] = False
        bad_percent = 0.1
        for feati in feats :
            bd_indx = np.random.choice(indx,int(bad_percent*nrow),replace=False)
            data_i.loc[bd_indx,feati] = np.random.choice(bad_inputs,int(bad_percent*nrow))
            if (feati in core_feats):
                data_i.loc[bd_indx,'bad_input_core'] = True
            else:
                data_i.loc[bd_indx,'bad_input'] = True
        # select only the main training pipelines not the CVs ones
        pip = train_func(data_i, model_params, params['globals'],
                         None, params['RANDOM_SEED'], filter_pip)[0]
        assert type(pip) == FilterPipeline
        # Predict the test expect target value :
        data_preds = data_i.loc[data_i.group == 'test',:].copy()
        preds = pip.predict(data_preds)
        # Output of the model are rounded
        assert (len(preds) == len(data_preds))
        # input with good feature should return target
        bd_test = data_preds['bad_input'] | data_preds['bad_input_core']
        assert (preds[~bd_test] == np.round(data_i.loc[0,'target'])).all()                
        # input with bad feature should return NaN (for now) for core
        # and it should return target for non-core
        # WARNING WE MIGHT CHANGE THIS !!!!!!!!!!! FOR CORE AT LEAST
        bd_test_core = data_preds['bad_input_core']
        # now it is the default response
        if (np.isfinite(pip.default_response)) :
            assert (preds[bd_test_core] == pip.default_response).all()
        else :
            assert (np.isfinite(preds[bd_test_core]) == False).all()
        bd_test_noncore = ~data_preds['bad_input_core'] & data_preds['bad_input']
        assert (preds[bd_test_noncore] == np.round(data_i.loc[0,'target'])).all()                
        


def _test_define_any_model(data, params, pipeline, model_params) :
    data[model_params['target']] = data['target'] # needed to define the filter
    out_pip = pipeline(data, model_params, params['globals'], params['RANDOM_SEED'])
    assert isinstance(out_pip, FilterPipeline)
    return out_pip


# Test main model definition/training functions

# unimpeded        
def test_define_unimp_full_model(data_fit, parameters) :
    """
      Test the node define_ and train_ unimp_full_model with data_fit
    """
    # extract informations to be passed to the generic test function
    model_params = parameters['unimp_full_model_params']
    pipeline = define_unimp_full_model
    defined_pip = _test_define_any_model(data_fit[0], parameters, pipeline, model_params)
    train_func = train_unimp_full_model
    _test_train_any_model(data_fit, parameters, train_func, model_params, defined_pip)

        
def test_define_unimp_ama_model(data_fit, parameters) :
    """
      Test the node define_ and train_ unimp_ama_model with data_fit
    """
    # extract informations to be passed to the generic test function
    model_params = parameters['unimp_ama_model_params']
    pipeline = define_unimp_ama_model
    defined_pip = _test_define_any_model(data_fit[0], parameters, pipeline, model_params)
    train_func = train_unimp_ama_model
    _test_train_any_model(data_fit, parameters, train_func, model_params, defined_pip)

              
        
def test_define_unimp_ramp_model(data_fit, parameters) :
    """
      Test the node define_ and train_ unimp_ramp_model with data_fit
    """
    # extract informations to be passed to the generic test function
    model_params = parameters['unimp_ramp_model_params']
    pipeline = define_unimp_ramp_model
    defined_pip = _test_define_any_model(data_fit[0], parameters, pipeline, model_params)
    train_func = train_unimp_ramp_model
    _test_train_any_model(data_fit, parameters, train_func, model_params, defined_pip)


# impeded    
def test_define_imp_full_model(data_fit, parameters) :
    """
      Test the node define_ and train_ imp_full_model with data_fit
    """
    # extract informations to be passed to the generic test function
    model_params = parameters['imp_full_model_params']
    pipeline = define_imp_full_model
    defined_pip = _test_define_any_model(data_fit[0], parameters, pipeline, model_params)
    train_func = train_imp_full_model
    _test_train_any_model(data_fit, parameters, train_func, model_params, defined_pip)

def test_define_imp_ama_model(data_fit, parameters) :
    """
      Test the node define_ and train_ imp_ama_model with data_fit
    """
    # extract informations to be passed to the generic test function
    model_params = parameters['imp_ama_model_params']
    pipeline = define_imp_ama_model
    defined_pip = _test_define_any_model(data_fit[0], parameters, pipeline, model_params)
    train_func = train_imp_ama_model
    _test_train_any_model(data_fit, parameters, train_func, model_params, defined_pip)


def test_define_imp_ramp_model(data_fit, parameters) :
    """
      Test the node define_ and train_ imp_ramp_model with data_fit
    """
    # extract informations to be passed to the generic test function
    model_params = parameters['imp_ramp_model_params']
    pipeline = define_imp_ramp_model
    defined_pip = _test_define_any_model(data_fit[0], parameters, pipeline, model_params)
    train_func = train_imp_ramp_model
    _test_train_any_model(data_fit, parameters, train_func, model_params, defined_pip)



###  Test predict function ;


def _test_predict_w_any(data, params, train_func, model_params, filter_pip) :
    """
      Test the predict node for any model (full, AMA, ramp)
    """
    ndata = len(data)
    for k  in range(ndata) :
        # Checking if predictions have proper size and value
        data_i = data[k].copy()
        data_i[model_params['target']] = data_i['target']
        # select only the main training pipelines, not the CV ones
        pip = train_func(data_i, model_params, params['globals'],
                         None, params['RANDOM_SEED'], filter_pip)[0]
        data_i_out = predict(pip, data_i, model_params)
        # Check we return a DataFrame
        assert isinstance(data_i_out,pd.DataFrame)
        # Check the number of rows and columns
        assert len(data_i) == len(data_i_out)
        assert len(data_i.columns) == len(data_i_out.columns)-2
        # Check keywords are in the output dataframe
        assert 'missing_core_features' in data_i_out
        assert 'predicted_{}'.format(model_params['name']) in data_i_out
        # Check if the missing_core_features field is working
        bad_core = (data_i[model_params['features_core']].isin(['',None,np.nan]) |
                    data_i[model_params['features_core']].isna()).any(axis=1)
        assert data_i_out.loc[bad_core,'missing_core_features'].all()
        assert (data_i_out.loc[~bad_core,'missing_core_features'] == False).all()
        

# unimpeded        
def test_predict_w_unimp_full(data_fit, parameters) :
    """
      Test the node predict with data_fit with unimp full taxi model
    """
    # predict is ran on a trained model
    model_params = parameters['unimp_full_model_params']
    defined_pip = define_unimp_full_model(data_fit[0], model_params, parameters['globals'],
                                      parameters['RANDOM_SEED'])
    train_func = train_unimp_full_model
    _test_predict_w_any(data_fit, parameters, train_func, model_params, defined_pip)
        

        
def test_predict_w_unimp_ama(data_fit, parameters) :
    """
      Test the node predict with data_fit with unimp ama taxi model
    """
    # predict is ran on a trained model
    model_params = parameters['unimp_ama_model_params']
    defined_pip = define_unimp_ama_model(data_fit[0], model_params, parameters['globals'],
                                         parameters['RANDOM_SEED'])
    train_func = train_unimp_ama_model
    _test_predict_w_any(data_fit, parameters, train_func, model_params, defined_pip)
        

            
def test_predict_w_unimp_ramp(data_fit, parameters) :
    """
      Test the node predict with data_fit with unimp ramp taxi model
    """
    # predict is ran on a trained model
    model_params = parameters['unimp_ramp_model_params']
    defined_pip = define_unimp_ramp_model(data_fit[0], model_params, parameters['globals'],
                                         parameters['RANDOM_SEED'])
    train_func = train_unimp_ramp_model 
    _test_predict_w_any(data_fit, parameters, train_func, model_params, defined_pip)


    
# impeded    
def test_predict_w_imp_full(data_fit, parameters) :
    """
      Test the node predict with data_fit with imp full taxi model
    """
    # predict is ran on a trained model
    model_params = parameters['imp_full_model_params']
    defined_pip = define_imp_full_model(data_fit[0], model_params, parameters['globals'],
                                         parameters['RANDOM_SEED'])
    train_func = train_imp_full_model 
    _test_predict_w_any(data_fit, parameters, train_func, model_params, defined_pip)

    
def test_predict_w_imp_ama(data_fit, parameters) :
    """
      Test the node predict with data_fit with imp ama taxi model
    """
    # predict is ran on a trained model
    model_params = parameters['imp_ama_model_params']
    defined_pip = define_imp_ama_model(data_fit[0], model_params, parameters['globals'],
                                         parameters['RANDOM_SEED'])
    train_func = train_imp_ama_model 
    _test_predict_w_any(data_fit, parameters, train_func, model_params, defined_pip)

    
def test_predict_w_imp_ramp(data_fit, parameters) :
    """
      Test the node predict with data_fit with imp ramp taxi model
    """
    # predict is ran on a trained model
    model_params = parameters['imp_ramp_model_params']
    defined_pip = define_imp_ramp_model(data_fit[0], model_params, parameters['globals'],
                                         parameters['RANDOM_SEED'])
    train_func = train_imp_ramp_model 
    _test_predict_w_any(data_fit, parameters, train_func, model_params, defined_pip)
        
