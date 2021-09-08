# Build artifacts to log train set distribution
import pandas as pd
import numpy as np
import seaborn as sn
import textwrap as twp
import matplotlib.pyplot as pl
import matplotlib as mat
pl.ioff()
mappables = []
      
# def time_transform(x):
#     return pd.to_datetime(x).apply( lambda t : (t.hour*60+t.minute)//10) 

def time_transform(x):
    return pd.to_datetime(x).apply( lambda t : (t.hour*60+t.minute)//60) 



def calculate_mappable_bounds(heatmap, cmap=mat.cm.viridis) :
    '''
    Create a scale with 5 discrete intervals instead of continuous color/scale
    from the minimum to the maximum of an heatmap
    '''
    vmin, vmax = heatmap.min().min(), heatmap.max().max()
    bounds = [vmin,(vmax-vmin)/50+vmin,(vmax-vmin)/20+vmin,(vmax-vmin)/5+vmin,(vmax-vmin)/2+vmin,vmax]
    bounds = [int(x) for x in bounds]
    norm = mat.colors.BoundaryNorm(bounds, cmap.N)
    mappable = mat.cm.ScalarMappable(norm = norm, cmap = cmap)
    return [mappable, bounds, norm]


def remove_low_num_and_nan(dataframe) :
    removed = {}
    kept = {}
    for col in dataframe.columns :
        value_ct = dataframe[col].value_counts()
        removed[col] = value_ct[value_ct < value_ct.max()/100.0].to_dict()
        kept[col] = value_ct[value_ct >= value_ct.max()/100.0].index
    for kepti in kept.items() :
        dataframe = dataframe[dataframe[kepti[0]].isin(kepti[1])]
    return dataframe, kept, removed    


def diag_plot_gen(max_val) :
    def diag_plot(xarray, **kwargs):
        histv = xarray.value_counts().sort_index()
        diag_plot.call_ct += 1
        if (diag_plot.call_ct < max_val) :
            kwargs.update({'position': -0.5})
        histv.plot(kind='bar', **kwargs)
        ax = pl.gca()
        ax.tick_params(direction='in', labelsize=7, which = 'both')
    diag_plot.call_ct = 0
    return diag_plot
    

def off_plot(xarray, yarray, **kwargs):
    dfcomb = pd.concat((yarray, xarray),axis=1)
    hist2d = dfcomb.groupby(dfcomb.columns.tolist()).size().unstack()
    cmap = mat.cm.viridis
    mappable, bounds, norm  = calculate_mappable_bounds(hist2d, cmap=cmap)
    mappables.append([mappable, bounds,[xarray.name,yarray.name]])
    sn.heatmap(hist2d,cbar=False,cmap=cmap,norm=norm, **kwargs, xticklabels=True)
    ax = pl.gca()
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    ax.tick_params(direction='in',labelsize=7, which='both')

    
def add_zero_infront_ofgate(dataframe, col) :
    selection = dataframe[col].str.match('^[A-K][0-9][A-Z]*$')
    selection = selection.fillna(False)
    dataframe.loc[selection,col] = dataframe.loc[selection, col].apply(lambda x : x[0]+'0'+x[1:])
    return dataframe
  






def training_set_feature_distributions(dataframe, model) :
    '''
    Create a triangular distribution plot of the feature values
    (For now only work up to 6 features, after issue with colorbar)
    
    Parameters
    ----------

    dataframe : dataframe containing the data of the pipeline

    model : model dictionary containing the feature names

    
    Returns
    -------

    filenames of the plots

    
    '''
    df = dataframe.copy()
    # impeded pipelines push the gufi column to index, so I need to push it back
    if (df.index.name == 'gufi') :
        df = df.reset_index()
    features = model['features']
    nfeat = len(features)
    df['terminal'] = df['departure_stand_actual'].str.extract('^([A-K])[0-9]+')
    df = add_zero_infront_ofgate(df, 'departure_stand_actual')

    terminal_names = df['terminal'].value_counts()
    # need terminals with enough flights
    terminal_names = terminal_names[terminal_names > 50].index.to_numpy()
    outfiles = []
    # switch time to binned values :
    if (("Binned_features" in model) and isinstance(model["Binned_features"],list)) :
        for featuri in model["Binned_features"] :
            df[featuri] = time_transform(df[featuri])


    
    for terminal_name in terminal_names :
        dfs = df[(df.terminal == terminal_name)].copy()
        if ('unimpeded_AMA' in dfs) :
            dfa = dfs[dfs.unimpeded_AMA & (dfs.group == 'train')]
        else :
            dfa = dfs[dfs.group == 'train']
        dfa, kept, removed = remove_low_num_and_nan(dfa[features])
        graph = sn.PairGrid(dfa, vars=features, corner=True,\
                            diag_sharey= False, despine=False, aspect = 1.5)
        graph.map_diag(diag_plot_gen(nfeat))
        graph.map_lower(off_plot)
        # rotate the bottom right xlabel
        axc = graph.axes[nfeat-1][nfeat-1]
        if (len(axc.get_xticks()) > 5) :
            axc.set_xticklabels(axc.get_xticklabels(), rotation = 90)
       
            

        # Missing/low feature info :
        dl = 0.015*6/nfeat
        miss_info_string = []
        low_info_string = []
        for k in range(nfeat) :
            feat = features[k]
            missing_feat = set(dfa[feat].unique()) - set(dfs[feat].unique())
            miss_info_string.append('missing '+feat+' :'+str(missing_feat))
            low_feat = removed[feat]
            lines = twp.wrap('low count '+feat+' :'+str(low_feat))
            low_info_string += lines
        pl.annotate("\n".join(miss_info_string), (0.4, 0.9),
                    xycoords='figure fraction', color='r')
        pl.annotate("\n".join(low_info_string),(0.4,0.9-dl*len(low_info_string)),\
                    xycoords="figure fraction",color='Orange')   

    
            
        # add some colorbars
        pos_col = nfeat*nfeat-nfeat
        pl.subplot(nfeat,nfeat,pos_col)
        k,l=2,0
        pl.axis('off')
        fracs = [0.3, 0.45, 0.9] # moving colorbar away from each other
        for map_ct in range(len(mappables)):        
            if (k == -1) :
                pos_col = pos_col - nfeat
                pl.subplot(nfeat,nfeat,pos_col)
                k = 2
                l = l+1
                pl.axis('off')
            mappable = mappables[k+l*3][0]
            bounds = mappables[k+l*3][1]
            label = "\n".join(mappables[k+l*3][2])
            cb=pl.colorbar(mappable, label=label, aspect = 10, fraction = fracs[2-k], ticks= bounds)
            cb.ax.tick_params(labelsize=7)
            cb.ax.yaxis.set_ticks_position('left')
            cb.set_label(cb._label,size=7,labelpad=0)
            k = k-1

        fign = pl.get_fignums()[-1]
        fig = pl.figure(fign)
        fig.suptitle('Terminal '+terminal_name)
        outfile = 'terminal_'+terminal_name+'_training_distributions.png'
        pl.savefig(outfile,bbox_inches='tight')
        pl.close()
        outfiles.append(outfile)
        mappables.clear()

    return outfiles
