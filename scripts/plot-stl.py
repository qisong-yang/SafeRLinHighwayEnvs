import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np


from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def plot_data(data, xaxis='Epoch', value="AverageEpRet", 
              condition="Condition1", smooth=1, paper=True,
              hidelegend=False, title=None, savedir=None, 
              clear_xticks=False, cost_limit1 = 15, cost_limit2 = 5, **kwargs):
    # special handling for plotting a horizontal line
    splits = value.split(',')
    value = splits[0]
    #if len(splits) > 1:
    #    y_horiz = float(splits[1])
    #else:
    #    y_horiz = None
    y_horiz, ymin, ymax = None, None, None
    if len(splits) > 1:
        for split in splits[1:]:
            if split[0]=='h':
                y_horiz = float(split[1:])
            elif split[0]=='l':
                ymin = float(split[1:])
            elif split[0]=='u':
                ymax = float(split[1:])

    if isinstance(data, list):
        # Seive data so only data with value column is present
        data = [x for x in data if value in x.columns]

    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    font_scale = 1 if paper else 1.5
    
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.3))

    #sns.set(style="whitegrid", font_scale=2.2)
    sns.set_context("paper",font_scale=2.2, rc={"lines.linewidth": 3})
    """
    #sns.set_palette(sns.color_palette('muted'))

    sns.set_palette([(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 #(1.0, 0.4980392156862745, 0.054901960784313725),
 (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
 (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
 (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
 #(0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
 #(0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
 #(0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
 (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
 (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]
)
    #"""
    sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', legend = False, **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    ax.legend(loc='best')#.draggable()
    #plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.ylim((0, 65))
    #plt.xlim((30000, 4.42e6))
    plt.xlim((30000, 2.7e6))
    #plt.xlim((30000, 2.4e6))

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xmax = np.max(np.asarray(data[xaxis]))
    xscale = xmax > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    old_ymin, old_ymax = plt.ylim()

    if ymin:
        plt.ylim(bottom=min(ymin, old_ymin))

    if ymax:
        plt.ylim(top=max(ymax, old_ymax))

    #if title:
    #    plt.title(title)

    if paper:
        ax.gcf().set_size_inches(3.85,2.75)
        plt.tight_layout(pad=0.5)
    else:
        plt.tight_layout(pad=0.5)


    if y_horiz:
        # y, xmin, xmax, colors='k', linestyles='solid', label='',
        ax.hlines(y_horiz, 0, xmax, colors='red', linestyles='dashed', label='limit')

    fname = osp.join(savedir, title+'_'+value).lower()

    if clear_xticks:
        x, _ = ax.xticks()
        ax.xticks(x, [])
        ax.xlabel('')
        fname += '_nox'

    if savedir is not '':
        os.makedirs(savedir, exist_ok=True)
        ax.savefig(fname+'.pdf', format='pdf')

    if hidelegend:
        ax.legend().remove()

        if savedir is not '':
            ax.savefig(fname + '_nolegend.pdf', format='pdf')

    if savedir is not '':
        # Separately save legend
        h, l = ax.axes().get_legend_handles_labels()
        legfig, legax = ax.subplots(figsize=(7.5,0.75))
        legax.set_facecolor('white')
        leg = legax.legend(h, l, loc='center', ncol=5, handlelength=1.5,
                   mode="expand", borderaxespad=0., prop={'size': 13})
        legax.xaxis.set_visible(False)
        legax.yaxis.set_visible(False)
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        ax.tight_layout(pad=0.5)
        ax.savefig(osp.join(savedir, title+'_legend.pdf'), format='pdf')
    #plt.xlabel('')
    ax.hlines(cost_limit1, 30000, 2.99e6, colors='b', linestyles='dashed')
    ax.hlines(cost_limit2, 30000, 2.99e6, colors='r', linestyles='dashed')
    ax.vlines(1.5e6, -10, 80, colors='k', linestyles='dashed')
    #ax.text(1.25e6,50,'$d^*$')
    #ax.text(1.25e6,40,'$d$')
    
    
    
    axins = ax.inset_axes((0.1, 0.55, 0.35, 0.35))
    sns.tsplot(data=data, ax = axins, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', legend = False, **kwargs)
    axins.hlines(cost_limit1, 30000, 2.99e6, colors='b', linestyles='dashed')
    axins.hlines(cost_limit2, 30000, 2.99e6, colors='r', linestyles='dashed')
    axins.set_xlim(2.3e6, 2.7e6)
    axins.set_ylim(2, 15)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    axins.set_xticks([])
    axins.set_yticks([])
    
    axins.set_xlabel('')
    axins.set_ylabel('')
    
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='0.5', lw=1)
    plt.legend(frameon=False)
    plt.savefig("stl2dx.pdf")
    
    


def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root,'progress.txt'))
            except:
                print('Could not read from %s'%os.path.join(root,'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns),'Condition1',condition1)
            exp_data.insert(len(exp_data.columns),'Condition2',condition2)
            exp_data.insert(len(exp_data.columns),'Performance',exp_data[performance])
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]=='/':
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split('/')[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,  
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean',
               paper=False, hidelegend=False, title=None, savedir=None, show=True,
               clear_xticks=False, cl1 = None, cl2 = None):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value, condition=condition, 
                  smooth=smooth, estimator=estimator,
                  paper=paper, hidelegend=hidelegend, 
                  title=title, savedir=savedir,
                  clear_xticks=clear_xticks, cost_limit1 = cl1, cost_limit2 = cl2)
    if show:
        '''
        '''
        #plt.xlim((0, 9.2e6))
        plt.show()
        


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    parser.add_argument('--paper', action='store_true')
    parser.add_argument('--hidelegend', '-hl', action='store_true')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--savedir', type=str, default='')
    parser.add_argument('--dont_show', action='store_true')
    parser.add_argument('--clearx', action='store_true')
    parser.add_argument('--cl1', type=float, default=15)
    parser.add_argument('--cl2', type=float, default=5)
    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """

    make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count, 
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est, paper=args.paper, hidelegend=args.hidelegend,
               title=args.title, savedir=args.savedir, show=not(args.dont_show),
               clear_xticks=args.clearx, cl1=args.cl1, cl2=args.cl2)

if __name__ == "__main__":
    main()