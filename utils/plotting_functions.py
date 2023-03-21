
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import sys 
sys.path.append('..')
from utils.dataloader import load_data, create_count_data

pio.templates.default = "plotly_white"

#plt.style.use('ggplot')
pd.set_option('display.max_columns', None)


def plot_time_series(df_in, start_date = 0, end_date = 0, save_figure = False):
    plot_list = df_in['Cluster'].unique()[:-1]
    fig, axes = plt.subplots(4,2, figsize=(12, 14), sharey = True)
    i = 0
    j = 0
    for key in plot_list:
        df_censored_temp = df_in[(df_in['Cluster'] == key)]

        if (start_date == 0):
            df_censored_temp.plot(y = ['Sessions', 'PlugCap'], x = 'Period', ax = axes[i,j])
        else:
            df_censored_temp = df_censored_temp[(df_censored_temp['Period'] >= start_date) & (df_censored_temp['Period'] <= end_date)]
            df_censored_temp.plot(y = ['Sessions', 'PlugCap'], x = 'Period', ax = axes[i,j])
        axes[i,j].set_title(str(key))
        axes[i,j].legend(loc = 'upper left')
        no_events = len(df_censored_temp)
        no_exceed_plugcap = (len(df_censored_temp[df_censored_temp['Sessions'] > df_censored_temp['PlugCap']]) / no_events)*100
        usage_fraction = np.mean(df_censored_temp['Sessions'] / df_censored_temp['PlugCap'])

        textstr = '\n'.join((
        r'$\eta =%.2f$' % (no_exceed_plugcap, ),
        r'$\mu =%.2f$' % (usage_fraction , )))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[i,j].text(0.855, 0.962, textstr, transform=axes[i,j].transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

        i += 1
        if i == 4:
            j += 1
            i = 0
    fig.tight_layout()
    print(r'$\eta is the percentage of time intervals where the amount sessions exceed the number of plugs (due to time aggregation)$')
    print(r'$\mu is the average fraction of plugs in use (out of the total number)$')

    if (save_figure == True):
        fig.savefig("/Users/julian/Documents/GitHub/Thesis/Figures/Time_series_{start_date}_{end_date}.png", bbox_inches='tight')


def plot_histograms(df_in, save_figure = False):
    plot_list = df_in['Cluster'].unique()[:-1]
    
    fig, axes = plt.subplots(4,2, figsize=(12, 16))
    i = 0
    j = 0
    for key in plot_list:
        cluster_events = df_in['Sessions'].loc[df_in['Cluster'] == key]
        cluster_mean = np.mean(df_in['Sessions'].loc[df_in['Cluster'] == key])
        cluster_sd = np.std(df_in['Sessions'].loc[df_in['Cluster'] == key])

        #cluster_events = df_event[key].values
        #cluster_mean = np.mean(cluster_events)
        #cluster_sd = np.std(cluster_events)

        axes[i,j].hist(cluster_events, bins = range(int(np.max(cluster_events))+3), rwidth=0.7)
        axes[i,j].set_title(str(key))
        axes[i,j].plot()
        axes[i,j].set_xticks(np.arange(np.max(cluster_events)+3) + 0.5)
        axes[i,j].set_xticklabels(np.arange(np.max(cluster_events)+3))

        textstr = '\n'.join((
        r'$\mu=%.2f$' % (cluster_mean, ),
        r'$\sigma=%.2f$' % (cluster_sd, )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[i,j].text(0.83, 0.962, textstr, transform=axes[i,j].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

        i += 1
        if i == 4:
            j += 1
            i = 0
    
    fig.tight_layout()
    if (save_figure == True):
        fig.savefig("/Users/julian/Documents/GitHub/Thesis/Figures/Distributions.png", bbox_inches='tight')
    return

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def ts_percentile(df, n=2, time_scale = 24, Cluster = 'HAMILTON', value = 'Sessions', percentile_min=10, percentile_max=90, color='r', plot_mean=False, plot_median=True, line_color='k', ax1 = 0, ax2 = 0, plt_axes = axes, **kwargs):
    x = np.arange(0,time_scale)

    # calculate the lower and upper percentile groups, skipping 50 percentile
    perc1 = np.zeros((n, time_scale))
    perc2 = np.zeros((n, time_scale))

    if time_scale == 7:
        group_var = 'Day'
    else: 
        group_var = 'Hour'

    label_list = []
    j = 0
    for i in np.linspace(percentile_min, 50, n)[:-1]:
        perc1[j,:] = df[df['Cluster'] == Cluster].groupby(group_var).agg([percentile(i)])[value].values.flatten()
        label_list.append(i)
        j +=1
    
    k = 0
    for i in np.linspace(50, percentile_max, n)[1:]:
        perc2[k,:] = df[df['Cluster'] == Cluster].groupby(group_var).agg([percentile(i)])[value].values.flatten()
        label_list.append(i)
        k +=1

    if 'alpha' in kwargs:
        alpha = kwargs.pop('alpha')
    else:
        alpha = 1/n
    # fill lower and upper percentile groups
    p = 0
    for p1, p2 in zip(perc1, perc2):
        if (p == n-1):
            label_name = '_nolegend_'
        else:
            label_name = f'{label_list[p]:.0f}-{100-label_list[p]:.0f}% percentile'
        axes[ax1, ax2].fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor=None, label = label_name)
        p += 1

    if plot_mean:
        axes[ax1, ax2].plot(x, df[df['Cluster'] == Cluster].groupby(group_var).mean()[value].values, color=line_color, label = 'Mean')

    if plot_median:
        axes[ax1, ax2].plot(x, df[df['Cluster'] == Cluster].groupby(group_var).median()[value].values, color=line_color)

    return