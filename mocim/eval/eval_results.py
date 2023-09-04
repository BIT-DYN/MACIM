# -*- coding:UTF-8 -*-

from re import S
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import MultipleLocator
import os
import json
import cv2

title_fontsize = 30
ax_label_fontsize = 32
ax_label_fontsize_large = 30
legend_fontsize = 17
ticks_fontsize = 18
table = np.zeros((6,8,3))



def restructure_res(res):
    times = []
    av_l1 = []
    mesh_acc = []
    mesh_com = []

    sdf_eval_res = res["sdf_eval"]
    for t in sdf_eval_res.keys():
        times.append(sdf_eval_res[t]['time'])
        av_l1.append(sdf_eval_res[t]['rays']['av_l1'])

    mesh_eval_res = res["mesh_eval"]
    for t in mesh_eval_res.keys():
        mesh_acc.append(mesh_eval_res[t]['acc'])
        mesh_com.append(mesh_eval_res[t]['comp'])

    return (
        times, av_l1, mesh_acc, mesh_com
    )


def do_plot(
    ax, times, av_l1, mesh_acc, mesh_com,
    color, label=None, linestyle="-", col_offset=0, title=None
):
    label_name = labels[methods.index(label)]
    # change to cm
    av_l1 = np.array(av_l1) * 100
    # statistics all robots eval results
    std = np.nanstd(av_l1, axis=0)
    av_l1 = np.nanmean(av_l1, axis=0)
    ax[0,col_offset].fill_between(times, av_l1 + std, av_l1 - std, alpha=0.5, color=color)
    ax[0,col_offset].plot(times, av_l1, label=label_name, color=color, linestyle=linestyle)
    x_major_locator=MultipleLocator(5)
    ax[0,col_offset].xaxis.set_major_locator(x_major_locator)
    ax[0,col_offset].tick_params(axis='x', which='major', labelsize=ticks_fontsize)
    if  label is not None:
        ax[0,col_offset].legend(fontsize=legend_fontsize)
    
    mesh_acc = np.array(mesh_acc) * 100
    std = np.nanstd(mesh_acc, axis=0)
    mesh_acc = np.nanmean(mesh_acc, axis=0)
    ax[1,col_offset].fill_between(times, mesh_acc + std, mesh_acc - std, alpha=0.5, color=color)
    ax[1,col_offset].plot(times, mesh_acc, label=label_name, color=color, linestyle=linestyle)
    x_major_locator=MultipleLocator(5)
    ax[1,col_offset].xaxis.set_major_locator(x_major_locator)
    ax[1,col_offset].tick_params(axis='x', which='major', labelsize=ticks_fontsize)
    if  label is not None:
        ax[1,col_offset].legend(fontsize=legend_fontsize)

    mesh_com = np.array(mesh_com) * 100
    std = np.nanstd(mesh_com, axis=0)
    mesh_com = np.nanmean(mesh_com, axis=0)
    ax[2,col_offset].fill_between(times, mesh_com + std, mesh_com - std, alpha=0.5, color=color)
    ax[2,col_offset].plot(times, mesh_com, label=label_name, color=color, linestyle=linestyle)
    x_major_locator=MultipleLocator(5)
    ax[2,col_offset].xaxis.set_major_locator(x_major_locator)
    ax[2,col_offset].tick_params(axis='x', which='major', labelsize=ticks_fontsize)
    if  label is not None:
        ax[2,col_offset].legend(fontsize=legend_fontsize)
    
    if title is not None:
        ax[0,col_offset].title.set_text(title)
        ax[0,col_offset].title.set_size(title_fontsize)
        ax[0,col_offset].title.set_style('italic')
        ax[0,col_offset].title.set_color('black')
        ax[1,col_offset].title.set_text(title)
        ax[1,col_offset].title.set_size(title_fontsize)
        ax[1,col_offset].title.set_style('italic')
        ax[1,col_offset].title.set_color('black')
        ax[2,col_offset].title.set_text(title)
        ax[2,col_offset].title.set_size(title_fontsize)
        ax[2,col_offset].title.set_style('italic')
        ax[2,col_offset].title.set_color('black')

    if label is not None and title is not None:
        table[methods.index(label),seqs.index(title),0]=av_l1[-1]
        table[methods.index(label),seqs.index(title),1]=mesh_acc[-1]
        table[methods.index(label),seqs.index(title),2]=mesh_com[-1]


def sdf_ax_ticks(ax):
    for c in range(ax.shape[0]):
        ax[c].set_yscale('log')
        ax[c].set_ylim([2, 50])
        yticks = [2, 5, 10, 20, 50]
        ytick_labels = [f'{y:.0f}' for y in yticks]
        ax[c].set_yticks(yticks)
        ax[c].set_yticklabels(ytick_labels, fontsize=ticks_fontsize)
        ax[c].minorticks_off()

def acc_ax_ticks(ax):
    for c in range(ax.shape[0]):
        ax[c].set_yscale('log')
        ax[c].set_ylim([5, 100])
        yticks = [5, 10, 20, 40, 100]
        ytick_labels = [f'{y:.0f}' for y in yticks]
        ax[c].set_yticks(yticks)
        ax[c].set_yticklabels(ytick_labels, fontsize=ticks_fontsize)
        ax[c].minorticks_off()

def com_ax_ticks(ax):
    for c in range(ax.shape[0]):
        ax[c].set_yscale('log')
        ax[c].set_ylim([2, 100])
        yticks = [2, 5, 20, 40, 100]
        ytick_labels = [f'{y:.0f}' for y in yticks]
        ax[c].set_yticks(yticks)
        ax[c].set_yticklabels(ytick_labels, fontsize=ticks_fontsize)
        ax[c].minorticks_off()

def save_plots(axax, seq, methods, col_offset = 0):
    # The time sequence for this scene.
    end_time_this = end_time.get(seq)
    modified = int(end_time_this)+1 if int(end_time_this)%2==0 else int(end_time_this)
    times = [x for x in range(2,modified,2)]
    times.append(end_time_this)
    
    for index, method in enumerate(methods):
        av_l1_all = []
        mesh_acc_all = []
        mesh_com_all = []
        for i in range(4):
            # Open each robot json
            res_file = os.path.join(root, seq+"_"+method, "robot_"+str(i)+"_res.json")
            with open(res_file, 'r') as f:
                res = json.load(f)
            (times_all, av_l1, mesh_acc, mesh_com) = restructure_res(res)
            # Choose the last time result as the final one.
            num_out = len(times_all) - len(times) + 1
            av_l1 = av_l1[:-num_out]+[av_l1[-1]]
            mesh_acc = mesh_acc[:-num_out]+[mesh_acc[-1]]
            mesh_com = mesh_com[:-num_out]+[mesh_com[-1]]
            # Stack four robot eval results.
            av_l1_all.append(av_l1)
            mesh_acc_all.append(mesh_acc)
            mesh_com_all.append(mesh_com)
        do_plot(axax, times, av_l1_all, mesh_acc_all, mesh_com_all, 
            "C"+str(index), label=method, col_offset=col_offset, title=seq)


# Directories --------------------------------------------

root = "/home/dyn/SDF/Multi-mocim/results/"

methods = ['single', 'dsgd', 'dsgt', 'dlm', 'dinno', 'mocim']
labels = ['Single', 'DSGD', 'DSGT', 'DLM', 'DiNNO', 'MOCIM']

seqs = ['apt_2', 'apt_3', 'scene_0004', 'scene_0005', 'scene_0009', 'scene_0010', 'scene_0030', 'scene_0031']

end_time =  {
    "apt_2": 17.45,
    "apt_3": 30.65,
    "scene_0004": 7.74,
    "scene_0005": 9.63,
    "scene_0009": 8.16,
    "scene_0010": 20.94,
    "scene_0030": 20.81,
    "scene_0031": 22.91
}

nrows = 3
ncols = 8

fig_all_show, fig_all_ax = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(5 * ncols, 24))

fig_all_show.subplots_adjust(hspace=0.28, wspace=0.23)
fig_all_show.subplots_adjust(left=0.05,right=0.95,top=0.9,bottom=0.1)


for i in range(len(seqs)):
    seq = seqs[i]
    print("\n", seq, "\n")
    save_plots(fig_all_ax, seq, methods, col_offset=i)



sdf_ax_ticks(fig_all_ax[0,:])
acc_ax_ticks(fig_all_ax[1,:])
com_ax_ticks(fig_all_ax[2,:])

fig_all_show.text(
    0.5, 0.05, 'Sequence time [s]', ha='center',
    fontsize=ax_label_fontsize,   fontweight='bold')

fig_all_show.text(
    0.018, 0.78, 'SDF Error [cm]', va='center',
    rotation='vertical', fontsize=ax_label_fontsize, color = "black",   fontweight='bold')

fig_all_show.text(
    0.018, 0.50, 'Mesh Accuracy [cm]', va='center',
    rotation='vertical', fontsize=ax_label_fontsize, color = "black",   fontweight='bold')

fig_all_show.text(
    0.018, 0.21,  'Mesh Completion [cm]', va='center',
    rotation='vertical', fontsize=ax_label_fontsize, color = "black",   fontweight='bold')

fig_all_show.savefig("/home/dyn/SDF/Multi-mocim/"+ f"plot.pdf")
fig_all_show.savefig("/home/dyn/SDF/Multi-mocim/"+ f"plot.png")


table = np.around(table,2)

print("apt_2 apt_3 scene_0004 scene_0005 scene_0009 scene_0010 scene_0030 scene_0031")
for k in range(3):
    for i in range(len(methods)):
        print(table[i,:,k])
    print("\n")
