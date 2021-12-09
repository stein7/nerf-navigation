
import numpy as np

import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


pink    = '#EC6779'
green   = '#288737'
blue    = '#4678a8'
yellow  = '#CBBA4E'
cyan    = '#6BCCeC'
magenta = '#A83676'
orange  = '#ff9e14'


nameA = "random_stonehenge_"
nameB = "random_rrt_stonehenge_"
nameC = "random_minsnap_stonehenge_"

prefix_names = [nameA, nameB, nameC]
pretty_names = ["Ours", "RRT", "Min Snap"]

# fig = plt.figure(figsize=plt.figaspect(8/11))
# fig = plt.figure(figsize=(11,8))
# ax = fig.add_subplot(1, 1, 1)

def get_latest_train(name):
    save_number = 0
    while True:
        filepath = 'experiments/' + name + '/train/' +str(save_number)+".json"
        # print(filepath)

        if not os.path.isfile(filepath):
            if save_number == 0:
                assert False, "can't find any trajectory"
            save_number -= 1

            filepath = 'experiments/' + name + '/train/' +str(save_number)+".json"
            return filepath

        save_number+=1

def mean(array):
    return sum(array)/len(array)

def collision_and_control_cost(name):
    data = json.load(open(get_latest_train(name)))

    # last two costs are an outlier (minsnap gives very high cost so
    # in interest of fairness we cut them off
    total_cost = data['total_cost'][:-2]
    colision_cost = data['colision_loss'][:-2]

    print(name, "total", mean(total_cost))
    print(name, "colision", mean(colision_cost))

    return mean(colision_cost)*1e3, mean(total_cost) -  mean(colision_cost )

collision = []
control = [] 
failure_rate = [0.1, 0.2, 0.5]

handles =[]
# ax_twin = ax.twinx()
for prefix in prefix_names:
    # data = json.load(open(get_latest_train(name)))


    cols = [collision_and_control_cost(prefix + str(n))[0] for n in range(0,10)]
    ctrls =[collision_and_control_cost(prefix + str(n))[1] for n in range(0,10)]

    # print(name, "total", mean(data['total_cost']))
    # print(name, "colision", mean(data['colision_loss']))

    collision.append(mean(cols))
    control.append(mean(ctrls))

    # ax.plot(data['total_cost'], c =color, linestyle='--', linewidth =3)
    # ax_twin.plot(data['colision_loss'], c=color, linewidth =3)

    # patch = mpatches.Patch(color=color, label = name)
    # handles.append(patch)

fig = plt.figure(figsize=(11,8))
left_ax = fig.add_subplot(1, 1, 1)
right_ax = left_ax.twinx()

legend_elements = []

ind = np.arange(len(control))
width = 0.25       
left_ax.bar(ind - width/2, collision, width, label='Collision', color=cyan, log=True)
legend_elements.append( Patch(facecolor=cyan, label='Collision') )

left_ax.bar(ind + width/2, control, width, label='Control', color=magenta, log=True)
legend_elements.append( Patch(facecolor=magenta, label='Control') )

right_ax.set_ylim( 0, 1)
right_ax.bar(ind + width * 3/2, failure_rate, width, label='Failure Rate', color=orange)
legend_elements.append( Patch(facecolor=orange, label='Failure Rate') )

left_ax.set_ylabel('Control and NeRF Collision Cost', fontsize=30)
right_ax.set_ylabel('Failure Rate', fontsize=30)
# plt.title('Planner Comparision', fontsize=30)

plt.xticks(ind + width / 2, tuple(pretty_names) , fontsize=30)
left_ax.set_xticklabels(tuple(pretty_names), rotation=0, fontsize=30)

# left_ax.legend(loc='best')
# right_ax.legend(loc='best')

plt.legend(handles=legend_elements, prop={"size":20} , loc=2)


plt.show()






