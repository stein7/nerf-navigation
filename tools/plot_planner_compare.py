
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


nameA = "ours_stonehenge_compare1"
nameB = "rrt_stonehenge_compare1"
nameC = "minsnap_stonehenge_compare1"

names = [nameA, nameB, nameC]
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

collision = []
control = [] 

handles =[]
# ax_twin = ax.twinx()
for name in names:
    data = json.load(open(get_latest_train(name)))

    print(name, "total", mean(data['total_cost']))
    print(name, "colision", mean(data['colision_loss']))

    collision.append(mean(data['colision_loss']))
    control.append(mean(data['total_cost']) -  mean(data['colision_loss'] ))

    # ax.plot(data['total_cost'], c =color, linestyle='--', linewidth =3)
    # ax_twin.plot(data['colision_loss'], c=color, linewidth =3)

    # patch = mpatches.Patch(color=color, label = name)
    # handles.append(patch)

fig = plt.figure(figsize=(11,8))
left_ax = fig.add_subplot(1, 1, 1)
right_ax = left_ax.twinx()

legend_elements = []

ind = np.arange(len(control))
width = 0.35       
left_ax.bar(ind, collision, width, label='Collision', color=pink, log=True)
legend_elements.append( Patch(facecolor=pink, label='Collision') )

right_ax.bar(ind + width, control, width, label='Control', color=green, log=True)
legend_elements.append( Patch(facecolor=green, label='Control') )

left_ax.set_ylabel('NeRF Collision Cost', fontsize=30)
right_ax.set_ylabel('Contorl Effort', fontsize=30)
plt.title('Planner Comparision', fontsize=30)

plt.xticks(ind + width / 2, tuple(pretty_names) , fontsize=30)
left_ax.set_xticklabels(tuple(pretty_names), rotation=0, fontsize=30)

# left_ax.legend(loc='best')
# right_ax.legend(loc='best')

plt.legend(handles=legend_elements, prop={"size":20} , loc=2)


plt.show()






