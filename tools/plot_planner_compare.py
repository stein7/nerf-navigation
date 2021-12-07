
import numpy as np

import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


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
colors = [pink,blue, green]

# fig = plt.figure(figsize=plt.figaspect(8/11))
fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(1, 1, 1)


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

handles =[]
ax_twin = ax.twinx()
for name,color in zip(names, colors):
    data = json.load(open(get_latest_train(name)))

    print(name, "total", mean(data['total_cost']))
    print(name, "colision", mean(data['colision_loss']))
    ax.plot(data['total_cost'], c =color, linestyle='--', linewidth =3)
    ax_twin.plot(data['colision_loss'], c=color, linewidth =3)

    patch = mpatches.Patch(color=color, label = name)
    handles.append(patch)


ax.set_ylabel("total loss", fontsize=20)
ax_twin.set_ylabel("NeRF collision", fontsize=20)
ax.set_xlabel("Trajectory time", fontsize=20)

# handles, labels = ax.get_legend_handles_labels()
# print(labels)

legend1 =  plt.legend(handles=handles, prop={"size":16} , loc=1)

# handels2 = []
# line1 = Line2D([0], [0], label='Mesh intersection volume', color='k', linestyle='--')
# line2 = Line2D([0], [0], label='NeRf collision loss', color='k', linestyle='-')
# handels2.extend([line1, line2])

# plt.legend(handles=handels2, prop={"size":16} , loc=2)

fig.add_artist(legend1)


# ax.legend()

plt.show()






