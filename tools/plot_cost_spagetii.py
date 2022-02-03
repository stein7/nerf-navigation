
import numpy as np

import json
import pathlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches



# name = "playground_mpc_4"
name = "playground_mpc_2"
name = "stonehenge_mpc"


# frames = [0, 5, 49]
# colors = ['r','b','g']

# fig = plt.figure(figsize=plt.figaspect(8/11))
fig = plt.figure(figsize=(11,8))
ax_colision = fig.add_subplot(2, 1, 1)

ax_control = fig.add_subplot(2, 1, 2)

handles =[]
# ax_twin = ax.twinx()

pink    = '#EC6779'
green   = '#288737'
blue    = '#4678a8'
yellow  = '#CBBA4E'
cyan    = '#6BCCeC'
magenta = '#A83676'


end_frame = 0
while 1:
    if pathlib.Path("experiments/" + name +"/mpc/"+str(end_frame)+".json").exists():
        end_frame +=1
    else:
        break


print(end_frame)

taken_cost_colision = []
taken_cost_control = []

for frame in range(end_frame):
    data = json.load(open("experiments/" + name +"/mpc/"+str(frame)+".json"))

    # color = "k"
    # color = {0:pink, 8:blue, 14:green}.get(frame, "k")

    color = {0:pink, 10:blue, 18:green}.get(frame, "k")

    #
    colision_cost = data['colision_loss']
    control_cost  = [x-y for x,y in zip(data['total_cost'], data['colision_loss'])]

    ax_colision.plot(frame + np.arange(len(colision_cost)), colision_cost, c =color, 
            alpha = 0.2 if color =="k" else 0.8,
            linewidth= 1 if color =='k' else 2)

    ax_control.plot(frame + np.arange(len(colision_cost)), colision_cost, c =color, 
            alpha = 0.2 if color =="k" else 0.8,
            linewidth= 1 if color =='k' else 2)


    taken_cost_colision.append(colision_cost[0])
    taken_cost_control.append(control_cost[0])

    # ax_twin.plot(data['colision_loss'], c=color)

    # patch = mpatches.Patch(color=color, label = "iter_"+str(frame))
    # handles.append(patch)
ax_colision.plot(taken_cost_colision, c ="k", alpha = 1, linewidth=4)
ax_control.plot(taken_cost_control, c ="k", alpha = 1, linewidth=4)


ax_colision.set_ylabel("Collision Cost", fontsize=20)
ax_control.set_ylabel("Control Cost", fontsize=20)
# ax_twin.set_ylabel("NeRF collision", fontsize=16)

ax_colision.set_xlabel("Trajectory time", fontsize=20)
ax_control.set_xlabel("Trajectory time", fontsize=20)

# handles, labels = ax.get_legend_handles_labels()
# print(labels)

handles.append(  Line2D([0], [0], label='Cost of executed trajectory', color='k', linewidth=4)  )
handles.append(  Line2D([0], [0], label='Cost of plan at t = 0', color=pink, linewidth=2)  )
handles.append(  Line2D([0], [0], label='Cost of plan at t = 8', color=blue, linewidth=2)  )
handles.append(  Line2D([0], [0], label='Cost of plan at t = 14', color=green, linewidth=2  ))

# handles.extend([line1, line2])
plt.legend(handles=handles, prop={"size":16})

# ax.legend()

plt.show()




