import bpy
import json
from mathutils import Matrix

import bmesh
import os

object_to_copy = bpy.data.objects["Boolean"]


def load_training(experiment_name):
    collection = bpy.data.collections.new("PLS remove")
    bpy.context.scene.collection.children.link(collection)

    save = 0
    while True:
        filepath = bpy.path.abspath('//../experiments/' + experiment_name + '/train/') +str(save)+".json"
        out_filepath = bpy.path.abspath('//../experiments/' + experiment_name + '/train_vol/') +str(save)+".json"
        if not os.path.isfile(filepath):
            if save == 0:
                bpy.context.window_manager.popup_menu(lambda s,c:s.layout.label(text=""), title = "can't find first file", icon = "ERROR")
            break

        print("LOADING SAVE #"+str(save))
        
        
        with open(filepath) as f:
            data = json.load(f)
            
            vols = []
            for i in range(len(data['poses'])):
                name = "delete_me_"+str(i)

                new_object = object_to_copy.copy()
                new_object.data = object_to_copy.data.copy()
                new_object.data.name = "REMOE"
                
                collection.objects.link(new_object)

                new_object.name = name 

                pose = data['poses'][i]
                new_object.matrix_world = Matrix(pose)

                bpy.context.view_layer.objects.active = new_object
                bpy.ops.object.modifier_apply(modifier="Boolean")
                
                bm = bmesh.new()
                bm.from_mesh(new_object.data)
                
                vol = bm.calc_volume()
                print(i, vol)
                vols.append(vol)
                
            data['mesh_colision'] = vols
            with open(out_filepath,"w+") as out:
                json.dump(data, out)
                
#                bpy.data.objects.remove(new_object)
#                bpy.data.meshes.remove(new_object.data)
                
                
                
#        break
                
        
        bpy.context.view_layer.update() 
        save += 1
        
experiment_name = "playground_slide"

load_training(experiment_name)
