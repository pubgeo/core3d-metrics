#################################################################################
# IARPA-CORE3D Blender Work

# This script takes a perspective image of an object, where the camera elevation angle, focal length, and radial distance to the object are controlled by input parameters
# View README for information on command line parameter inputs required

# Author: Erika Rashka, JHU APL, July 2020
# References:
    # diffuse_to_emissive() function (and all function dependencies): authored by Robert H. Forsman Jr. on Blender Stack exchange, source code from: https://blender.stackexchange.com/questions/79595/change-diffuse-shader-to-emission-shader-without-affecting-shader-color
    # Changes made by Erika Rashka and Shea Hagstrom

################################################################################


############################################
# Imports and functions
############################################

import bpy
import os
import math
from math import radians
from mathutils import Vector
import sys, getopt


def replace_with_emission(node, node_tree):
    new_node = node_tree.nodes.new('ShaderNodeEmission')
    connected_sockets_out = []
    sock = node.inputs[0]
    if len(sock.links)>0:
        color_link = sock.links[0].from_socket
    else:
        color_link=None
    defaults_in = sock.default_value[:]

    for sock in node.outputs:
        if len(sock.links)>0:
            connected_sockets_out.append( sock.links[0].to_socket)
        else:
            connected_sockets_out.append(None)

    new_node.location = (node.location.x, node.location.y)

    if color_link is not None:
        node_tree.links.new(new_node.inputs[0], color_link)
    new_node.inputs[0].default_value = defaults_in

    if connected_sockets_out[0] is not None:
        node_tree.links.new(connected_sockets_out[0], new_node.outputs[0])


def material_diffuse_to_emission(mat):

    doomed=[]
    if mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type=='BSDF_DIFFUSE':
                replace_with_emission(node, mat.node_tree)
                doomed.append(node)

    # wait until we are done iterating and adding before we start wrecking things
    for node in doomed:
        mat.node_tree.nodes.remove(node)


def replace_on_selected_objects():
    mats = set()
    for obj in bpy.context.scene.objects:
        if obj.select:
            for slot in obj.material_slots:
                mats.add(slot.material)

    for mat in mats:
        material_diffuse_to_emission(mat)

def replace_in_all_materials():
    for mat in bpy.data.materials:
        material_diffuse_to_emission(mat)


def diffuse_to_emissive(): #svgpath, scale_value, savepath):
    # Start with new blender file
    if False:
        replace_on_selected_objects()
    else:
        replace_in_all_materials()

def rotate_and_render(output_dir, output_file_format, rotation_steps, rotation_angle, elev_ang, radius):
    # Reference for general structure of function:
        #https://stackoverflow.com/questions/14982836/rendering-and-saving-images-through-blender-python
    #Author of below camera translation calculation using cylindrial coordinates: Erika Rashka

    #Change camera location on loop: based on number of rotation steps

    theta = (math.pi/2)-(elev_ang * (2*math.pi/360))
    phi = rotation_angle/rotation_steps * (2*math.pi/360);

    for step in range(0, rotation_steps):
        # Select camera and make active
        bpy.data.objects['Camera'].select = True
        bpy.context.scene.objects.active = bpy.data.objects['Camera']

        #Move the camera, then render the scene
        print(phi)
        print(phi*step)

        # Change camera location
        bpy.context.object.location[0] = bpy.data.objects['Empty'].location[0] + (radius * math.sin(theta) * math.cos(step*phi))         #MATH: x = radius * math.sin(theta) * math.cos(phi*step)  # x = rsin(theta)*cos(phi), or cylind x = p*cos(phi)
        bpy.context.object.location[1] = bpy.data.objects['Empty'].location[1] + (radius * math.sin(theta) * math.sin(step*phi))        #MATH: y = radius * math.sin(theta) * math.sin(phi*step)  # y = rsin(theta)*sin(phi), or cylind y = p*sin(phi)
        bpy.context.object.location[2] = bpy.data.objects['Empty'].location[2] + (radius * math.cos(theta))                           #MATH: z = radius * math.cos(theta)  # z = r*cos(theta),         or cylind z = z

        # Create filepath and render image
        bpy.context.scene.render.filepath = output_dir + (output_file_format % step)
        bpy.ops.render.render(write_still=True)


def read_in_args(argumentlist):

   path = ''
   gsd = 0.5 #default GSD
   x1 = 0.0
   x2 = 0.0
   y1 = 0.0
   y2 = 0.0
   N  = 1
   elev_ang = 360.0
   f_length = 0.0
   radius = 0.0
   test = 0.0
   z_up = True #default import without specifying +Z up


   #Options
   options = "hp:g:x:y:X:Y:z:N:e:f:r:t"

   #Long options
   long_options = ["path=","gsd=", "x=", "y=", "X=", "Y=", "z=", "N=", "e=", "f=", "r=","t="]

   try:
      opts, args = getopt.getopt(argumentlist,options,long_options)
      for opt, arg in opts:
          if opt == '-h':
              print('test.py -- -p <filepath> -g <gsd> -x <x1> -y <y1> -X <x2> -Y <y2> -z <+z up?> -N <frame#> -e <elevation angle (deg)> -f <focal length> -r <range from center of AOI>')
              sys.exit()
          elif opt in ("-p", "--path"):
              path = arg
          elif opt in ("-g", "--gsd"):
              gsd = arg
          elif opt in ("-x", "--x"):
              x1 = arg
          elif opt in ("-y", "--y"):
              y1 = arg
          elif opt in ("-X", "--X"):
              x2 = arg
          elif opt in ("-Y", "--Y"):
              y2 = arg
          elif opt in ("-z", "--z"):
              z_up = arg.lower() == 'true'  # Z_up will save as bool True if string matches
          elif opt in ("-N", "--N"):
              N = arg
          elif opt in ("-e", "--e"):
              elev_ang = arg
          elif opt in ("-f", "--f"):
              f_length = float(arg)
          elif opt in ("-r", "--r"):
              radius = arg
          elif opt in ("-t", "--t"):
              test = arg


   except getopt.error as err:
      print('test.py -- -p <filepath> -g <gsd> -x <x1> -y <y1> -X <x2> -Y <y2> -z <+z up?> -N <frame#> -e <elevation angle (deg)> -f <focal length> -r <range from center of AOI>')
      print('Please add in a filepath directory (string), GSD value (float), coordinates startpoint(x,y) and endpoint (X,Y) (float), a boolean indicator if you want to load file with +Z up loaded, the number of frames (N), the elevation angle (e), the camera focal length in mm(f), and the range(r) from the center of AOI')
      print(str(err))
      sys.exit(2)


   return str(path), float(gsd), float(x1), float(y1), float(x2), float(y2), bool(z_up), int(N), float(elev_ang), float(f_length), float(radius), float(test)


if __name__ == '__main__':


    ############################################
    # Read in command line arguments, starting after '--'
    ############################################
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

    path, GSD, x1, y1, x2, y2, z_up, N, elev_ang, f_length, radius, test = read_in_args(argv)
    if y1 != 0.0 and y2 != 0.0:
        y1 = -y1 #Ensure y inputs can be +
        y2 = -y2
    print(GSD, x1, y1, x2, y2, z_up, N, elev_ang, f_length, radius, test)

    ############################################
    # Set up Blender
    ############################################

    # Change to Blender's more modern rendering method, enables you to have sampling and geometry options
    bpy.context.scene.render.engine = 'CYCLES'

    # Optional: Change setting so that the rendering can be done on the GPU
    ############################################################
    #CyclesPreferences.compute_device_type = 'CUDA'?
    # bpy.data.scenes["Scene"].cycles.device = 'GPU'?
    # bpy.context.scene.cycles.device = 'GPU'


    ############################################
    # Part 1: Load OBJ and Create Camera
    ############################################

    # Delete all objects in scene (lamp and cube)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Import .obj file
    if z_up:
        imported_object = bpy.ops.import_scene.obj(filepath=path, axis_forward='Y', axis_up='Z')
    else:
        imported_object = bpy.ops.import_scene.obj(filepath=path)

    obj_object = bpy.context.selected_objects[0]


    # Select .obj file in the scene
    bpy.ops.object.select_all(action='SELECT')
    obj_file = bpy.context.selected_objects[0]  # Choose object file
    bpy.context.scene.objects.active = obj_file  # Make object file active

    # Get information on bounding box of object
    if x1 == x2 == y2 == y1 == 0.0:
        print('Taking image of entire region')
        x_dim, y_dim, z_dim = bpy.context.active_object.dimensions
        x_size = math.ceil(x_dim)
        y_size = math.ceil(y_dim)
        max_dim = max(x_dim, y_dim, z_dim)
        str_loc = '_loc_default'
    else:
        print('Taking image of specified region')
        x_dim, y_dim, z_dim = bpy.context.active_object.dimensions
        x_dim = abs(x1-x2)
        y_dim = abs(y1-y2)
        x_size = math.ceil(x_dim)
        y_size = math.ceil(y_dim)
        x_center = min(x1, x2) + x_dim/2;
        y_center = min(y1, y2) + y_dim/2;
        max_dim = max(x_dim, y_dim, z_dim)
        str_loc = '_loc_x1y1x2y2_' + str(x1) + '_' + str(y1) + '_' + str(x2) + '_'+ str(y2)



    # Ensure object rotaton is at 0,0,0 or change if desired
    bpy.context.object.rotation_euler[0] = 0  # x -90 * (math.pi / 180)
    bpy.context.object.rotation_euler[1] = 0  # y
    bpy.context.object.rotation_euler[2] = 0  # z


    # Get coordinates of the center of the object bounding box
    # Code snippet below for bounding box center coordinates pulled from:
        # https://blender.stackexchange.com/questions/62040/get-center-of-geometry-of-an-object?noredirect=1&lq=1
    # (Below bounding box calculation works on ACTIVE object)
    o = bpy.data.objects['combined'] #bpy.context.object
    local_bbox_center = 0.125 * sum((Vector(b) for b in o.bound_box), Vector())
    global_bbox_center = o.matrix_world * local_bbox_center
    print(global_bbox_center)

    # Create camera based on coordinate system setup
    if x1 == x2 == y2 == y1 == 0.0:
        if z_up: # Correct for ARA D4
            bpy.ops.object.camera_add(view_align=True, enter_editmode=False, location=(global_bbox_center[0], global_bbox_center[1], z_dim + z_dim/10),
                                      rotation=(0, 0, 0), layers=(
                    True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                    False, False, False, False))
        else: #Correct for other ARAs
            bpy.ops.object.camera_add(view_align=True, enter_editmode=False, location=(global_bbox_center[0], global_bbox_center[2], (-1 * global_bbox_center[1]) + max_dim + max_dim / 10),
                                      rotation=(0, 0, 0), layers=(
                    True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                    False,False, False, False, False))
    else:
        if z_up: # Correct for ARA D4
            bpy.ops.object.camera_add(view_align=True, enter_editmode=False, location=(x_center, y_center, z_dim + z_dim/10),
                                      rotation=(0, 0, 0), layers=(
                    True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                    False, False, False, False))
        else:
            bpy.ops.object.camera_add(view_align=True, enter_editmode=False, location=(x_center, y_center, (-1 * global_bbox_center[1]) + max_dim + max_dim / 10),
                                      rotation=(0, 0, 0), layers=(
                    True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                    False,False, False, False, False))


    ############################################
    # Part 2: Adjust Camera as needed
    ############################################

    #Code to adjust camera that works from the command line:
        #Code snippet below for changing 3D view perspective pulled from:
            #https://blenderartists.org/t/incorrect-context-error-when-calling-bpy-ops-view3d-object-as-camera/618776
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.region_3d.view_perspective = 'CAMERA'
            break

    #Change to ortho
    bpy.context.object.data.type = 'PERSP' #''PERSP'

    #Ensure camera rotation is 0,0,0
    bpy.context.object.rotation_euler[0] = 0
    bpy.context.object.rotation_euler[1] = 0
    bpy.context.object.rotation_euler[2] = 0

    #Change focal length of camera
    bpy.context.object.data.lens_unit = 'MILLIMETERS'
    bpy.context.object.data.lens = f_length

    #Optional: Change field of view (FOV) of camera instead of focal length
    #bpy.context.object.data.lens_unit = 'FOV'
    #bpy.context.object.data.angle = 0.977384

    # Adjust camera clipping
    bpy.context.object.data.clip_end = 10000
    bpy.context.object.data.clip_start = 0.1

    #Specify labels
    scene = bpy.context.scene
    scene.camera = bpy.data.objects['Camera']

    #Change render resolution
    ResX = int(math.ceil(float(1/GSD) * x_size))
    ResY = int(math.ceil(y_size/x_size * ResX)) #math.ceil(float(1/GSD) * y_size)
    bpy.context.scene.render.resolution_x = ResX
    bpy.context.scene.render.resolution_y = ResY
    scene.render.resolution_percentage = 100
    print('Resolution of image: (Y,X)')
    print(ResY)
    print(ResX)


    #Change sampling to reduce rendering time, ideal sample # ~between 8-16
    bpy.context.scene.cycles.samples = 8

    if z_up:
        #Create an empty at combined.obj location
        bpy.ops.object.add(type='EMPTY')
        bpy.context.object.location[0] = global_bbox_center[0]
        bpy.context.object.location[1] = global_bbox_center[1] #global_bbox_center[2]
        bpy.context.object.location[2] = 0 #(-1 * global_bbox_center[1])
    else:
        #Create an empty at combined.obj location
        bpy.ops.object.add(type='EMPTY')
        bpy.context.object.location[0] = global_bbox_center[0]
        bpy.context.object.location[1] = global_bbox_center[2] #global_bbox_center[2]
        bpy.context.object.location[2] = 0 #(-1 * global_bbox_center[1])


    #Select camera and make active
    bpy.data.objects['Camera'].select = True
    bpy.context.scene.objects.active = bpy.data.objects['Camera']

    #Add locked track to camera (use empty)
    #See example of how-to here: https://www.youtube.com/watch?v=ageV_llb0Hk
    bpy.ops.object.constraint_add(type='TRACK_TO')
    bpy.context.object.constraints["Track To"].target = bpy.data.objects["Empty"]
    bpy.context.object.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    bpy.context.object.constraints["Track To"].up_axis = 'UP_Y'


    ############################################
    # Part 3: Build scene and save image rendering
    ############################################

    # Run emissive code
    diffuse_to_emissive()

    #Perspective RENDER Image

    imgname = 'persp_image_' + str(ResX) + '_' + str(ResY) + '_gsd'+ str(GSD)+ str_loc+ '_z_' + str(z_up) + '_N_' + str(N) + '_elev_' + str(elev_ang) + '_flen_' + str(f_length) + '_radius_' + str(radius) + '_'
    file_dir = '\\'.join(path.split('\\')[0:-1])
    savepath = file_dir + '\\' + imgname
    rotate_and_render(savepath, 'render%d.png', int(N), 360.0, elev_ang, radius)



