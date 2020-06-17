#########################################
# IARPA-CORE3D Blender Work

# This script takes a top down image an area of interest, fitting the camera to the bounding box of the object in the OBJ file
#See TO DOs below to modify file paths per your system

#Ideal: Perspective camera at orbiting oblique viewpoints
#Parameters: # of frames (e.g., 4 would give cardinal directions), elevation angle (90 degrees would be looking straight down), camera focal length, pixel width/height
#Given the model bounds it should be relatively easy to compute the camera distance needed to encompass the full model in the render.


# Author: Erika Rashka, JHU APL, Jan 2020, Update May 2020
#########################################


#############################################################################################################################33
#Testing script
###########################################################################################################################

# Start with new blender file
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

    #print( defaults_in )

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

def rotate_and_render(output_dir, output_file_format, rotation_steps=4, rotation_angle=360.0):
    # Reference for some source code in function:
    # https://stackoverflow.com/questions/14982836/rendering-and-saving-images-through-blender-python/15906299
    #https://stackoverflow.com/questions/14982836/rendering-and-saving-images-through-blender-python

    #Create an empty that is located where the object of interest is located
    #bpy.ops.object.select_all(action='SELECT')
    aoi = bpy.data.objects['Camera'] #bpy.context.selected_objects[1]
    #^^^ Might have to specify bpy.data.objects['combined.obj']
    #bpy.ops.object.add(type='EMPTY')
    origin = aoi #bpy.context.object

    #Constrain camera to look at empty
    bpy.context.space_data.context = 'CONSTRAINT'
    bpy.ops.object.constraint_add(type='LOCKED_TRACK')
    bpy.context.object.constraints["Locked Track"].target = bpy.data.objects["Empty"]

    for step in range(0, rotation_steps):
        origin.rotation_euler[2] = radians(step * (rotation_angle / rotation_steps))
        bpy.context.scene.render.filepath = output_dir + (output_file_format % step)
        bpy.ops.render.render(write_still=True)

    #ways of rotating camera- camera MUST be selected
    #See: https://docs.blender.org/manual/en/2.79/editors/3dview/navigate/camera_view.html

    #rotate up and down at fixed top position
    bpy.ops.transform.trackball(value=(0.61, 0.14), mirror=False, proportional='DISABLED',
                                proportional_edit_falloff='SMOOTH', proportional_size=1)
    #rotate side to side horiz- only allowing translation in Y
    bpy.ops.transform.translate(value=(0, -3.15546, 0), constraint_axis=(False, True, False),
                                constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                                proportional_edit_falloff='SMOOTH', proportional_size=1)
    #rotate up and down veritically- only adjusting translation in X
    bpy.ops.transform.translate(value=(-8.29144, 0, 0), constraint_axis=(True, False, False),
                                constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                                proportional_edit_falloff='SMOOTH', proportional_size=1)
    #lock x axis, and rotate view
    bpy.ops.transform.rotate(value=0.285643, axis=(1, 0, 0), constraint_axis=(True, False, False),
                             constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                             proportional_edit_falloff='SMOOTH', proportional_size=1)


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
   z_up = True #default import without specifying +Z up

   # Remove 1st argument from the
   # list of command line arguments
   #argv[argv.index("--") + 1:] #sys.argv[1:]

   #Options
   options = "hp:g:x:y:X:Y:z:N:e:f:"

   #Long options
   long_options = ["path=","gsd=", "x=", "y=", "X=", "Y=", "z=", "N=", "e=", "f="]

   try:
      opts, args = getopt.getopt(argumentlist,options,long_options)
      for opt, arg in opts:
          if opt == '-h':
              print(
                  'test.py -- -p <filepath> -g <gsd> -x <x1> -y <y1> -X <x2> -Y <y2> -z <+z up?> -N <frame#> -e <elevation angle (deg)> -f <focal length>')
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
              f_length = arg

   except getopt.error as err:
      print('test.py -- -p <filepath> -g <gsd> -x <x1> -y <y1> -X <x2> -Y <y2> -z <+z up?> -N <frame#> -e <elevation angle (deg)> -f <focal length>')
      print('Please add in a filepath directory (string), GSD value (float), coordinates startpoint(x,y) and endpoint (X,Y) (float), a boolean indicator if you want to load file with +Z up loaded (1 = yes, 0 = no), the number of frames (N, rotation around z axis), the elevation angle (e), and the camera focal length (f)')
      print(str(err))
      sys.exit(2)


   return str(path), float(gsd), float(x1), float(y1), float(x2), float(y2), bool(z_up), int(N), float(elev_ang), float(f_length)


if __name__ == '__main__':


    ####################
    # Get command line arguments for path and image resolution
    ####################
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    # ResX = int(argv[0])
    # ResY = int(argv[1])
    # path = str(argv[2])
    path, GSD, x1, y1, x2, y2, z_up, N, elev_ang, f_length = read_in_args(argv)
    y1 = -y1; #Ensure y inputs can be +
    y2 = -y2;
    print(GSD, x1, y1, x2, y2, z_up, N, elev_ang, f_length)
    #
    # #arguments: path, gsd, up_dir, front_dir, x1, y1, x2,y2
    #
    #
    # ####################
    # # Set up Blender
    # ####################
    #
    # # Change to Blender's more modern rendering method, enables you to have sampling and geometry options
    # bpy.context.scene.render.engine = 'CYCLES'
    #
    # ####################
    # # Change setting so that the rendering can be done on the GPU
    # #CyclesPreferences.compute_device_type = CUDA?
    # # bpy.data.scenes["Scene"].cycles.device = GPU?
    # # bpy.context.scene.cycles.device = 'GPU'
    #
    # ####################
    #
    #
    # ####################
    # # Step 1: Load OBJ and Create CAMERA
    # ####################
    #
    # # Delete all objects in scene (lamp and cube)
    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.object.delete(use_global=False)
    #
    # # Import .obj file EX:ARA-D1
    # file_loc = path + 'combined.obj'
    # if z_up:
    #     imported_object = bpy.ops.import_scene.obj(filepath=file_loc, axis_forward='Y', axis_up='Z')
    # else:
    #     imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
    #
    # obj_object = bpy.context.selected_objects[0]
    #
    # # Select all objects in the scene to rotate
    # bpy.ops.object.select_all(action='SELECT')
    # obj_file = bpy.context.selected_objects[0]  # Choose object file
    # bpy.context.scene.objects.active = obj_file  # Make object file active
    #
    # if x1 == x2 == y2 == y1 == 0.0:
    #     print('Taking image of entire region')
    #     x_dim, y_dim, z_dim = bpy.context.active_object.dimensions
    #     x_size = math.ceil(x_dim)
    #     y_size = math.ceil(y_dim)
    #     max_dim = max(x_dim, y_dim, z_dim)
    #     str_loc = '_loc_default'
    # else:
    #     print('Taking image of specified region')
    #     x_dim, y_dim, z_dim = bpy.context.active_object.dimensions
    #     x_dim = abs(x1-x2)
    #     y_dim = abs(y1-y2)
    #     x_size = math.ceil(x_dim)
    #     y_size = math.ceil(y_dim)
    #     x_center = min(x1, x2) + x_dim/2;
    #     y_center = min(y1, y2) + y_dim/2;
    #     max_dim = max(x_dim, y_dim, z_dim)
    #     str_loc = '_loc_x1y1x2y2_' + str(x1) + '_' + str(y1) + '_' + str(x2) + '_'+ str(y2)
    #
    #
    #
    # # Rotate object in scene: Keep rotation at 000 if keeping 0 2 1 ordering of global bounding box center below
    # #TO DO configurable rotations
    # bpy.context.object.rotation_euler[0] = 0  # x -90 * (math.pi / 180)
    # bpy.context.object.rotation_euler[1] = 0  # y
    # bpy.context.object.rotation_euler[2] = 0  # z
    #
    #
    # #Create camera - was previously 0 2 1
    # # Get coordinates of active object (OBJ file)
    # # Below bounding box calculation works on ACTIVE object
    # o = bpy.context.object
    # local_bbox_center = 0.125 * sum((Vector(b) for b in o.bound_box), Vector())
    # global_bbox_center = o.matrix_world * local_bbox_center
    # print(global_bbox_center)
    #
    #
    # if x1 == x2 == y2 == y1 == 0.0:
    #     if z_up: # Correct for ARA D4
    #         bpy.ops.object.camera_add(view_align=True, enter_editmode=False, location=(global_bbox_center[0], global_bbox_center[1], z_dim + z_dim/10),
    #                                   rotation=(0, 0, 0), layers=(
    #                 True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
    #                 False, False, False, False))
    #     else: #Correct for other ARAs
    #         bpy.ops.object.camera_add(view_align=True, enter_editmode=False, location=(global_bbox_center[0], global_bbox_center[2], (-1 * global_bbox_center[1]) + max_dim + max_dim / 10),
    #                                   rotation=(0, 0, 0), layers=(
    #                 True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
    #                 False,False, False, False, False))
    # else:
    #     if z_up: # Correct for ARA D4
    #         bpy.ops.object.camera_add(view_align=True, enter_editmode=False, location=(x_center, y_center, z_dim + z_dim/10),
    #                                   rotation=(0, 0, 0), layers=(
    #                 True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
    #                 False, False, False, False))
    #     else:
    #         bpy.ops.object.camera_add(view_align=True, enter_editmode=False, location=(x_center, y_center, (-1 * global_bbox_center[1]) + max_dim + max_dim / 10),
    #                                   rotation=(0, 0, 0), layers=(
    #                 True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
    #                 False,False, False, False, False))
    #
    #
    # ####################
    # # Step 2: Adjust CAMERA as needed
    # ####################
    #
    # #Set active object as active camera, show view of camera and set render (Ctrl Numpad0)
    #     #https://docs.blender.org/manual/en/latest/editors/3dview/navigate/camera_view.html
    #
    # #Adjust camera: (BEST method) Works in Blender AND command line
    #     #Reference: https://blenderartists.org/t/incorrect-context-error-when-calling-bpy-ops-view3d-object-as-camera/618776
    #
    # for area in bpy.context.screen.areas:
    #     if area.type == 'VIEW_3D':
    #         area.spaces.active.region_3d.view_perspective = 'CAMERA'
    #         break
    #
    # #Change to orthographic view (no scaling)
    # #https://blender.stackexchange.com/questions/7686/how-to-set-cameras-type-to-orthographic-with-python-scripting
    # bpy.context.object.data.type = 'PERSP' #''PERSP'
    #
    #
    # #Change focal length of camera
    # bpy.context.object.data.lens_unit = 'MILLIMETERS'
    # bpy.context.object.data.lens = 35
    #
    # #OR Change field of view of camera
    # bpy.context.object.data.lens_unit = 'FOV'
    # bpy.context.object.data.angle = 0.872665 #radians
    #
    #
    # # Adjust camera clipping - per Shea's recommendation
    # bpy.context.object.data.clip_end = 10000 #max_dim * 10 #10,000 meters
    # bpy.context.object.data.clip_start = 0.1 #0.001 #Don't really want any clipping close up
    #
    # #Specify labels
    # scene = bpy.context.scene
    # scene.camera = bpy.data.objects['Camera']
    #
    # #Change render resolution
    # ResX = int(math.ceil(float(1/GSD) * x_size))
    # ResY = int(math.ceil(y_size/x_size * ResX)) #math.ceil(float(1/GSD) * y_size)
    # bpy.context.scene.render.resolution_x = ResX
    # bpy.context.scene.render.resolution_y = ResY
    # scene.render.resolution_percentage = 100
    # print('Resolution of image: (Y,X)')
    # print(ResY)
    # print(ResX)
    #
    #
    # #Change sampling, sample # to be between 8-16
    # bpy.context.scene.cycles.samples = 8
    #
    #
    # # Run emissive code: Set to replace on all materials, added "FALSE" within script
    # # bpy.ops.object.select_all(action='SELECT')
    # # obj_file = bpy.context.selected_objects[0]  # Choose object file
    # # bpy.context.scene.objects.active = obj_file  # Make object file active
    # diffuse_to_emissive()
    #
    # print(x_size)
    # print(y_size)
    # print(GSD)
    # print(math.ceil(float(1/GSD) * x_size))
    #
    # #Create an empty at combined.obj location
    # bpy.ops.object.add(type='EMPTY')
    # bpy.context.object.location[0] = global_bbox_center[0]
    # bpy.context.object.location[1] = global_bbox_center[2]
    # bpy.context.object.location[2] = (-1 * global_bbox_center[1])
    #
    # #Select camera and make active
    # bpy.data.objects['Camera'].select = True
    # bpy.context.scene.objects.active = bpy.data.objects['Camera']
    #
    # #Add locked track to camera (use empty)
    # #Source: https://www.youtube.com/watch?v=ageV_llb0Hk
    # bpy.ops.object.constraint_add(type='TRACK_TO')
    # bpy.context.object.constraints["Track To"].target = bpy.data.objects["Empty"]
    # bpy.context.object.constraints["Track To"].track_axis = 'TRACK_Z'
    # bpy.context.object.constraints["Track To"].up_axis = 'UP_Z'
    #
    # ############
    # # Perspective RENDER Image
    # ##############
    # imgname = 'persp_image_' + str(ResX) + '_' + str(ResY) + '_gsd'+ str(GSD)+ str_loc+ '_z_' + str(z_up) + '_'
    # rotate_and_render(path+imgname, 'render%d.png')
    #
    #
    # ############
    # # Orthographic RENDER Image
    # ##############
    # # bpy.context.scene.render.filepath = path + imgname
    # # #bpy.context.scene.render.filepath = 'C:/Users/rashkek1/Documents/Projects/IARPA_CORE3D/ARA-D4/Buildings/overhead_testing_image.jpg'
    # # bpy.ops.render.render(write_still=True)
    # #######
    #
    #
