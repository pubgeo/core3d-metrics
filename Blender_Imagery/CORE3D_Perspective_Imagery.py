#################################################################################
# IARPA-CORE3D Blender Work

# This script takes a perspective image of an object, where the camera elevation angle, focal length, and radial distance to the object are controlled by input parameters
# View README for information on command line arguments required

# Dependency: pip install bpy-cuda && bpy_post_install

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
from math import radians, degrees


def render_on_gpu():
    scene = bpy.data.scenes["Scene"]
    bpy.context.scene.cycles.device = 'GPU'
    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'
        scene.render.resolution_percentage = 100 
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    for devices in bpy.context.preferences.addons['cycles'].preferences.get_devices():
        for d in devices:
            d.use = True
            if d.type == 'CPU':
                d.use = False

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
            if (node.type=='BSDF_PRINCIPLED') or (node.type=='BSDF_DIFFUSE'):
                replace_with_emission(node, mat.node_tree)
                doomed.append(node)
    for node in doomed:
        mat.node_tree.nodes.remove(node)

def replace_on_selected_objects():
    mats = set()
    for obj in bpy.context.scene.objects:
        if obj.select_get():
            for slot in obj.material_slots:
                mats.add(slot.material)
    for mat in mats:
        material_diffuse_to_emission(mat)

def replace_in_all_materials():
    for mat in bpy.data.materials:
        material_diffuse_to_emission(mat)

def diffuse_to_emissive():
    if False:
        replace_on_selected_objects()
    else:
        replace_in_all_materials()

def rotate_and_render(cam, empty, output_dir, output_file_format, rotation_steps, rotation_angle, elev_ang, radius):
    theta = (math.pi/2)- radians(elev_ang) #degrees off nadir
    phi = radians(rotation_angle)/ rotation_steps #azimuth step

    #if N = 4, this creates the N S E and W renders
    for step in range(0, rotation_steps):
        azimuth = step*phi
        cam.location[0] = empty.location[0] + (radius * math.sin(theta) * math.cos(azimuth))
        cam.location[1] = empty.location[1] + (radius * math.sin(theta) * math.sin(azimuth))
        cam.location[2] = empty.location[2] + (radius * math.cos(theta))
        bpy.context.scene.render.filepath = output_dir + (output_file_format % step)
        bpy.ops.render.render(write_still=True, use_viewport=True)
    #Nadir render
    cam.location[0] = empty.location[0]
    cam.location[1] = empty.location[1]
    cam.location[2] = empty.location[2] + radius
    bpy.context.scene.render.filepath = output_dir + (output_file_format % rotation_steps)
    bpy.ops.render.render(write_still=True, use_viewport=True)

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
   options = "hp:g:x:y:X:Y:z:N:e:f:r:"

   #Long options
   long_options = ["path=","gsd=", "x=", "y=", "X=", "Y=", "z=", "N=", "e=", "f=", "r="]

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


   except getopt.error as err:
      print('test.py -- -p <filepath> -g <gsd> -x <x1> -y <y1> -X <x2> -Y <y2> -z <+z up?> -N <frame#> -e <elevation angle (deg)> -f <focal length> -r <range from center of AOI>')
      print('Please add in a filepath directory (string), GSD value (float), coordinates startpoint(x,y) and endpoint (X,Y) (float), a boolean indicator if you want to load file with +Z up loaded, the number of frames (N), the elevation angle (e), the camera focal length in mm(f), and the range(r) from the center of AOI')
      print(str(err))
      sys.exit(2)


   return str(path), float(gsd), float(x1), float(y1), float(x2), float(y2), bool(z_up), int(N), float(elev_ang), float(f_length), float(radius), float(test)


if __name__ == '__main__':

    #argv = sys.argv[1:]
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    total = len(sys.argv)
    cmdargs = str(sys.argv)
    print("The total numbers of args passed to the script: %d " % total)
    print("Args list: %s " % cmdargs)

    # redirect stdout to stderr
    # in command line, pipe stdout to nul: python name.py 1> nul
    # the rendering engine writes a lot of status messages to stdout we don't want to see
    sys.stdout = sys.stderr

    path, gsd, x1, y1, x2, y2, z_up, N, elev_ang, f_length, radius, test = read_in_args(argv)
    if y1 != 0.0 and y2 != 0.0:
        y1 = -y1 #Ensure y inputs can be +
        y2 = -y2
    print(gsd, x1, y1, x2, y2, z_up, N, elev_ang, f_length, radius, test)
    print(path)

    # delete all objects in scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    #Create file_list to be aware of all potential models in path directory
    file_list = []
    check = os.path.isdir(path)

    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".obj"): #OBJ model
                file_list.append(file_path)

            elif file.endswith(".glb"): #glTF 2.0 model (2.0 only supported by Blender)
                file_list.append(file_path)

            elif file.endswith(".dae"): #Collada tile
                file_list.append(file_path)

    num_models = len(file_list)
    print("Number of models found: "+ str(num_models))

    for file_path in file_list: #[:40]:
        # load models of interest
        if file_path.find('.obj') > 0:
            print('loading OBJ model...')
            if z_up:
                bpy.ops.import_scene.obj(filepath=file_path, axis_forward='Y', axis_up='Z')
            else:
                bpy.ops.import_scene.obj(filepath=file_path)
        elif file_path.find('.glb') > 0:
            # only supports glTF 2.0
            print('loading glTF 2.0 model...')
            bpy.ops.import_scene.gltf(filepath=file_path)
        elif file_path.find('.dae') > 0:
            print('loading Collada model...')
            bpy.ops.wm.collada_import(filepath=file_path)
            # if heavy importing needed- option below to save .blend file after each tile is loaded
            bpy.ops.wm.save_as_mainfile(filepath= path + "all_collada_tiles.blend")

    # define the model as the currently selected object (last loaded part)
    model = bpy.context.selected_objects[0]

    # define image size and factor for scaling focal length to match
    pixels = int(max(model.dimensions) / gsd)
    print('pixels')
    print(pixels)
    half_width_meters = (pixels / 2) * gsd * 1.1
    lens_factor = radius / half_width_meters

    # force object rotation to zero
    model.rotation_euler[0] = 0
    model.rotation_euler[1] = 0
    model.rotation_euler[2] = 0

    # Get coordinates of the center of the object bounding box
    # Code snippet below for bounding box center coordinates pulled from:
    # https://blender.stackexchange.com/questions/62040/get-center-of-geometry-of-an-object?noredirect=1&lq=1
    local_bbox_center = 0.125 * sum((Vector(b) for b in model.bound_box), Vector())
    global_bbox_center = model.matrix_world @ local_bbox_center
    print('Bounding box: ', global_bbox_center)

    #determine the global bounding box centroid that includes all models if multiple were loaded
    global_bbox_center_allmod = Vector((0.0,0.0,0.0))
    print(bpy.context.scene.objects)
    if num_models > 1:
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                print('new object')
                print(obj.name)
                local_center = 0.125 * sum((Vector(b) for b in obj.bound_box), Vector())
                global_center = model.matrix_world @ local_bbox_center
                print(global_center)
                print(obj.dimensions)
                global_bbox_center_allmod[0] += global_center[0]
                global_bbox_center_allmod[1] += global_center[1]
                global_bbox_center_allmod[2] += global_center[2]
        global_bbox_center_allmod = global_bbox_center_allmod/num_models;
        print('Bounding box for all models: ', global_bbox_center_allmod)

    #if multiple models are loaded in, adjust the coordinates
    if num_models > 1:
        center_x = global_bbox_center_allmod[0] #center coordinates that is the centroid of all model centers
        center_y = global_bbox_center_allmod[1]
        center_z = global_bbox_center_allmod[2]
    else:
        center_x = global_bbox_center[0] #center coordinates for first object selected
        center_y = global_bbox_center[1]
        center_z = global_bbox_center[2]


    # Create camera based on coordinate system setup
    if z_up:
        print('option 1 - z is up')
        camx = center_x
        camy = center_y
    else:
        print('option 2 - z is down')
        camx = center_x
        camy = center_z

    # add a camera just above the scene and rotate_and_render() changes the location
    bpy.ops.object.camera_add(align='VIEW', location=(camx, camy, 0))
    cam = bpy.context.selected_objects[0]
    print('CAMX = ', camx)
    print('CAMY = ', camy)

    #Change focal length of camera
    cam.data.type = 'PERSP'
    cam.data.lens_unit = 'MILLIMETERS'
    cam.data.lens = 1 #f_length
    cam.data.sensor_width = 2# * bpy.context.object.data.lens
    cam.data.sensor_height = 2# * bpy.context.object.data.lens
    cam.data.lens *= lens_factor
    cam.data.clip_end = radius*100
    cam.data.clip_start = radius/100
    # print('sensor width = ', cam.data.sensor_width)
    # print('sensor height = ', cam.data.sensor_height)
    # print('focal length = ', cam.data.lens)
    # print('range = ', radius)

    # define scene for rendering
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.resolution_x = pixels
    scene.render.resolution_y = pixels
    scene.render.resolution_percentage = 100
    print('Resolution of image: (Y,X): ', pixels, pixels)

    # set cycles rendering options
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 8
    render_on_gpu()

    # set blender background color
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value[:3] = (float(220)/255, float(220)/255, float(220)/255)
    bg.inputs[1].default_value = 1.0

    # add locked track to camera (use empty)
    # see example of how-to here: https://www.youtube.com/watch?v=ageV_llb0Hk
    # add empty object to use for tracking and center at center of scene with zero z
    if z_up:
        #Create an empty at combined.obj location
        bpy.ops.object.add(type='EMPTY')
        empty = bpy.context.selected_objects[0]
        empty.location[0] = center_x
        empty.location[1] = center_y #global_bbox_center[2]
        empty.location[2] = 0 #(-1 * global_bbox_center[1])
    else:
        #Create an empty at combined.obj location
        bpy.ops.object.add(type='EMPTY')
        empty = bpy.context.selected_objects[0]
        empty.location[0] = center_x
        empty.location[1] = center_z #global_bbox_center[2]
        empty.location[2] = 0 #(-1 * global_bbox_center[1])
    bpy.data.objects['Camera'].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['Camera']
    bpy.ops.object.constraint_add(type='TRACK_TO')
    bpy.context.object.constraints["Track To"].target = empty
    bpy.context.object.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    bpy.context.object.constraints["Track To"].up_axis = 'UP_Y'

    # Run emissive code to render
    diffuse_to_emissive()

    # Write the images
    imgname = 'persp_image_' + str(pixels) + '_' + str(pixels) + '_gsd'+ str(gsd)+'_z_' + str(z_up) + '_N_' + str(N) + '_elev_' + str(elev_ang) + '_flen_' + str(f_length) + '_radius_' + str(radius) + '_'
    # file_dir = '\\'.join(path.split('\\')[0:-1])
    # file_dir = '\\'.join(path.split('\\')[0:-1])
    # savepath = file_dir + '\\' + 'image_'
    file_dir = os.path.dirname(path)
    savepath = file_dir + '/rendered_images/' + imgname
    print(savepath)

    rotate_and_render(cam, empty, savepath, 'render%d.png', int(N), 360.0, elev_ang, radius)

    # Note: There is an exception upon termination in engine.free() call in cycles\__init__.py
    # This is a known issue: https://developer.blender.org/T52203


