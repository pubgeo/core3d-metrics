# Blender Rendered Images at Unique Perspective Angles

Saves Blender rendered images of .OBJ files based on user input of desired orthographic or perspective imagery and relevant input parameters. Orthographic overhead images can be specified with unique areas of interest (AOI) in xy coordinates. Perspective images can also be created, and additional command line arguments such as focal length, elevation angle, radial distance, and number of orbital viewpoints can be specified.

## Getting Started

These instructions will show you how to run Blender in the background and have the relevant python scripts run within Blender via the command line. There are examples provided below to demsontrate the proper command line arguments to be passed in and their required data types.

### Prerequisites

You will need Blender 2.79b (or another compatible Blender version).

Model assumptions:
	Typically when loading .OBJ files into Blender, one is required to change from a Y up coordinate system to Z up coordinate system. The .OBJ files typically treat Y as up, therefore the user has the option to have Blender recognize the coordinate system change during import. To do this, change the -z parameter to True. If this parameter is set to False, the .OBJ model will not be reoriented such that +Z is considered up, but instead the coordinate system for all translations and calculations is adjusted separately to treat +Z as up and provide reasonable imagery.  
	For more information on coordinate system requirements, see [Blender Wavefront OBJ](https://docs.blender.org/manual/en/2.80/addons/io_scene_obj.html "Blender Wavefront OBJ").
    
    
## Examples
Input arguments for camera control parameters are specified as follows:

| Input Details | Data Type |
| ------ | ------ |
| -p: Path to the .obj file location | (float) |
| -g: Desired pixel Ground Sample Distance (GSD) | (float) | 
| -x: Coordinate x1 of bounding box around AOI | (float) | 
| -y: Coordinate y1 of bounding box around AOI | (float) | 
| -X: Coordinate x2 of bounding box around AOI | (float) | 
| -Y: Coordinate y2 of bounding box around AOI | (float) | 
| -z: Boolean specifier for loading in the .obj file with coordinate system oriented +Z up (True) or no specification (False) | (bool) | 
| -N: Number of orbital locations to take images at (360/N is the rotation around the z axis), and therefore the number of images saved. Also known as frame # | (int) | 
| -e: Camera elevation angle (measured from xy plane) (deg) | (float) |
| -f: Camera focal length (mm) | (float) | 
| -r: Camera radial distance from center of object bounding box	 | (float) | 

Below examples are written for Windows, for other OS please use command line equivalent

```
#From the command line, navigate to the Blender folder on your pc
cd C:\Program Files\Blender Foundation\Blender
```

```
#Run CORE3D_Overhead_Imagery.py to save overhead orthographic images
#Note: 
#The first directory specified should be the saved location of CORE3D_Overhead_Imagery.py 
#The second directory specified should be the path where the .obj file is saved. 
#If you use -h for help, Blender -help options will be provided

blender --background --python C:\Your_Directory\CORE3D_Overhead_Imagery.py -- -p "C:/Users/Your_OBJ_Directory/example.obj" -g 1.0 -x 200.0 -y 200.0 -X 400.0 -Y 400.0 -z False
```

```
#Run CORE3D_Perspective_Imagery.py to save perspective images
#Note: 
#Perspective imagery code does not allow specification of AOI bounding box, however this can be controlled via camera focal length adjustments
#The first directory specified should be the saved location of CORE3D_Perspective_Imagery.py 
#The second directory specified should be the path where the .obj file is saved. 
#If you use -h for help, Blender -help options will be provided

blender --background --python C:\Your_Directory\CORE3D_Perpsective_Imagery.py -- -p "C:/Users/Your_OBJ_Directory/example.obj" -g 1.0 -z False -N 4 -e 60.0 -f 30.0 -r 1800.0
```

```
#General command line formatting
test.py -- -p <filepath> -g <gsd> -x <x1> -y <y1> -X <x2> -Y <y2> -z <+z up?> -N <frame#> -e <elevation angle (deg)> -f <focal length (mm)> -r <range from center of AOI>'
```
