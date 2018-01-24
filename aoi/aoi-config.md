This document accompanies the [AOI Example Configuration](aoi-example.config) to elaborate on proper configuration setup.

The structure of the data follows closely to Vricon satellite imagery packages.

# Reference Inputs
This section is denoted by the \[INPUT.REF\] tag and is used to identify ground truth files to use for metrics analysis.

## DSMFilename
 Relative or absolute path to associated DSM (Digital Surface Model) GeoTIFF file - a DSM represents the first reflective surface
## DTMFilename
 Relative or absolute path to associated DTM (Digital Terrain Model) GeoTIFF file - a DTM represents the bare earth surface
## CLSFilename
 Relative or absolute path to associated CLS (landcover classification) GeoTIFF file - buildings and other man-made structures are labeled
## NDXFilename
 Relative or absolute path to associated NDX (unique index for each man-made structure) GeoTIFF file
## MTLFilename
 Relative or absolute path to associated MTL (material label) GeoTIFF file
## CLSMatchValue
 Classification value for man-made structure type (e.g., building) to evaluate using the metrics

# Test Inputs
 This section is denoted by the \[INPUyou ou T.TEST\] tag and is used to identify the test data set to be compared with the ground truth files.

## DSMFilename
 Relative or absolute path to associated DSM (Digital Surface Model) GeoTIFF file - a DSM represents the first reflective surface
## DTMFilename
 Relative or absolute path to associated DTM (Digital Terrain Model) GeoTIFF file - a DTM represents the bare earth surface
## CLSFilename
 Relative or absolute path to associated CLS (landcover classification) GeoTIFF file - buildings and other man-made structures are labeled
## NDXFilename
 Relative or absolute path to associated NDX (unique index for each man-made structure) GeoTIFF file
## MTLFilename
 Relative or absolute path to associated MTL (material label) GeoTIFF file
## CLSMatchValue
 Classification value for man-made structure type (e.g., building) to evaluate using the metrics

# Options
This section is denoted by the \[OPTIONS\] tag and is used to configure optional parameters for metric analysis.
## QuantizeHeight
 This boolean flag is used to turn height quantization on or off. Suggested default is 'True'

# Plots
This section is denoted by the \[PLOTS\] tag and is used to set options for drawing and saving visualization plots.
## DoPlots
 This boolean is used to enable plots.
## SavePlots
 After this is implemented, this boolean will enable saving plots to file.

# Registration Executable Path
This section is denoted by the \[REGEXEPATH\] tag and is used to locate executable files.
## Align3DPath
 Relative or absolute path to pubgeo's Align3d executable

# Materials Reference
This section is denoted by the \[MATERIALS.REF\] tag and is used to describe material labels
## MaterialIndices
 A comma separated list of integer indices to use for material labeling
## MaterialNames
 A comma separated list of strings labels associated with the material types
## MaterialIndicesToIgnore
 A comma separated list of integer indices to be ignored in metric analysis