This document accompanies the [AOI Example Configuration](aoi-example.config) to elaborate on proper configuration setup.

The structure of the data follows closely to Vricon satellite imagery packages.

# Reference Inputs
This section is denoted by the \[INPUT.REF\] tag and is used to identify ground truth files to use for metrics analysis.

#### DSMFilename
 Relative or absolute path to associated DSM (Digital Surface Model) GeoTIFF file - a DSM represents the first reflective surface
#### DTMFilename
 Relative or absolute path to associated DTM (Digital Terrain Model) GeoTIFF file - a DTM represents the bare earth surface
#### CLSFilename
 Relative or absolute path to associated CLS (landcover classification) GeoTIFF file - buildings and other man-made structures are labeled
#### NDXFilename
 Relative or absolute path to associated NDX (unique index for each man-made structure) GeoTIFF file
#### MTLFilename
 Relative or absolute path to associated MTL (material label) GeoTIFF file.  This field is options. If not specified material metrics are not computed.
#### CLSMatchValue
 Classification value for man-made structure type (e.g., building) to evaluate using the metrics. This is specified as an single value or array of arrays.

# Test Inputs
 This section is denoted by the \[INPUT.TEST\] tag and is used to identify the test data set to be compared with the ground truth files.

#### DSMFilename
 Relative or absolute path to associated DSM (Digital Surface Model) GeoTIFF file - a DSM represents the first reflective surface
#### DTMFilename
 Relative or absolute path to associated DTM (Digital Terrain Model) GeoTIFF file - a DTM represents the bare earth surface. This field is optional.  If not specified, \[INPUT.REF\]\[DTMFilename\] is used in it's place.
#### CLSFilename
 Relative or absolute path to associated CLS (landcover classification) GeoTIFF file - buildings and other man-made structures are labeled
#### MTLFilename
 Relative or absolute path to associated MTL (material label) GeoTIFF file. This field is options. If not specified material metrics are not computed.
#### CLSMatchValue
 Classification value for man-made structure type (e.g., building) to evaluate using the metrics.  This field is optional.  Value defaults to \[INPUT.REF\]\[CLSMatchValue\].

# Options
This section is denoted by the \[OPTIONS\] tag and is used to configure optional parameters for metric analysis.
#### QuantizeHeight
 This boolean flag is used to turn height quantization on or off. Suggested default is 'True'
#### TerrainZErrorThreshold
 Threshold value used to determine height error in terrain accuracy metrics
# Plots
This section is denoted by the \[PLOTS\] tag and is used to set options for drawing and saving visualization plots.
#### ShowPlots
Boolean flag to enable displaying plots.
#### SavePlots
Boolean flag to enable saving plots to file.  ShowPlots does not need to be enabled to save plots.

# Registration Executable Path
This optional section is denoted by the \[REGEXEPATH\] tag and is used to locate executable files. By default, the application will search the $PATH variable for an align3d executable.
#### Align3DPath
 Relative or absolute path to a custom pubgeo Align3d executable

# Materials Reference
This section is denoted by the \[MATERIALS.REF\] tag and is used to describe material labels
#### MaterialIndices
 A comma separated list of integer indices to use for material labeling
#### MaterialNames
 A comma separated list of strings labels associated with the material types
#### MaterialIndicesToIgnore
 A comma separated list of integer indices to be ignored in metric analysis
