% Approved for public release, 21-371

% Copyright (c) 2014 The Johns Hopkins University Applied Physics Laboratory
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
%

%
% Disclaimer: 
% These functions are provided as an archive of engineering code that was developed and used in 
% 2014-2015 with specific point cloud datasets and test targets. Assumptions were made that may 
% not generalize well for other uses. If you plan to use these for your own purposes, please 
% review carefully and revise as necessary. 
%

% 
% Some of the methods implemented here were based on work reported in the following paper:
%
% Jeffrey R. Stevens, Norman A. Lopez, and Robin R. Burton, 
% "Quantitative data quality metrics for 3D laser radar systems", 
% Proc. SPIE 8037, Laser Radar Technology and Applications XVI, 80370J (8 June 2011)
% https://doi.org/10.1117/12.888832
%

%
% Crop a point cloud to points within a specified volume.
%
function [cropped_points] = crop_volume(points, rectangle, tolerance_xy, tolerance_z, apply_rotation)
% Get min and max X, Y, and Z values.
xmin = min(rectangle(:,1)) - tolerance_xy;
xmax = max(rectangle(:,1)) + tolerance_xy;
ymin = min(rectangle(:,2)) - tolerance_xy;
ymax = max(rectangle(:,2)) + tolerance_xy;
zmin = min(rectangle(:,3)) - tolerance_z;
zmax = max(rectangle(:,3)) + tolerance_z;
% Crop in X, Y, and Z.
cropped_points = points((points(:,1) >= xmin) & (points(:,1) <= xmax),:);
cropped_points = cropped_points((cropped_points(:,2) >= ymin) & (cropped_points(:,2) <= ymax),:);
cropped_points = cropped_points((cropped_points(:,3) >= zmin) & (cropped_points(:,3) <= zmax),:);
% If there is no overlap, then stop now.
if (numel(cropped_points) == 0)
    return;
end
% Demean the cropped point cloud.
% This is required to reliably apply rotation matrix.
mx = mean(cropped_points(:,1));
my = mean(cropped_points(:,2));
mz = mean(cropped_points(:,3));
cropped_points(:,1) = cropped_points(:,1) - mx;
cropped_points(:,2) = cropped_points(:,2) - my;
cropped_points(:,3) = cropped_points(:,3) - mz;
% Demean the rectangle coordinates.
% This is necessary for reliable polynomial fit.
demeaned_rectangle(:,1) = rectangle(:,1) - mx;
demeaned_rectangle(:,2) = rectangle(:,2) - my;
demeaned_rectangle(:,3) = rectangle(:,3) - mz;
% Compute rotation matrix to align rectangle with coordinate axes.
xy_angles = [(atan((demeaned_rectangle(1,2)-demeaned_rectangle(2,2))/(demeaned_rectangle(1,1)-demeaned_rectangle(2,1)))+pi/2.), ...
    (atan((demeaned_rectangle(2,2)-demeaned_rectangle(3,2))/(demeaned_rectangle(2,1)-demeaned_rectangle(3,1)))), ...
    (atan((demeaned_rectangle(3,2)-demeaned_rectangle(4,2))/(demeaned_rectangle(3,1)-demeaned_rectangle(4,1)))+pi/2.), ...
    (atan((demeaned_rectangle(4,2)-demeaned_rectangle(1,2))/(demeaned_rectangle(4,1)-demeaned_rectangle(1,1))))];
xy_angle = median(xy_angles)-pi/2.;
xy_matrix = [[cos(xy_angle), sin(xy_angle), 0]; [-sin(xy_angle), cos(xy_angle), 0]; [0, 0, 1]];
xz_poly = polyfit(demeaned_rectangle(:,1), demeaned_rectangle(:,3), 1);
xz_angle = atan(xz_poly(1));
xz_matrix = [[cos(xz_angle),0,sin(xz_angle)]; [0,1,0]; [-sin(xz_angle),0,cos(xz_angle)]];
yz_poly = polyfit(demeaned_rectangle(:,2), demeaned_rectangle(:,3), 1);
yz_angle = atan(yz_poly(1));
yz_matrix = [[1,0,0]; [0,cos(yz_angle),sin(yz_angle)]; [0,-sin(yz_angle),cos(yz_angle)]];
M = xy_matrix * yz_matrix * xz_matrix;
% Rotate both the point cloud and the rectangle.
cropped_points = (M * cropped_points')';
demeaned_rectangle = (M * demeaned_rectangle')';
% Get new rotated min and max X and Y values.
xmin = min(demeaned_rectangle(:,1)) - tolerance_xy;
xmax = max(demeaned_rectangle(:,1)) + tolerance_xy;
ymin = min(demeaned_rectangle(:,2)) - tolerance_z;
ymax = max(demeaned_rectangle(:,2)) + tolerance_z;
% Crop in rotated X, Y, and Z.
cropped_points = cropped_points((cropped_points(:,1) >= xmin) & (cropped_points(:,1) <= xmax),:);
cropped_points = cropped_points((cropped_points(:,2) >= ymin) & (cropped_points(:,2) <= ymax),:);
mean_z = mean(cropped_points(:,3));
cropped_points = cropped_points((cropped_points(:,3) >= (mean_z - tolerance_z)) & (cropped_points(:,3) <= (mean_z + tolerance_z)),:);
% Rotate the points back to the original coordinate system.
if (~apply_rotation)
    cropped_points = (inv(M) * cropped_points')';
    cropped_points(:,1) = cropped_points(:,1) + mx;
    cropped_points(:,2) = cropped_points(:,2) + my;
    cropped_points(:,3) = cropped_points(:,3) + mz;
end


%
% Estimate Ground Spatial Resolution (GSR) using Contrast Transfer Function (CTF) tri-bar target.
% Input:
% - points - point cloud for analysis
% - rectangle - cell array defining CTF bar target corner coordinates
% - gsd - ground sample distance used to lower bound resolution
% - thresh - contrast threshold, typical is 0.2
% - do_align - refine alignment to maximize contrast ratio
% - htol - horizontal tolerance added to bounding box; 50cm works well
% - ztol - vertical tolerance added to bounding box; 50cm works well
% Output:
% - gsr - estimated resolution
% - num_points - number of points cropped near the CTF target
% - best_offset - alignment offset that maximizes contrast ratio
%
function [gsr, num_points, best_offset] = rj_ctf(points, rectangle, gsd, thresh, do_align, htol, ztol)
% Get bounding rectangle for the entire CTF target.
num_rectangles = length(rectangle);
xmin = min(rectangle{1}(:,1));
xmax = max(rectangle{1}(:,1));
ymin = min(rectangle{1}(:,2));
ymax = max(rectangle{1}(:,2));   
zmin = min(rectangle{1}(:,3));
zmax = max(rectangle{1}(:,3));  
for ndx=2:num_rectangles
    xmin = min(xmin, min(rectangle{ndx}(:,1)));
    xmax = max(xmax, max(rectangle{ndx}(:,1)));
    ymin = min(ymin, min(rectangle{ndx}(:,2)));
    ymax = max(ymax, max(rectangle{ndx}(:,2)));   
    zmin = min(zmin, min(rectangle{ndx}(:,3)));
    zmax = max(zmax, max(rectangle{ndx}(:,3))); 
end
zval = (zmin + zmax)/2.0;
bounding_box = [[xmin, ymin, zval]; ...
    [xmax, ymin, zval]; ...
    [xmax, ymax, zval]; ...
    [xmin, ymax, zval]];
% Crop to bounding box.
[cropped_points] = crop_volume(points, bounding_box, htol, ztol, 0); 
if (isempty(cropped_points))
    gsr = 0.0;
    num_points = 0;
    best_offset = 0;
    return;
end
% Rotate point cloud and all polygons to align with XY axis.
mx = mean(cropped_points(:,1));
my = mean(cropped_points(:,2));
mz = mean(cropped_points(:,3));
cropped_points(:,1) = cropped_points(:,1) - mx;
cropped_points(:,2) = cropped_points(:,2) - my;
cropped_points(:,3) = cropped_points(:,3) - mz;
% Pick biggest bar and estimate rotation angle.
rect = rectangle{num_rectangles};
xy_angles = [(atan((rect(1,2)-rect(2,2))/(rect(1,1)-rect(2,1)))+pi/2.), ...
    (atan((rect(2,2)-rect(3,2))/(rect(2,1)-rect(3,1)))), ...
    (atan((rect(3,2)-rect(4,2))/(rect(3,1)-rect(4,1)))+pi/2.), ...
    (atan((rect(4,2)-rect(1,2))/(rect(4,1)-rect(1,1))))];
xy_angle = median(xy_angles)-pi/2.;
xy_matrix = [[cos(xy_angle), sin(xy_angle), 0]; [-sin(xy_angle), cos(xy_angle), 0]; [0, 0, 1]];
% Update angle to make sure long end is mapped to X.
xmin = inf;
xmax = -inf;
ymin = inf;
ymax = -inf;
for ndx=1:num_rectangles
    r =rectangle{ndx};
    r(:,1) = r(:,1) - mx;
    r(:,2) = r(:,2) - my;
    r(:,3) = r(:,3) - mz;
    r = (xy_matrix * r')';    
    xmin = min(xmin,min(r(:,1)));
    xmax = max(xmax,max(r(:,1)));
    ymin = min(ymin,min(r(:,2)));
    ymax = max(ymax,max(r(:,2)));
end
xdim = xmax-xmin;
ydim = ymax-ymin;
if (xdim < ydim) 
    xy_angle = xy_angle + pi/2.0;
    xy_matrix = [[cos(xy_angle), sin(xy_angle), 0]; [-sin(xy_angle), cos(xy_angle), 0]; [0, 0, 1]];
end
% Update angle to make sure bars increase in size along X.
x1 = rectangle{1}(1,1);
x2 = rectangle{num_rectangles}(1,1);
if (x1 > x2)
    xy_angle = xy_angle + pi;
    xy_matrix = [[cos(xy_angle), sin(xy_angle), 0]; [-sin(xy_angle), cos(xy_angle), 0]; [0, 0, 1]];
end
% Now rotate everything.
cropped_points = (xy_matrix * cropped_points')';
ymin = inf;
ymax = -inf;
for ndx=1:num_rectangles
    r =rectangle{ndx};
    r(:,1) = r(:,1) - mx;
    r(:,2) = r(:,2) - my;
    r(:,3) = r(:,3) - mz;
    r = (xy_matrix * r')';
    rectangle{ndx} = r;
    ymin = min(ymin,min(r(:,2)));
    ymax = max(ymax,max(r(:,2)));
end   
% Crop out points near the bar corners to avoid including the supports.
eps = 0.5;
cropped_points = cropped_points((cropped_points(:,2) >= (ymin+eps)) & (cropped_points(:,2) <= (ymax-eps)),:);
x = cropped_points(:,1);
ydim = (ymax-ymin-2*eps);
% Confirm the series of distances between adjacent bounding boxes.
x1 = zeros(num_rectangles,1);
x2 = zeros(num_rectangles,1);
on_width = zeros(num_rectangles,1);
for ndx=1:num_rectangles
    x1(ndx) = min(rectangle{ndx}(:,1));
    x2(ndx) = max(rectangle{ndx}(:,1));
    on_width(ndx) = x2(ndx) - x1(ndx);
end
off_width = zeros(num_rectangles-1,1);
for ndx=1:num_rectangles-1
    off_width(ndx) = x1(ndx+1) - x2(ndx);
end
% Align points with largest rectangles.
best_ratio = 0.0;
best_offset = 0.0;
if (do_align)
    for xoff=-1.0:0.1:1
        on_count1 = sum((x+xoff > x1(num_rectangles-1)) & (x+xoff < x2(num_rectangles-1)));
        off_count1 = sum((x+xoff > x2(num_rectangles-1)) & (x+xoff < x1(num_rectangles)));
        on_count2 = sum((x+xoff > x1(num_rectangles-2)) & (x+xoff < x2(num_rectangles-2)));
        off_count2 = sum((x+xoff > x2(num_rectangles-2)) & (x+xoff < x1(num_rectangles-1)));
        on_count = (on_count1 + on_count2)/2.0;
        off_count = (off_count1 + off_count2)/2.0;
        ratio = (on_count - off_count)/(on_count + off_count);
        if (ratio > best_ratio)
            best_ratio = ratio;
            best_offset = xoff;
        end    
    end
    disp(['Tri-bar X offset = ' num2str(best_offset)]);
    x = x + best_offset;
    cropped_points(:,1) = cropped_points(:,1) + best_offset;
end
% Count points in and between bounding boxes and compute GSR.
num_tribars = num_rectangles / 3;
ratios = zeros(num_tribars,1);
freqs = zeros(num_tribars,1);
widths = zeros(num_tribars,1);
for num=1:num_tribars
    ndx = (num-1)*3 + 1;
    avg_width = (on_width(ndx) + on_width(ndx+1) + on_width(ndx+2))/3.0;
    freqs(num) = 1.0 / (2.0 * avg_width);
    widths(num) = avg_width;
    on_count1 = sum((x > x1(ndx)) & (x < x2(ndx)));
    off_count1 = sum((x > x2(ndx)) & (x < x1(ndx+1)));
    on_count2 = sum((x > x1(ndx+1)) & (x < x2(ndx+1)));
    off_count2 = sum((x > x2(ndx+1)) & (x < x1(ndx+2)));
    on_count3 = sum((x > x1(ndx+2)) & (x < x2(ndx+2)));
    on_count = (on_count1 + on_count2 + on_count3) / 3.0;
    off_count = (off_count1 + off_count2)/2.0;
    % If no points or bar width too narrow, then set contrast to zero.
    % This results in a conservative estimate but avoids outliers.
    if ((on_count > 0) && (avg_width >= gsd))
        ratios(num) = (on_count - off_count)/(on_count + off_count);        
    end
 end
% Get Ground Spatial Resolution (GSR).
widths10 = interp(widths,10,3);
ratios10 = interp(ratios,10,3);
maxval = length(widths10);
gsr = inf;
contrast_value = 0.0;
for num=maxval:-1:1
    if (ratios10(num) > thresh)
        gsr = widths10(num);
        contrast_value = ratios10(num);
    else
        break;
    end
end
if (contrast_value == 1.0)
    gsr = inf;
end
disp(['Tri-bar GSR = ' num2str(gsr)]);
% Keep number of points on CTF target.
num_points = length(cropped_points);
% Draw plot.
figure('Position',[50 50 800 800]);
subplot(2,1,1);
hold on;
for i=1:length(cropped_points)
    plot(cropped_points(i,1),cropped_points(i,2),'.');
end
for ndx=1:num_rectangles   
    mypoly = [rectangle{ndx};rectangle{ndx}(1,:)];
    plot(mypoly(:,1), mypoly(:,2),'Color',[1.0,0.0,0.0]);
end
hold off;
title('Points Cropped at Center of Tri-bar CTF Target Array', 'FontWeight' , 'bold');
xlabel('Long Axis (meters)');
ylabel('Short Axis (meters)');
xrange = [min(rectangle{1}(:,1))-1 max(rectangle{num_rectangles}(:,1))+1];
yrange = [min(rectangle{1}(:,2))-1 max(rectangle{num_rectangles}(:,2))+1];
xlim(xrange);
ylim(yrange);
subplot(2,1,2);
plot(widths, ratios);
hold on;
plot(xrange,[0.2,0.2],'Color',[1.0,0.0,0.0]);
plot([gsd,gsd],yrange,'Color',[0.0,1.0,0.0]);
ylim([0,1]);
xlim([0,max(widths)+0.5]);
title('Contrast Transfer Function (CTF)', 'FontWeight' , 'bold');
xlabel('Bar Size (meters)');
ylabel('Ratio');


%
% Estimate point cloud density, ground sample distance, and void fraction
%
function [density, gsd, void_fraction] = rj_density_rectangle(points, rectangle, tolerance_xy, tolerance_z, bin_size)
% Crop the point cloud and rotate to horizontal.
% This should also work for vertical selection rectangles (e.g., for walls on a building).
apply_rotation = 1;
[cropped_points] = crop_volume(points, rectangle, tolerance_xy, tolerance_z, apply_rotation);
num_points = numel(cropped_points);
% Define a grid size and compute density and GSD on the grid.
gsd = rj_gsd(cropped_points, bin_size);
void_fraction = rj_void_fraction(cropped_points, gsd);
density = 1/(gsd*gsd);
disp(['GSD = ' num2str(gsd)]);
disp(['Density = ' num2str(density)]);
disp(['Void Fraction = ' num2str(void_fraction)]);


%
% Estimate false detection rate using point cloud ground truth
%
function [fraction] = rj_false_detection(points, truth, horizontal_distance, vertical_distance)
% Crop points to truth XY bounds.
xmin = min(truth(:,1));
xmax = max(truth(:,1));
ymin = min(truth(:,2));
ymax = max(truth(:,2));
cropped_points = points((points(:,1) >= xmin) & (points(:,1) <= xmax),:);
cropped_points = cropped_points((cropped_points(:,2) >= ymin) & (cropped_points(:,2) <= ymax),:);
if (numel(cropped_points) == 0)
    fraction = 0.0;
    return;
end
% For every target point, look for a matching truth point.
count = 0;
[num_truth, dims_truth] = size(truth);
[num_target, dims_target] = size(cropped_points);
num_target_points = 0;
for i=1:num_target
    found_overlap = 0;
    for j=1:num_truth
        overlap = (abs(cropped_points(i,1) - truth(j,1)) < horizontal_distance) ...
                && (abs(cropped_points(i,2) - truth(j,2)) < horizontal_distance);
        if (overlap)
            found_overlap = 1;
            if (abs(cropped_points(i,3) - truth(j,3)) < vertical_distance)
                count = count + 1;
                break;
            end
        end
    end
    if (found_overlap)
        num_target_points = num_target_points + 1;
    end
end
% Return fraction of truth points found.
fraction = (num_target_points - count)/num_target_points;


%
% Estimate ground sample distance (GSD) for point cloud
% by counting points in a grid of bins of size 'bin_size_meters' which
% is much larger than the GSD but smaller than the point cloud extents.
% For example, for RJ metric analysis with GSD ranging from centimeters 
% to a couple meters, a 5m grid provides very consistent results. For a 
% heavily cropped data set that may not have 5m extent, a smaller bin size
% should be used to ensure a good distribution of bins. The median value 
% is used to provide some robustness to nonuniform sampling, voids, etc.
%
function [gsd] = rj_gsd(points, bin_size_meters)
% Place points in a quantized grid.
xmin = min(points(:,1));
xmax = max(points(:,1));
ymin = min(points(:,2));
ymax = max(points(:,2));
numxbins = ceil((xmax-xmin)/bin_size_meters) + 1;
numybins = ceil((ymax-ymin)/bin_size_meters) + 1;
x = round((points(:,1) - xmin)/bin_size_meters + 0.5);
y = round((points(:,2) - ymin)/bin_size_meters + 0.5);
% Get average point density from grid bins and invert to get GSD.
D = zeros(numxbins, numybins);
for i=1:length(x)
    D(x(i),y(i)) = D(x(i),y(i)) + 1;
end
density = median(median(D(D > 0)));
gsd = bin_size_meters/sqrt(density);


%
% Estimate horizontal mensuration error for a rectangular structure compared to a 
% known rectangle with accurate edge lengths.
% Inputs:
% - points - point cloud for analysis
% - rectangle - array defining rectangle corner coordinates
% - gsd - ground sample distance used to define point sampling rate
% - htol - horizontal tolerance added to bounding box; 1m works well
% - ztol - vertical tolerance added to bounding box; 50cm works well
% Ouputs:
% - hrms - Root mean square horizontal mensuration error (meters)
%
function [hrms] = rj_mensuration_error(points, rectangle, gsd, htol, ztol)
% Crop points to rectangle.
apply_rotation = 0;
[cropped_points] = crop_volume(points, rectangle, htol, ztol, apply_rotation);
% If there are no points, return zero.
num_points = length(cropped_points);
if (num_points == 0)
    hrms = 0.0;
    return;
end
% Demean the cropped point cloud.
mxp = mean(cropped_points(:,1));
myp = mean(cropped_points(:,2));
mzp = mean(cropped_points(:,3));
cropped_points(:,1) = cropped_points(:,1) - mxp;
cropped_points(:,2) = cropped_points(:,2) - myp;
cropped_points(:,3) = cropped_points(:,3) - mzp;
% Demean the rectangle coordinates.
mxr = mean(rectangle(:,1));
myr = mean(rectangle(:,2));
mzr = mean(rectangle(:,3));
rectangle(:,1) = rectangle(:,1) - mxr;
rectangle(:,2) = rectangle(:,2) - myr;
rectangle(:,3) = rectangle(:,3) - mzr;
% Compute the XY rotation and apply.
xy_angles = [mod(atan((rectangle(1,2)-rectangle(2,2))/(rectangle(1,1)-rectangle(2,1))),pi), ...
    mod(atan((rectangle(2,2)-rectangle(3,2))/(rectangle(2,1)-rectangle(3,1)))-pi/2.,pi), ...
    mod(atan((rectangle(3,2)-rectangle(4,2))/(rectangle(3,1)-rectangle(4,1))),pi), ...
    mod(atan((rectangle(4,2)-rectangle(1,2))/(rectangle(4,1)-rectangle(1,1)))-pi/2.,pi)];
xy_angle = median(xy_angles);
xy_matrix = [[cos(xy_angle), sin(xy_angle), 0]; [-sin(xy_angle), cos(xy_angle), 0]; [0, 0, 1]];
pts = (xy_matrix * cropped_points')';
rect = (xy_matrix * rectangle')';
rect = [rect; rect(1,:)];
% Compute XY bounds of rotated ground truth rectangle.
rxmin = min(rect(:,1));
rxmax = max(rect(:,1));
rymin = min(rect(:,2));
rymax = max(rect(:,2));
% Report the XY offset.
off_x = mxr - mxp;
off_y = myr - myp;
off_z = mzr - mzp;
disp(['Rectangle X Offset = ' num2str(off_x)]);
disp(['Rectangle Y Offset = ' num2str(off_y)]);
disp(['Rectangle Z Offset = ' num2str(off_z)]);
% Compute edge lengths of ground truth rectangle.
x_edge_length = rxmax - rxmin;
y_edge_length = rymax - rymin;
disp(['Rectangle X Length = ' num2str(x_edge_length)]);
disp(['Rectangle Y Length = ' num2str(y_edge_length)]);
% Initialize the search for point distances from edges.
spacing = gsd/4;
num_vertical = (rymax-rymin)/spacing + 1;
num_horizontal = (rxmax-rxmin)/spacing + 1;
indices = [];
left_diffs = [];
right_diffs = [];
top_diffs = [];
bottom_diffs = [];
% Sample points along the vertical edges.
for i=1:num_vertical
    y = rymin + spacing*(i-1);
    % Left edge.
    minx = rxmax;
    best_index = -1;
    for ndx=1:num_points
        if (pts(ndx,1) < minx) && (abs(pts(ndx,2) - y) < gsd)
            minx = pts(ndx,1);
            best_index = ndx;
        end
    end
    if (best_index > 0)
        indices = [indices best_index];
        left_diffs = [left_diffs (rxmin-pts(best_index,1))];
    end
    % Right edge.
    maxx = rxmin;
    best_index = -1;
    for ndx=1:num_points
        if (pts(ndx,1) > maxx) && (abs(pts(ndx,2) - y) < gsd)
            maxx = pts(ndx,1);
            best_index = ndx;
        end
    end
    if (best_index > 0)
        indices = [indices best_index];
        right_diffs = [right_diffs (rxmax-pts(best_index,1))];
    end    
end
% Sample points along the horizontal edges.
for i=1:num_horizontal
    x = rxmin + spacing*(i-1);
    % Bottom edge.
    miny = rymax;
    best_index = -1;
    for ndx=1:num_points
        if (pts(ndx,2) < miny) && (abs(pts(ndx,1) - x) < gsd)
            miny = pts(ndx,2);
            best_index = ndx;
        end
    end
    if (best_index > 0)
        indices = [indices best_index];
        bottom_diffs = [bottom_diffs (rymin-pts(best_index,2))];
    end
    % Top edge.
    maxy = rymin;
    best_index = -1;
    for ndx=1:num_points
        if (pts(ndx,2) > maxy) && (abs(pts(ndx,1) - x) < gsd)
            maxy = pts(ndx,2);
            best_index = ndx;
        end
    end
    if (best_index > 0)
        indices = [indices best_index];
        top_diffs = [top_diffs (rymax-pts(best_index,2))];
    end    
end
% Review the statistics.
selected_pts = pts(indices,:);
left_rms = sqrt(mean(left_diffs.*left_diffs));
right_rms = sqrt(mean(right_diffs.*right_diffs));
top_rms = sqrt(mean(top_diffs.*top_diffs));
bottom_rms = sqrt(mean(bottom_diffs.*bottom_diffs));
disp(['Left RMS (m) = ' num2str(left_rms)]);
disp(['Right RMS (m) = ' num2str(right_rms)]);
disp(['Top RMS (m) = ' num2str(top_rms)]);
disp(['Bottom RMS (m) = ' num2str(bottom_rms)]);
all_diffs = [left_diffs right_diffs top_diffs bottom_diffs];
hrms = sqrt(mean(all_diffs.*all_diffs));
disp(['H-RMS (m) = ' num2str(hrms)]);
% Draw points and rectangle.
figure;
hold on;
for i=1:length(pts)
    plot(pts(i,1),pts(i,2),'.');
end
for i=1:length(selected_pts)
    plot(selected_pts(i,1),selected_pts(i,2),'*','Color',[0.0,1.0,0.0]);
end
plot(rect(:,1), rect(:,2),'Color',[1.0,0.0,0.0]);
title(['H-RMS (m) = ' num2str(hrms)], 'FontWeight' , 'bold');
xlabel('X (meters)');
ylabel('Y (meters)');
xrange = [rxmin-2 rxmax+2];
yrange = [rymin-2 rymax+2];
xlim(xrange);
ylim(yrange);

%
% Estimate slope error on a flat plane
%
function[ave_surface_slope_rms,ave_surface_slope_std,ave_surface_slope_mean,fig_handle] = rj_slope_error(target,gt,scales)
% crop target to truth bounds.
xmin = min(gt.x);
xmax = max(gt.x);
ymin = min(gt.y);
ymax = max(gt.y);
zmin = min(gt.z);
zmax = max(gt.z);
ind = (target.x >= xmin) & (target.x <= xmax) & (target.y >= ymin) & (target.y <= ymax);
if numel(find(ind==1))==0
    fprintf('no points found');
    ave_surface_slope_rms=NaN;
    ave_surface_slope_std=NaN;
    ave_surface_slope_mean=NaN;
    fig_handle=[];
    return
end
data.x = target.x(ind);
data.y = target.y(ind);
data.z = target.z(ind);
% center data at zero for better vis
cx = (max(gt.x)-min(gt.x))/2;
cy = (max(gt.y)-min(gt.y))/2;
cz = (max(gt.z)-min(gt.z))/2;
x_gt = gt.x - cx;
y_gt = gt.y - cy;
z_gt = gt.z - cz;
x_test = data.x - cx;
y_test = data.y - cy;
z_test = data.z - cz;
%get x,y extents
xmin=min(min(x_test),min(x_gt));
xmax=max(max(x_test),max(x_gt));
ymin=min(min(y_test),min(y_gt));
ymax=max(max(y_test),max(y_gt));
% grid data at each scale
scale_idx=0;
ave_surface_slope_rms=cell(1,length(scales));
ave_surface_slope_std=cell(1,length(scales));
ave_surface_slope_mean=cell(1,length(scales));
slope_errs_means=cell(1,length(scales));
for scale = scales
    scale_idx=scale_idx+1;
    iis = floor((x_test-xmin)/scale+0.5);
    jjs = floor((y_test-ymin)/scale+0.5);   
    width = max(iis);
    height = max(jjs);
    iisgt = floor((x_gt-xmin)/scale+0.5);
    jjsgt = floor((y_gt-ymin)/scale+0.5); 
    ncells = width*height;
    cell_x=cell([width,height]);
    cell_y=cell([width,height]);
    cell_z=cell([width,height]);
    cell_x_gt=cell([width,height]);
    cell_y_gt=cell([width,height]);
    cell_z_gt=cell([width,height]);
    for c=1:ncells
        [si, sj] = ind2sub([width,height],c);
        touse = (si == iis) & (sj == jjs);
        cell_x{ind2sub([width,height],c)}=x_test(touse);
        cell_y{ind2sub([width,height],c)}=y_test(touse);
        cell_z{ind2sub([width,height],c)}=max(z_test(touse));        
        clear si sj touse
        [si, sj] = ind2sub([width,height],c);
        touse = (si == iisgt) & (sj == jjsgt);
        cell_x_gt{ind2sub([width,height],c)}=x_gt(touse);
        cell_y_gt{ind2sub([width,height],c)}=y_gt(touse);
        cell_z_gt{ind2sub([width,height],c)}=max(z_gt(touse));        
        hold all
    end
    slope_errs = cell([width,height]);
    slope_rmss = cell([width,height]);
    slope_errs_std=cell([width,height]);
    for w=2:width-1
        for h=2:height-1
            clear wm hm inds c_ind center zs dzs slopes_data slopes_gt
            [wm,hm]=meshgrid(w-1:w+1,h-1:h+1);
            hm(1:2:end)=0;
            wm(1:2:end)=0;
            inds=sub2ind([width,height],wm(wm>0),hm(hm>0));            
            inds=inds(inds<=numel(cell_z)&inds<=numel(cell_z_gt));
            c_ind=sub2ind([width,height],w,h);
            %to use cells
            zs=cell_z(inds);
            center=cell(size(cell_z(inds)));
            center(:)=cell_z(c_ind);
            if isempty(cell2mat(zs)) || isempty(cell2mat(center))
                continue; 
            end;
            %compute delta zs, distance = scale
            dzs=cellfun(@abs,cellfun(@minus,center,zs,'un',0),'un',0);
            scale_cell=cell(size(dzs));
            scale_cell(:)={scale};
            slopes_data=cellfun(@atan,cellfun(@rdivide,dzs,scale_cell,'un',0),'un',0);
            slopes_data=cellfun(@rad2deg,slopes_data,'un',0);
            clear center zs dzs slopes_gt
            zs=cell_z_gt(inds);
            center=cell(size(cell_z_gt(inds)));
            center(:)=cell_z_gt(c_ind);
            if isempty(cell2mat(zs)) || isempty(cell2mat(center))
                continue; 
            end;
            %compute delta zs, distance = scale
            dzs=cellfun(@abs,cellfun(@minus,center,zs,'un',0),'un',0);
            scale_cell=cell(size(dzs));
            scale_cell(:)={scale};
            slopes_gt=cellfun(@atan,cellfun(@rdivide,dzs,scale_cell,'un',0),'un',0);
            slopes_gt=cellfun(@rad2deg,slopes_gt,'un',0);
            if isempty(slopes_gt) || isempty(slopes_data)
                continue
            end
            slope_errs{w,h} = cellfun(@abs,cellfun(@minus,slopes_gt,slopes_data,'un',0),'un',0);
            %compute rms
            tmp_minus_sq=cellfun(@(x) x.^2, cellfun(@minus,slopes_gt,slopes_data,'un',0),'un',0);
            tmp_sum=sum(cell2mat(tmp_minus_sq))/length(tmp_minus_sq(~cellfun(@isempty,tmp_minus_sq)));
            slope_rmss{w,h}=sqrt(tmp_sum);
            slope_errs_std{w,h}=std(cell2mat(slope_errs{w,h}));
            slope_errs_means{w,h} = mean(cell2mat(slope_errs{w,h}));
        end
    end
    ave_surface_slope_rms{scale_idx}=mean(cell2mat(slope_rmss(:)));
    ave_surface_slope_std{scale_idx}=mean(cell2mat(slope_errs_std(:)));
    ave_surface_slope_mean{scale_idx}=mean(cell2mat(slope_errs_means(:)));
end
figure
fig_handle=gcf;
plot(scales,cell2mat(ave_surface_slope_rms),'.b-');
title('Average slope error (rms) vs. scale');
xlabel('scale (m)');
ylabel('average surface slope error (degrees)');




%
% Estimate surface completeness fraction using aligned point cloud ground truth
%
function [fraction] = rj_surface_completeness(points, truth, horizontal_distance, vertical_distance)
% Crop points to truth XYZ bounds.
xmin = min(truth(:,1)) - horizontal_distance;
xmax = max(truth(:,1)) + horizontal_distance;
ymin = min(truth(:,2)) - horizontal_distance;
ymax = max(truth(:,2)) + horizontal_distance;
zmin = min(truth(:,3)) - vertical_distance;
zmax = max(truth(:,3)) + vertical_distance;
cropped_points = points((points(:,1) >= xmin) & (points(:,1) <= xmax),:);
cropped_points = cropped_points((cropped_points(:,2) >= ymin) & (cropped_points(:,2) <= ymax),:);
cropped_points = cropped_points((cropped_points(:,3) >= zmin) & (cropped_points(:,3) <= zmax),:);
if (numel(cropped_points) == 0)
    fraction = 0.0;
    return;
end
% For every truth point, look for a matching target point.
count = 0;
[num_truth, dims_truth] = size(truth);
[num_target, dims_target] = size(cropped_points);
for i=1:num_truth
    for j=1:num_target
        overlap = ((abs(cropped_points(j,1) - truth(i,1)) < horizontal_distance) ...
                && (abs(cropped_points(j,2) - truth(i,2)) < horizontal_distance) ...
                && (abs(cropped_points(j,3) - truth(i,3)) < vertical_distance));            
        if (overlap)
            count = count + 1;
            break;
        end
    end
end
% Return fraction of truth points found.
fraction = count/num_truth;


%
% Estimate surface error w.r.t. ground truth point cloud.
%
function [z_rms_vert,mean_vert_error,std_vert_error,mean_gt_roughness,fig_handle] = rj_surface_error(target,gt,gsd_gt,debug)
%{
inputs:
    1. target - 1x1 lasdata object from lasdata.m containing the test data
    points
    2. truth points - 1x1 lastdata object from lasdata.m containing ground
    truth points
    3. gsd_gt - ground sample distance at which you want to grid the ground truth 
    4. debug - flag for turning on verbose outputs and plotting.
    
        
%outputs (all values in meters):
   1. z_rms_vert - average vertical rms over point cloud (scalar)
    mean_vert_error
   2. mean_vert_err - unsigned mean of error over point cloud (scalar)
   3. std_vert_error - standard deviation of error over point cloud
   (scalar)
   4. mean_gt_roughness - average error over ground truth when fit to a
    plane i.e. ground truth roughness
%}
tic;  
%% crop target to truth bounds.
xmin = min(gt.x);
xmax = max(gt.x);
ymin = min(gt.y);
ymax = max(gt.y);
zmin = min(gt.z);
zmax = max(gt.z);
ind = (target.x >= xmin) & (target.x <= xmax) & (target.y >= ymin) & (target.y <= ymax);
if numel(find(ind==1))==0
    fprintf('no points found');
    z_rms_vert=NaN;
    mean_vert_error=NaN;
    std_vert_error=NaN;
    mean_gt_roughness=NaN;
    fig_handle=[];
    return
end
data.x = target.x(ind);
data.y = target.y(ind);
data.z = target.z(ind);
%% pre-process data
%center at zero for easier visualization
cx = (max(gt.x)-min(gt.x))/2;
cy = (max(gt.y)-min(gt.y))/2;
cz = (max(gt.z)-min(gt.z))/2;
x_gt = gt.x - cx;
y_gt = gt.y - cy;
z_gt = gt.z - cz;
x_d = data.x - cx;
y_d = data.y - cy;
z_d = data.z - cz;
%get extents
xmin_gt=min(x_gt);
xmax_gt=max(x_gt);
ymin_gt=min(y_gt);
ymax_gt=max(y_gt);
%% grid ground truth
%gsd_gt=.15;
id = 'MATLAB:scatteredInterpolant:DupPtsAvValuesWarnId';
warning('off',id)
[X_gt,Y_gt]=meshgrid(xmin_gt:gsd_gt:xmax_gt,ymin_gt:gsd_gt:ymax_gt);
Z_gt=griddata(x_gt,y_gt,z_gt,X_gt,Y_gt);
%% Set up options
%set surf color for plotting in debug
C=zeros([size(Z_gt),3]);
C(:,:,1) = 0.5.*ones(size(Z_gt));
C(:,:,2) = 0.5.*ones(size(Z_gt));
C(:,:,3) = 0.5.*ones(size(Z_gt));
[n,m]=size(Z_gt);
Z_interp=zeros(length(z_d),1);
Z_interp_p=zeros(length(z_d),1);
%whether you want the results plotted
plot_me=0;
plot_orth=0;
if debug
   plot_me=1;
   %randomly pick points to plot so easier to visualize
   dontPlot=randi([0,10],[1,length(z_d)]); %PLOTS AT PLOTS=0
   fprintf('\nFor visibility, %d out of %d points were plotted\n',length(dontPlot(dontPlot==0)),length(z_d))
end
clear Z_interp Z_interp_p diffs surface_rms orth_dists
diffs=zeros(length(z_d),1);
orth_dists=zeros(length(z_d),1);
Z_interp=zeros(length(z_d),1);
Z_interp_p=zeros(length(z_d),1);
surface_rms=zeros(length(z_d),1);
%% Main loop
for i=1:length(z_d)
   %get grid coordinates 
   ii=(x_d(i)-xmin_gt)/gsd_gt;
   jj=(y_d(i)-ymin_gt)/gsd_gt;  
   %check grid coordinates
   if ii==0 || jj ==0 
       Z_interp(i)=NaN;
       diffs(i)=NaN;
       continue
   end
   %note: may need to fix this if a whole number comes up
   if mod(ii,1) == 0 % then its a whole number
       disp('ii is a whole number');
       [iis,jjs]=meshgrid(ii-1:ii,floor(jj):ceil(jj));
   elseif mod(jj,1) == 0 % then its a whole number
       disp('jj is a whole number');
       [iis,jjs]=meshgrid(floor(ii):ceil(ii),jj-1:jj);
   else
       [iis,jjs]=meshgrid(floor(ii):ceil(ii),floor(jj):ceil(jj)); 
   end
   %make sure all indices are valid
    if ~isempty(iis(iis<=0)) || ~isempty(jjs(jjs<=0)) || ~isempty(iis(iis>=m)) || ~isempty(jjs(jjs>=n))
        Z_interp(i)=NaN;
        diffs(i)=NaN;
        continue
    end
   %get values of ground truth at x,y of data
   xxs=X_gt(sub2ind(size(X_gt),jjs+1,iis+1));
   yys=Y_gt(sub2ind(size(Y_gt),jjs+1,iis+1));
   zzs=Z_gt(sub2ind(size(Z_gt),jjs+1,iis+1)); %or iis,jjs, not sure why this is backwards   
   %interpolate z value of ground truth at x,y of data
   Z_interp(i) = interp2(xxs,yys,zzs,x_d(i),y_d(i)); 
   %compute difference between interpolated and actual
   diffs(i)=Z_interp(i)-z_d(i);   
   %compute orthogonal distance between data point and projection of it on
   %fit plane
   [p_orth,d_orth,plane_rms] = getOrthDist([x_d(i),y_d(i),z_d(i)],jj,ii,X_gt,Y_gt,Z_gt);
   orth_dists(i)=d_orth;
   %compute surface rms
   surface_rms(i)=plane_rms;
end
%% Compute metrics
%compute rms with diffs
numNans=numel(diffs(isnan(diffs)));%num nans to subtract from num points
diffs=diffs(~isnan(diffs));%get rid of nans in z differences
n=length(diffs)-numNans;
z_rms_vert=sqrt(sum((diffs).^2)/n);
mean_vert_error=mean(abs(diffs));
std_vert_error=std(abs(diffs));
if debug
    fprintf('----vertical----\n');
    fprintf('z_rms of data = %.2f cm\n',z_rms_vert*100)
    fprintf('mean unsigned distance = %.2f cm\n',mean_vert_error*100)
    fprintf('std unsigned distance = %.2f cm\n',std_vert_error*100)
    fprintf('----------------\n\n');
end
%compute rms with orthogonal
numNans=numel(orth_dists(isnan(orth_dists)));%num nans to subtract from num points
orth_dists=orth_dists(~isnan(orth_dists));%get rid of nans 
surface_rms=surface_rms(~isnan(surface_rms));
n=length(orth_dists)-numNans;
z_rms_orth=sqrt(sum((orth_dists).^2)/n);
mean_gt_roughness=mean(surface_rms);
if debug
    fprintf('----orthogonal----\n');
    fprintf('z_rms of data = %.2f cm\n',z_rms_orth*100)
    fprintf('mean distance = %.2f cm\n',mean(orth_dists)*100)
    fprintf('std distance = %.2f cm\n',std(orth_dists)*100)
    fprintf('----------------\n\n');
    fprintf('average rms of ground truth (roughness) = %f cm\n',mean_gt_roughness*100);
    fprintf('----------------\n\n');
end
if debug
   fprintf('Time elapsed = %.2f seconds \n',toc) 
end
%% plot error distribution
figure;
fig_handle=gcf;
%signed distance
[counts,vals]=hist(diffs*100,20);
num_counts = sum(counts);
if (num_counts ~= 0)
    subplot(1,2,1)
    normalizedCounts = 100 * counts / num_counts;
    b1=bar(vals, normalizedCounts, 'barwidth', 1);
    ylabel('% of data points');
    xlabel('signed distance of data point to surface (cm)')
    title('Distribution of vertical surface errors (signed)'); 
    hold on
    top=ceil(max(normalizedCounts));
    m1=plot(mean(diffs)*100*ones(1,top+1),0:top,'r-.','linewidth',3);
    legend(m1,{['mean = ',num2str(mean(diffs)*100),' cm']}); 
    axis tight
end
%unsigned distance
[counts,vals]=hist(abs(diffs)*100,20);
num_counts = sum(counts);
if (num_counts ~= 0)
    subplot(1,2,2)
    normalizedCounts = 100 * counts / num_counts;
    b2=bar(vals, normalizedCounts, 'barwidth', 1);
    ylabel('% of data points');
    xlabel('unsigned distance of data point to surface (cm)')
    title('Distribution of vertical surface errors (unsigned)'); 
    hold on
    top=ceil(max(normalizedCounts));
    m2=plot(mean(abs(diffs))*100*ones(1,top+1),0:top,'r-.','linewidth',3);
    legend(m2,{['mean = ',num2str(mean(abs(diffs))*100),' cm']});
    axis tight
end
end


%
% Estimate orthogonal distance between data point and projection of it on fit plane.
%
function [p_orth,d,rms] = getOrthDist(p_d,jj,ii,X_gt,Y_gt,Z_gt)
if mod(ii,1) == 0 % then its a whole number
    disp('ii is a whole number');
    [iis,jjs]=meshgrid(ii-1:ii,floor(jj):ceil(jj)); 
elseif mod(jj,1) == 0 % then its a whole number
    disp('jj is a whole number');
    [iis,jjs]=meshgrid(floor(ii):ceil(ii),jj-1:jj);
else
    [iis,jjs]=meshgrid(floor(ii-1):ceil(ii+1),floor(jj-1):ceil(jj+1));
end
[n,m]=size(Z_gt);
if ~isempty(iis(iis<=0)) || ~isempty(jjs(jjs<=0)) || ~isempty(iis(iis>=m)) || ~isempty(jjs(jjs>=n))
    p_orth=NaN;
    d=NaN;
    rms=NaN;
    return
end
[mm,nn]=size(iis);
%get values of ground truth at x,y of data
xxs=X_gt(sub2ind(size(X_gt),jjs+1,iis+1));
yys=Y_gt(sub2ind(size(Y_gt),jjs+1,iis+1));
zzs=Z_gt(sub2ind(size(Z_gt),jjs+1,iis+1));
if ~isempty(zzs(isnan(zzs)))
    p_orth=NaN;
    d=NaN;
    rms=NaN;
    return
end
%reshape to get list of points
xs=reshape(xxs,[mm*nn,1]);
ys=reshape(yys,[mm*nn,1]);
zs=reshape(zzs,[mm*nn,1]);
%fit to plane
points_aug = [xs-mean(xs) ys-mean(ys) zs-mean(zs) ones(size(zs))];  % represent a point as an augmented row vector
[~,~,v] = svd(points_aug, 0);
Theta = v(:,4);
normal=Theta(1:3)/norm(Theta(1:3));
dnew=Theta(4)-dot(normal,[mean(xs),mean(ys),mean(zs)]);
%plot plane to check
Zplane=(-Theta(1)*xxs-Theta(2)*yys-dnew)/Theta(3);
%project original data point onto plane and get distance
p0=p_d;
p_orth=p0'-Theta(1:3)*(Theta(1)*p0(1)+Theta(2)*p0(2)+Theta(3)*p0(3)+dnew)/(Theta(1)^2+Theta(2)^2+Theta(3)^2);
d=sqrt((p0(1)-p_orth(1))^2 + (p0(2)-p_orth(2))^2 + (p0(3)-p_orth(3))^2);
%get plane rms
p_orth_all=[xs,ys,zs]-(points_aug*Theta*(Theta(1:3)'/sum(Theta(1:3)'*Theta(1:3))));
dists=sqrt((xs-p_orth_all(:,1)).^2 + (ys-p_orth_all(:,2)).^2+ (zs-p_orth_all(:,3)).^2);
nnn=length(dists);
rms=sqrt(sum(dists.^2)/nnn);
end



%
% Compute the surface precision by rotating points associated with a flat
% surface to horizontal and calculating both full-width-half-max (FWHM)  
% and standard deviation statistics. The FWHM statistic does not capture
% uncertainty due to a multi-modal distribution, so standard deviation is
% preferred for cross-phenomenology metric assessment.
% Inputs:
% - points - point cloud for analysis
% - rectangle - array defining rectangle corner coordinates
% - htol - horizontal tolerance added to bounding box
% - ztol - vertical tolerance added to bounding box
% Ouputs:
% - fwhm - Full-width-half-max statistic for precision
% - stdev - standard deviation statistic for precision
% - num_points - number of points cropped near the target
%
function [fwhm, stdev, num_points] = rj_surface_precision(points, rectangle, htol, ztol)
apply_rotation = 1;
[cropped_points] = crop_volume(points, rectangle, htol, ztol, apply_rotation);
% If there are no points, return zeros.
num_points = length(cropped_points);
if (num_points == 0)
    fwhm = 0.0;
    stdev = 0.0;
    return;
end
% Plot point scatter for visual inspection.
close all;
figure('Position',[50 50 500 800]);
subplot(3,1,1);
hold on;
for i=1:length(cropped_points)
    plot(cropped_points(i,1),cropped_points(i,2),'.');
end
title('XY Scatter After Rotation', 'FontWeight' , 'bold');
xlabel('X (meters)');
ylabel('Y (meters)');
hold on;
subplot(3,1,2);
hold on;
for i=1:length(cropped_points)
    plot(cropped_points(i,1),cropped_points(i,3),'.');
end
title('XZ Scatter After Rotation', 'FontWeight' , 'bold');
xlabel('X (meters)');
ylabel('Z (meters)');
subplot(3,1,3);
hold on;
for i=1:length(cropped_points)
    plot(cropped_points(i,2),cropped_points(i,3),'.');
end
title('YZ Scatter After Rotation', 'FontWeight' , 'bold');
xlabel('Y (meters)');
ylabel('Z (meters)');
hold off;
% Compute FWHM.
% The number of histogram bins is defined by the Z spread to ensure that results are stable.
zmin = min(cropped_points(:,3));
zmax = max(cropped_points(:,3));
fwhm_arr = [];
for numBins = 64:32:196
    dz = (zmax-zmin)/numBins;
    h = hist(cropped_points(:,3), numBins);
    [hmax, pos] = max(h);
    halfmax = hmax/2.0;
    sh = interp(median(h,5),10);
    fwhm = sum(sh >= halfmax) * dz / 10.0;
    fwhm_arr = [fwhm_arr, fwhm];
end
fwhm = mean(fwhm_arr);
% Compute standard deviation.
stdev = std(cropped_points(:,3));
disp(['FWHM = ' num2str(fwhm)]);
disp(['STDEV = ' num2str(stdev)]);


%
% Estimate void fraction for a point cloud given a specified GSD.
% GSD is calculated using the RJ_GSD function. For RJ, the calculation
% of void fraction is less conservative than that specified by the USGS
% and ASPRS standards because point cloud comparisons are being made for
% data sets not necessarily collected with the intent of producing a 
% gridded digital surface model and also to avoid overly penalizing any 
% point cloud source for non-uniform sampling. Points are placed in a grid 
% with bin size equal to 3xGSD and the empty bins counted as voids. The 
% input point cloud is assumed to be clipped to a North-East aligned 
% rectangle to avoid including empty border bins in the void counts. The 
% RJ test data is preconditioned to meet this expectation.
%
function [void_fraction] = rj_void_fraction(points, gsd)
% Get min/max values.
xmin = min(points(:,1));
xmax = max(points(:,1));
ymin = min(points(:,2));
ymax = max(points(:,2));
% Sample at 3xGSD to avoid small voids.
bin_size = gsd*3.0;
% Grid data with GSD.
% Place each point in any of the four bins it overlaps to avoid
% sensitivity to exact placement of the grid centers.
numxbins = ceil((xmax-xmin)/bin_size) + 1;
numybins = ceil((ymax-ymin)/bin_size) + 1;
D = zeros(numxbins, numybins);
x = round((points(:,1) - xmin)/bin_size + 0.5);
y = round((points(:,2) - ymin)/bin_size + 0.5);
for i=1:length(x)
    D(x(i),y(i)) = D(x(i),y(i)) + 1;
end
x = round((points(:,1) - xmin)/bin_size + 1);
y = round((points(:,2) - ymin)/bin_size + 0.5);
for i=1:length(x)
    D(x(i),y(i)) = D(x(i),y(i)) + 1;
end
x = round((points(:,1) - xmin)/bin_size + 0.5);
y = round((points(:,2) - ymin)/bin_size + 1);
for i=1:length(x)
    D(x(i),y(i)) = D(x(i),y(i)) + 1;
end
x = round((points(:,1) - xmin)/bin_size + 1);
y = round((points(:,2) - ymin)/bin_size + 1);
for i=1:length(x)
    D(x(i),y(i)) = D(x(i),y(i)) + 1;
end
% Clip borders to avoid any inaccuracies there.
D = D(4:numxbins-3, 4:numybins-3);
if (numel(D) == 0)
    void_fraction = Inf;
else
    D = D';
    D = flipud(D);
    void_fraction = numel(D(D == 0))/numel(D);
end
% Round off to the nearest hundredth of a percent to avoid reporting
% inconsequentially small numbers.
void_fraction = round(void_fraction * 10000.0)/10000.0;
% Plot the voids.
figure(1);
imagesc(D>0);
title('Void Image');
colormap('bone');






