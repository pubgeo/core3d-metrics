import os
import platform
import numpy as np
import matplotlib  as mpl
from core3dmetrics.geometrics.image_pair_plot import ImagePairPlot, ImagePair

if os.getenv('DISPLAY') is None and not platform.system() == "Windows":
    import matplotlib

    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import cv2

try:
    from cv2 import cv2
except ImportError:
    pass


class plot:
    saveDir = '.'
    savePrefix = ''
    saveExe = '.png'

    defaultCM = 'jet'
    badColor = 'white'  # Color used to display NaNs

    showPlots = True
    autoSave = False  # Saves figure at end of call to plot.make()
    dpi = 500

    def __init__(self, **kwargs):

        if 'showPlots' in kwargs:
            self.showPlots = kwargs['showPlots']

        if 'saveDir' in kwargs:
            self.saveDir = kwargs['saveDir']

        if 'autoSave' in kwargs:
            self.autoSave = kwargs['autoSave']

        if 'savePrefix' in kwargs:
            self.savePrefix = kwargs['savePrefix']

        if 'badColor' in kwargs:
            self.badColor = kwargs['badColor']

        if 'cmap' in kwargs:
            self.defaultCM = kwargs['cmap']

        if 'dpi' in kwargs:
            self.dpi = kwargs['dpi']

        if (os.getenv('DISPLAY') is None) and self.showPlots:
            if not platform.system() == "Windows":
                print('DISPLAY not set.  Disabling plot display')
                self.showPlots = False

        plt.rcParams['image.cmap'] = self.defaultCM

        print("showPlots = " + str(self.showPlots))

    def make(self, image=None, title='', fig=None, **kwargs):

        if 'badValue' in kwargs:
            image = np.array(image)
            image[image == kwargs['badValue']] = np.nan

        plt.figure(fig)
        plt.clf()
        plt.title(title)

        # When no image is provided, setup figure and return handle to matplotlib
        if image is None:
            return plt

        imshow_kwargs = {}
        keys = ['vmin', 'vmax']
        for key in keys:
            if key in kwargs:
                imshow_kwargs[key] = kwargs[key]

        hImg = plt.imshow(image, **imshow_kwargs)
        mpl.cm.get_cmap().set_bad(color=self.badColor)

        if 'cmap' in kwargs:
            cmap = kwargs['cmap']

            # Sometimes matplotlib fails on saving the figure do to
            # rgba values out of bounds [0 1]. Setting the max/min
            # with a tolerance seems to work
            if type(cmap) is list:
                cmap = np.array(cmap)
                cmap[cmap >= 1.0] = 0.9999
                cmap[cmap <= 0.0] = 0.0001
                cmap = cmap.tolist()
                cmap = mpl.colors.ListedColormap(cmap)
            hImg.set_cmap(cmap)

        if 'colorbar' in kwargs:
            if kwargs['colorbar'] is True:
                hCM = plt.colorbar()

                if 'cm_ticks' in kwargs:
                    hCM.set_ticks(kwargs['cm_ticks'], True)

                if 'cm_labels' in kwargs:
                    hCM.set_ticklabels(kwargs['cm_labels'], True)

        if self.showPlots:
            plt.show(block=False)

        if self.autoSave:
            if "saveName" in kwargs:
                title = kwargs['saveName']
            self.save(title)

        if not self.showPlots:
            plt.close(plt.gcf())

    def make_distance_histogram(self, data, fig=None, bin_width=None, bins=None, plot_title=''):
        if len(self.savePrefix) > 0:
            saveName = self.savePrefix + plot_title
        plt.figure(fig)
        plt.clf()
        plt.xlim(data.min(), data.max())
        n, bins_out, patches = plt.hist(data, histtype='stepfilled', bins=int(bins))
        plt.xlabel("Distance (m)")
        plt.ylabel("Counts")
        plt.title(plot_title)
        plt.legend(["Histogram\nBin Width: " + str(bin_width) + " m"], loc='best')
        filename = os.path.join(self.saveDir, saveName + self.saveExe)
        np.savetxt(os.path.join(self.saveDir, saveName + ".csv"), data)
        plt.savefig(filename, dpi=self.dpi)

    def make_instance_stoplight_charts(self, stoplight_chart, **kwargs):
        if "saveName" in kwargs:
            title = kwargs['saveName']

        if len(self.savePrefix) > 0:
            saveName = self.savePrefix + title

        fn = os.path.join(self.saveDir, saveName + self.saveExe)

        cv2.imwrite(fn, stoplight_chart)

    def make_stoplight_plot(self, fp_image=None, fn_image=None, ref=None, title='', fig=None, **kwargs):
        if ref is None:
            return plt
        plt.figure(fig)
        plt.clf()
        plt.title(title)
        if 'badValue' in kwargs:
            fp_image = np.array(fp_image)
            fp_image[fp_image == kwargs['badValue']] = np.nan
            fn_image = np.array(fn_image)
            fn_image[fp_image == kwargs['badValue']] = np.nan
        if fp_image is None or fn_image is None:
            return plt
        # Create the image
        if fp_image.shape != fn_image.shape:
            raise ValueError("Dimension mismatch")
        stoplight_chart = np.multiply(np.ones((fp_image.shape[0], fp_image.shape[1], 3), dtype=np.uint8), 220)
        red = [255, 0, 0]
        black = [0, 0, 0]
        blue = [0, 0, 255]
        white = [255, 255, 255]

        ref = np.uint8(ref)

        if cv2.__version__[0] == "4":
            contours, hierarchy = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, hierarchy = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(stoplight_chart, contours, -1, black, 2)
        cv2.fillPoly(stoplight_chart, contours, white)
        fp_image_8 = np.uint8(fp_image)
        fn_image_8 = np.uint8(fn_image)
        stoplight_chart[fp_image_8 == 1] = blue
        stoplight_chart[fn_image_8 == 1] = red

        if "saveName" in kwargs:
            title = kwargs['saveName']

        if len(self.savePrefix) > 0:
            saveName = self.savePrefix + title

        fn = os.path.join(self.saveDir, saveName + self.saveExe)

        cv2.imwrite(fn, stoplight_chart[..., ::-1])
        return fn

    def stretch_contrast(self, image):
        a = image.min()
        b = image.max()
        r = b - a
        if not (np.unique(image).__len__() == 1 and np.unique(image)[0] == 0):
            image = ((image - a) / r) * 255
        image = np.uint8(image)
        return image

    def make_error_map(self, error_map=None, ref=None, title='', fig=None, ignore=None, **kwargs):
        if ref is None:
            return plt
        plt.figure(fig)
        plt.clf()
        plt.title(title)
        if 'badValue' in kwargs:
            error_map = np.array(error_map)
            error_map[error_map == kwargs['badValue']] = np.nan
        # Edit error map for coloring
        black = [0, 0, 0]
        gray = [220, 220, 220]
        # Trucnate errors to +-5 meters
        error_ground_track = np.nan_to_num(error_map)
        error_map_temp = error_map
        error_map_temp[error_map < -5] = -5
        error_map_temp[error_map > 5] = 5
        error_map_temp = np.nan_to_num(error_map_temp)
        error_map_temp = self.stretch_contrast(error_map_temp)
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        # Hue
        lut[0:127, 0, 0] = 0  # Red Hue
        lut[128:256, 0, 0] = 120  # Blue Hue
        # Saturation
        lut[0:127, 0, 1] = np.uint8(np.linspace(255, 0, num=127))  # Red Hue
        lut[128:256, 0, 1] = np.uint8(np.linspace(0, 255, num=128))  # Blue Hue
        # Value
        lut[:, 0, 2] = 255  # Always max value
        lut = cv2.cvtColor(lut, cv2.COLOR_HSV2BGR)
        # Apply colormap
        err_color = cv2.applyColorMap(error_map_temp, lut)
        err_color[error_ground_track == 0] = gray

        ref = np.uint8(ref)

        if cv2.__version__[0] == "4":
            contours, hierarchy = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, hierarchy = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(err_color, contours, -1, black, 2)

        if "saveName" in kwargs:
            title = kwargs['saveName']

        if len(self.savePrefix) > 0:
            saveName = self.savePrefix + title

        fn = os.path.join(self.saveDir, saveName + self.saveExe)

        cv2.imwrite(fn, err_color)
        return fn

    def make_iou_histogram(self, iou_list, sort_type='', title='', fig=None, width=4000, **kwargs):
        plt.figure(fig)
        plt.clf()
        plt.title(title)
        # Create bar plot
        plt.bar(iou_list.keys(), iou_list.values(), width, color='g', edgecolor='k', linewidth=1)
        plt.xlabel(sort_type)
        plt.ylabel('IOU')
        if self.showPlots:
            plt.show(block=False)

        if self.autoSave:
            if "saveName" in kwargs:
                title = kwargs['saveName']
            self.save(title)

        if not self.showPlots:
            plt.close(plt.gcf())

    def make_iou_scatter(self, iou_list, sort_type='', title='', fig=None, width=4000, **kwargs):
        ax = plt.figure(fig)
        plt.clf()
        plt.title(title)
        # Create bar plot
        plt.scatter(iou_list.keys(), iou_list.values(), s=20, color='r', edgecolor='black')
        plt.ylim(0, 1.0)
        plt.xlim(1, 10000000)
        plt.xscale('log')
        plt.xlabel(sort_type)
        plt.ylabel('IOU')
        if self.showPlots:
            plt.show(block=False)

        if self.autoSave:
            if "saveName" in kwargs:
                title = kwargs['saveName']
            self.save(title)

        if not self.showPlots:
            plt.close(plt.gcf())

    def make_obj_error_map(self, error_map=None, ref=None, title='', fig=None, **kwargs):
        if ref is None:
            return plt
        plt.figure(fig)
        plt.clf()
        plt.title(title)
        if 'badValue' in kwargs:
            error_map = np.array(error_map)
            error_map[error_map == kwargs['badValue']] = np.nan
        # Edit error map for coloring
        black = [0, 0, 0]
        gray = [220, 220, 220]
        dark_blue = [205, 0, 0]
        # Trucnate errors to +5 meters
        error_ground_track = np.nan_to_num(error_map)
        error_map_temp = error_map.copy()
        error_map_zeros = error_map.copy()
        error_map_zeros[error_map == 0] = 300  # some big number over 255
        error_map_zeros = np.uint16(error_map_zeros)
        error_map_temp[error_map > 2] = 2
        error_map_temp = np.nan_to_num(error_map_temp)
        error_map_temp = self.stretch_contrast(error_map_temp)

        # Apply colormap
        err_color = cv2.applyColorMap(error_map_temp, cv2.COLORMAP_JET)
        err_color[error_ground_track == 0] = gray
        err_color[error_map_zeros == 300] = dark_blue

        ref = np.uint8(ref)

        if cv2.__version__[0] == "4":
            contours, hierarchy = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, hierarchy = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(err_color, contours, -1, black, 2)

        if "saveName" in kwargs:
            title = kwargs['saveName']

        if len(self.savePrefix) > 0:
            saveName = self.savePrefix + title

        fn = os.path.join(self.saveDir, saveName + self.saveExe)

        cv2.imwrite(fn, err_color)

    def save(self, saveName, figNum=None):

        if saveName is None:
            return

        if figNum is not None:
            plt.figure(figNum)

        if len(self.savePrefix) > 0:
            saveName = self.savePrefix + saveName

        fn = os.path.join(self.saveDir, saveName + self.saveExe)
        plt.savefig(fn, dpi=self.dpi)

    def make_image_pair_plots(self, performer_pair_data_file, performer_pair_file, performer_files_chosen_file, figNum,
                              **kwargs):
        if performer_pair_data_file is None or performer_pair_file is None or performer_files_chosen_file is None:
            return
        data_file = performer_pair_data_file
        image_pair_file = performer_pair_file
        image_pair_plot = ImagePairPlot(data_file, image_pair_file, performer_files_chosen_file)

        plt.set_cmap('viridis')
        image_pair_plot.create_plot(figNum)

        if self.showPlots:
            plt.show(block=False)

        if self.autoSave:
            if "saveName" in kwargs:
                title = kwargs['saveName']
            self.save(title)

        if not self.showPlots:
            plt.close(plt.gcf())

        plt.set_cmap('jet')

    def make_final_metrics_images(self, stoplight_fn, errhgt_fn, test_conf_filename, cls_iou_fn, cls_z_iou_fn,
                                  cls_z_slope_fn, ref_cls, output_dir):
        # TODO: “metrics.png”
        # Top 3: 2D IOU, 3D IOU, CONF VIZ
        # Bottom 3: CLS IOU, CLS+Z IOU, and CLS+Z+SLOPE IOU
        from PIL import Image
        from pathlib import Path
        Image.MAX_IMAGE_PIXELS = None

        iou_2d_path = Path(stoplight_fn)
        iou_2d_image = Image.open(str(iou_2d_path.absolute()))
        iou_3d_path = Path(errhgt_fn)
        iou_3d_image = Image.open(str(iou_3d_path.absolute()))
        stoplight_shape = np.shape(iou_2d_image)
        if test_conf_filename.is_file():
            conf_image = Image.open(test_conf_filename)
            conf_viz_image = conf_image.resize((stoplight_shape[1], stoplight_shape[0]), resample=0)
        else:
            print("Can't find image. Aborting metrics.png creation...")
            return

        # Recolor conf_viz_image
        background_color = (0, 0, 0)
        gray_color = (220, 220, 220)
        red_color = (255, 0, 0)
        blue_color = (0, 0, 255)
        conf_viz_image_array = np.array(conf_viz_image)
        conf_viz_image_array[np.all(conf_viz_image_array == background_color, axis=-1)] = (gray_color)
        conf_viz_image_recolor = Image.fromarray(np.uint8(conf_viz_image_array))

        def assign_colors_to_images(metrics_image, ref_cls):
            array = np.array(metrics_image)
            correct = (array == 1) & (ref_cls == 6)
            incorrect = (array != 1) & (ref_cls == 6)

            map_image = np.zeros(array.shape)
            map_image[correct == True] = 1
            map_image[incorrect == True] = 6
            map_image = np.uint8(map_image)

            x, y = array.shape
            rgb_image = np.zeros((x, y, 3))
            rgb_image[map_image == 0] = [220, 220, 220]
            rgb_image[map_image == 1] = [0, 0, 255]
            rgb_image[map_image == 6] = [255, 0, 0]

            rgb_PIL = Image.fromarray(np.uint8(rgb_image), 'RGB')
            return rgb_PIL

        cls_iou_image = Image.open(cls_iou_fn).convert("L")
        cls_iou_image_rgb = assign_colors_to_images(cls_iou_image, ref_cls)

        cls_z_iou = Image.open(cls_z_iou_fn).convert("L")
        cls_z_iou_rgb = assign_colors_to_images(cls_z_iou, ref_cls)

        cls_z_slope = Image.open(cls_z_slope_fn).convert("L")
        cls_z_slope_rgb = assign_colors_to_images(cls_z_slope, ref_cls)

        # Create image mosaic/stack
        num_rows, num_cols, ch_num = np.shape(iou_2d_image)
        separation_bar_vert = np.ones([num_rows, int(np.floor(num_cols * 0.02)), ch_num], dtype=np.uint8)
        separation_bar_vert.fill(255)
        # Stack images horizontally
        image_stack_top = np.hstack((iou_2d_image, separation_bar_vert, iou_3d_image, separation_bar_vert,
                                     conf_viz_image_recolor))
        # TODO: Change bot stack to clz z images
        image_stack_bot = np.hstack((cls_iou_image_rgb, separation_bar_vert, cls_z_iou_rgb, separation_bar_vert,
                                     cls_z_slope_rgb))

        num_rows, num_cols, ch_num = np.shape(image_stack_top)
        separation_bar_horz = np.ones([int(np.floor(num_rows * 0.02)), num_cols, ch_num], dtype=np.uint8)
        separation_bar_horz.fill(255)

        # Stack images vertically
        final_stack = np.vstack((image_stack_top, separation_bar_horz, image_stack_bot))
        import cv2
        cv2.imwrite(str(Path(output_dir, "metrics.png").absolute()), cv2.cvtColor(final_stack, cv2.COLOR_BGR2RGB))

    def make_final_input_images_grayscale(self, plot_1, plot_2, plot_3, plot_4, plot_5, plot_6, output_dir):

        from PIL import Image
        from pathlib import Path
        Image.MAX_IMAGE_PIXELS = None

        plot_1_image = Image.fromarray(plot_1)
        plot_2_image = Image.fromarray(plot_2)
        plot_3_image = Image.fromarray(plot_3)
        plot_4_image = Image.fromarray(plot_4)
        plot_5_image = Image.fromarray(plot_5)
        plot_6_image = Image.fromarray(plot_6)

        # colorize_no_Data
        def colorize_image(input_img, minval=None, maxval=None, is_cls=False):
            if is_cls:
                building_map = input_img == 6
                not_building_map = np.bitwise_and((input_img != 6), (input_img != 65))
            nan_map = np.isnan(input_img)
            if minval == None and maxval == None:
                maxval = np.nanpercentile(input_img, 95)
                minval = np.nanpercentile(input_img, 1)
            input_img[input_img < minval] = minval
            input_img[input_img > maxval] = maxval
            input_img[np.isnan(input_img)] = minval
            input_img = input_img - minval
            input_img = (255.0 * (input_img / (maxval-minval))).astype(np.uint8)

            img = np.zeros((input_img.shape[0], input_img.shape[1], 3))
            img[:, :, 0] = np.copy(input_img)
            img[:, :, 1] = np.copy(input_img)
            img[:, :, 2] = np.copy(input_img)

            img[nan_map, 0] = 135.0
            img[nan_map, 1] = 206.0
            img[nan_map, 2] = 250.0

            if is_cls:
                img[building_map, 0] = 230
                img[building_map, 1] = 0
                img[building_map, 2] = 0
                img[not_building_map, 0] = 0
                img[not_building_map, 1] = 0
                img[not_building_map, 2] = 0

            img = img.astype(np.uint8)
            return img, minval, maxval

        def nodata_to_nan(img):
            nodata = -9999
            img[img == nodata] = np.nan
            nodata = -10000
            img[img == nodata] = np.nan
            return (img)

        # Convert PIL image to nparray and insert np.nan
        # Uint8
        plot_1_image = np.array(plot_1_image)
        plot_4_image = np.array(plot_4_image)
        # Floats
        plot_2_image = nodata_to_nan(np.array(plot_2_image))
        plot_3_image = nodata_to_nan(np.array(plot_3_image))
        plot_5_image = nodata_to_nan(np.array(plot_5_image))
        plot_6_image = nodata_to_nan(np.array(plot_6_image))

        # Colorize images with same scale
        plot_1_image, minval, maxval = colorize_image(plot_1_image, is_cls=True)
        plot_4_image, _, _ = colorize_image(plot_4_image, minval, maxval, is_cls=True)
        plot_2_image, minval, maxval = colorize_image(plot_2_image)
        plot_3_image, _, _ = colorize_image(plot_3_image, minval, maxval)
        plot_5_image, minval, maxval = colorize_image(plot_5_image)
        plot_6_image, _, _ = colorize_image(plot_6_image, minval, maxval)

        # Convert back to PIL images
        plot_1_image = Image.fromarray(plot_1_image)
        plot_4_image = Image.fromarray(plot_4_image)
        plot_2_image = Image.fromarray(plot_2_image)
        plot_3_image = Image.fromarray(plot_3_image)
        plot_5_image = Image.fromarray(plot_5_image)
        plot_6_image = Image.fromarray(plot_6_image)

        # Autocontrast
        # from PIL import ImageOps
        # plot_2_image = ImageOps.autocontrast(plot_2_image)
        # plot_5_image = ImageOps.autocontrast(plot_5_image)
        # plot_3_image = ImageOps.autocontrast(plot_3_image)
        # plot_6_image = ImageOps.autocontrast(plot_6_image)

        # Create image mosaic/stack
        try:
            num_rows, num_cols, ch_num = np.shape(plot_2_image)
        except ValueError:
            num_rows, num_cols = np.shape(plot_2_image)

        # Resize test images to same size as ref images
        plot_1_image = plot_1_image.resize((num_cols, num_rows), resample=0)
        plot_3_image = plot_3_image.resize((num_cols, num_rows), resample=0)
        plot_4_image = plot_4_image.resize((num_cols, num_rows), resample=0)
        plot_5_image = plot_5_image.resize((num_cols, num_rows), resample=0)
        plot_6_image = plot_6_image.resize((num_cols, num_rows), resample=0)

        # Create seperation bar
        separation_bar_vert = np.ones([num_rows, int(np.floor(num_cols * 0.02)), ch_num], dtype=np.uint8)
        separation_bar_vert.fill(255)

        # Stack images horizontally
        image_stack_top = np.hstack((plot_1_image, separation_bar_vert, plot_2_image, separation_bar_vert,
                                     plot_3_image))

        image_stack_bot = np.hstack((plot_4_image, separation_bar_vert, plot_5_image, separation_bar_vert,
                                     plot_6_image))

        num_rows, num_cols, ch = np.shape(image_stack_top)
        separation_bar_horz = np.ones([int(np.floor(num_rows * 0.02)), num_cols, ch_num], dtype=np.uint8)
        separation_bar_horz.fill(255)

        # Stack images vertically
        final_stack = np.vstack((image_stack_top, separation_bar_horz, image_stack_bot))
        import cv2
        cv2.imwrite(str(Path(output_dir, "input.png").absolute()), cv2.cvtColor(final_stack, cv2.COLOR_BGR2RGB))

    def make_final_input_images_rgb(self, plot_fn: list, output_dir):

        plot_1 = plot_fn[0]
        plot_2 = plot_fn[1]
        plot_3 = plot_fn[2]
        plot_4 = plot_fn[3]
        plot_5 = plot_fn[4]
        plot_6 = plot_fn[5]

        from PIL import Image
        from pathlib import Path

        plot_1_path = Path(plot_1)
        plot_1_image = Image.open(str(plot_1_path.absolute()))
        plot_2_path = Path(plot_2)
        plot_2_image = Image.open(str(plot_2_path.absolute()))
        plot_3_path = Path(plot_3)
        plot_3_image = Image.open(str(plot_3_path.absolute()))
        plot_4_path = Path(plot_4)
        plot_4_image = Image.open(str(plot_4_path.absolute()))
        plot_5_path = Path(plot_5)
        plot_5_image = Image.open(str(plot_5_path.absolute()))
        plot_6_path = Path(plot_6)
        plot_6_image = Image.open(str(plot_6_path.absolute()))

        # Create image mosaic/stack
        num_rows, num_cols, ch_num = np.shape(plot_1_image)

        # Resize test images to same size as ref images
        plot_4_image = plot_4_image.resize((num_cols, num_rows), resample=0)
        plot_5_image = plot_5_image.resize((num_cols, num_rows), resample=0)
        plot_6_image = plot_6_image.resize((num_cols, num_rows), resample=0)

        separation_bar_vert = np.ones([num_rows, int(np.floor(num_cols * 0.02)), ch_num], dtype=np.uint8)
        separation_bar_vert.fill(255)
        # Stack images horizontally
        image_stack_top = np.hstack((plot_1_image, separation_bar_vert, plot_2_image, separation_bar_vert,
                                     plot_3_image))
        # TODO: Change bot stack to clz z images
        image_stack_bot = np.hstack((plot_4_image, separation_bar_vert, plot_5_image, separation_bar_vert,
                                     plot_6_image))

        num_rows, num_cols, ch_num = np.shape(image_stack_top)
        separation_bar_horz = np.ones([int(np.floor(num_rows * 0.02)), num_cols, ch_num], dtype=np.uint8)
        separation_bar_horz.fill(255)

        # Stack images vertically
        final_stack = np.vstack((image_stack_top, separation_bar_horz, image_stack_bot))
        import cv2
        cv2.imwrite(str(Path(output_dir, "textured.png").absolute()), cv2.cvtColor(final_stack, cv2.COLOR_BGR2RGB))