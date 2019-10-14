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
    badColor = 'white'   # Color used to display NaNs

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
        keys = ['vmin','vmax']
        for key in keys:
            if key in kwargs:
                imshow_kwargs[key] = kwargs[key]

        hImg = plt.imshow(image,**imshow_kwargs)
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

    def make_instance_stoplight_charts(self, stoplight_chart, **kwargs):
        if "saveName" in kwargs:
            title = kwargs['saveName']

        if len(self.savePrefix) > 0:
            saveName = self.savePrefix + title

        fn = os.path.join(self.saveDir, saveName + self.saveExe)

        cv2.imwrite(fn, stoplight_chart)

    def make_stoplight_plot(self, fp_image=None, fn_image= None, ref=None, title='', fig=None, **kwargs):
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
        white = [255,255,255]

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

    def stretch_contrast(self, image):
        a = image.min()
        b = image.max()
        r = b-a
        if not (np.unique(image).__len__() == 1 and np.unique(image)[0] == 0):
            image = ((image - a)/r)*255
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

    def make_image_pair_plots(self, performer_pair_data_file, performer_pair_file, performer_files_chosen_file, figNum, **kwargs):
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
