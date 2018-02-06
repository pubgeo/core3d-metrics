import os

import numpy as np

import matplotlib  as mpl
if os.getenv('DISPLAY') is None:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt



class plot:

    saveDir = '.'
    savePrefix = ''
    saveExe = '.png'

    defaultCM = 'jet'
    badColor = 'white'   # Color used to display NaNs

    showPlots = True
    autoSave = False  # Saves figure at end of call to plot.make()

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
            
        if (os.getenv('DISPLAY') is None) and self.showPlots:
            print('DISPLAY not set.  Disabling plot display')
            self.showPlots = False
            
        plt.rcParams['image.cmap'] = self.defaultCM

        print("showPlots = " + str(self.showPlots))

    def make(self, image, title='', fig=None, **kwargs):
    
        if 'badValue' in kwargs:
            image = np.array(image)
            image[image == kwargs['badValue']] = np.nan
                    
        plt.figure(fig)
        plt.clf()
        hImg = plt.imshow(image)
        plt.title(title)

        mpl.cm.get_cmap().set_bad(color=self.badColor)

        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
            if type(cmap) is list:
                cmap = mpl.colors.ListedColormap(cmap)
            hImg.cmap = cmap

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




    def save(self, saveName, figNum=None):

        if saveName is None:
            return

        if figNum is not None:
            plt.figure(figNum)

        if len(self.savePrefix) > 0:
            saveName  = self.savePrefix + saveName

        fn  = os.path.join(self.saveDir, saveName + self.saveExe)
        plt.savefig(fn)