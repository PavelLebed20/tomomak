import matplotlib.pyplot as plt


class DetectorPlotSlicer:

    def __init__(self, data, ax):
        self.ind = 0
        self.data = data
        self.ax = ax
        self.i = 0

    def redraw(self):
        new_title = 'Detector {}/{}'.format(self.ind + 1, self.data.shape[0])
        self.ax.set(title=new_title.format(self.data.shape[0]))
        self.ax.relim()
        self.ax.autoscale_view()

    def next(self, _):
        self.ind = self.ind = (self.ind + 1) % self.data.shape[0]
        self.redraw()
        plt.draw()

    def prev(self, _):
        self.ind = self.ind = (self.ind - 1) % self.data.shape[0]
        self.redraw()
        plt.draw()