import matplotlib.pyplot as plt
import imageio

class GIFVisualizer(object):
    def __init__(self):
        self.frames = []

    def set_data(self, img):
        self.frames.append(img)

    def reset(self):
        self.frames = []

    def repeat_last_frame(self):
        # repeat last frame for clear view of the final state
        self.frames.append(self.frames[-1]) 

    def get_gif(self, filename='pushing_visualization.gif'):
        # generate the gif
        print(f"Creating animated gif {filename}...")
        imageio.mimsave(filename, self.frames, format="GIF", duration=0.01)
        print("Done!")
        return filename


class NotebookVisualizer(object):
    def __init__(self, fig, hfig):
        self.fig = fig
        self.hfig = hfig

    def set_data(self, img):
        plt.clf()
        plt.imshow(img)
        plt.axis('off')
        self.fig.canvas.draw()
        self.hfig.update(self.fig)

    def reset(self):
        pass