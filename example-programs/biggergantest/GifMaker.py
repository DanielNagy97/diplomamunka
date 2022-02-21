import glob
import imageio


class GifMaker:
    def __init__(self):
        pass

    def make_gif(self, source_files, output_file, fps):
        anim_file = output_file

        with imageio.get_writer(anim_file, mode='I', fps=fps) as writer:
            filenames = glob.glob(source_files)
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                image = imageio.imread(filename)
                writer.append_data(image)
