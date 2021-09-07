import os
import argparse
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./', help='Model name [default: pointnet2_part_seg]')
FLAGS = parser.parse_args()


def crawlFolder(path):
    allFiles = os.listdir(os.path.abspath(path))
    all_subdirs = [d for d in allFiles if os.path.isdir(os.path.join(path, d))]
    if "opensfm" in all_subdirs:
        try:
            shutil.rmtree(os.path.join(path, "opensfm"))
            print("deleted opensfm in ", path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        return

    for dirs in all_subdirs:
        crawlFolder(os.path.join(path, dirs))
    

if __name__ == "__main__":
    baseDir = FLAGS.path
    crawlFolder(baseDir)
    

