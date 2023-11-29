import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def get_shape(filepath, dataset_dir):
    image = cv2.imread(os.path.join(dataset_dir,filepath), cv2.IMREAD_COLOR)
    return image.shape


def read_resize_write_one_image(filepath, dataset_dir, output_dir, shape):
    return cv2.imwrite(os.path.join(output_dir, filepath), cv2.resize(cv2.imread(os.path.join(dataset_dir,filepath), cv2.IMREAD_COLOR), [int(elem) + 1 for elem in shape[:-1]][::-1]))


def check_dataset():
    DATASET_DIR = "dataset_gion"

    filepaths = os.listdir(DATASET_DIR)

    with Pool(12) as p:
        shapes = p.starmap(get_shape, [(filepath, DATASET_DIR) for filepath in filepaths])

    shapes = np.array(shapes)
    print(f"MEAN: {shapes.mean(0)}")
    print(f"MEDIAN: {np.median(shapes, 0)}")
    print(f"MIN: {shapes.min(0)}")
    print(f"MAX: {shapes.max(0)}")

    print(shapes.argmin(0))

    import matplotlib.pyplot as plt
    heigths = [elem[0] for elem in shapes]
    widths = [elem[1] for elem in shapes]

    plt.hist(heigths, bins=len(shapes))
    plt.title("hieghts")
    plt.show()

    plt.hist(widths, bins=len(shapes))
    plt.title("widths")
    plt.show()
    
    cv2.imshow("minheight", cv2.imread(os.path.join(DATASET_DIR, filepaths[shapes.argmin(0)[0]])))
    cv2.imshow("minwidth", cv2.imread(os.path.join(DATASET_DIR, filepaths[shapes.argmin(0)[1]])))
    cv2.waitKey()
    cv2.destroyAllWindows()


def resize_dataset():
    DATASET_DIR = "dataset_2"
    OUTPUT_DIR = "dataset_gion"

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    filepaths = os.listdir(DATASET_DIR)

    with Pool(12) as p:
        shapes = p.starmap(get_shape, [(filepath, DATASET_DIR) for filepath in filepaths])

        median = np.median(shapes, 0)
        median = (int(np.mean([median[0], median[1]])), int(np.mean([median[0], median[1]])), 69)
        p.starmap(read_resize_write_one_image, [(filepath, DATASET_DIR, OUTPUT_DIR, median) for filepath in filepaths])

def main():
    resize_dataset()
    check_dataset()


if __name__ == "__main__":
    main()