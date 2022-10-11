import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from progress import print_progress_bar


# TODO: Adjust the following variables
# the path to the models/research repo
sys.path.insert(0, '../models/research')
from object_detection.utils import dataset_util

# the labels in your dataset in correct order
ValidLabels = ["Car", "Pedestrian", "Truck", "Van", "Cyclist", "Tram", "DontCare", "Misc", "Person_sitting"]

# the directory containing all the images and respective .txt files
DatasetDirectory = "./Kitti/alltest/"  # MUST end with '/'
# the directory in which you want the .record to be saved
OutputDirectory = "./"  # MUST end with '/'
# the desired filename of the .record file
Output_filename = "record/tf_record_eval.record"


def create_tf_dataset_entry(filename, encoded_image_data, xmins, xmaxs, ymins, ymaxs,
                            classes_text, classes, image_height, image_width):
    """Build the dataset entry
    Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        encoded_image_data: encoded image bytes
        xmins: List of normalized left x coordinates in bounding box (1 per box)
        xmaxs: List of normalized right x coordinates in bounding box (1 per box)
        ymins: List of normalized top y coordinates in bounding box (1 per box)
        ymaxs: List of normalized bottom y coordinates in bounding box (1 per box)
        classes_text: List of string class name of bounding box (1 per box)
        classes: List of integer class id of bounding box (1 per box)
        image_height:  height of the image
        image_width:  width of the image
    """
    filename = filename.encode('utf-8')
    tf_dataset_entry = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(b'JPEG'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(x.encode('utf-8') for x in classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_dataset_entry


def read_bboxes_from_label_txt(filename):
    """Read bounding boxes from label.txt in KITTI format
    Args:
        filename: string, path to an image file, e.g., '/path/to/example.txt'
    Returns:
        bboxes: list of bounding boxes in the file
        labels: list of labels in the file
    """
    labels = []
    bboxes = []

    # strip line ending
    lines = [line.rstrip('\n') for line in open(filename, 'r')]

    for line in lines:
        # process each line of the label file
        label = convert_line_to_label(line)
        bbox = label['bbox']

        # add the current label to the list of found labels
        labels.append(label)
        bboxes.append(bbox)

    return bboxes, labels


def convert_line_to_label(line, label_type='KITTI'):
    """ extract variables from one string label entry (one line of text)
    Args:
        line: string, containing the label data
        label_type: type of the dataset format
    Returns:
        label: the encoded variables in a dictionary
    """
    content = line.split(' ')

    if label_type == 'KITTI':
        label_name = content[0]
        truncated = int(float(content[1]))
        occ_state = int(content[2])
        alpha = float(content[3])
        bbox = {'left': float(content[4]),
                'top': float(content[5]),
                'right': float(content[6]),
                'bottom': float(content[7])
                }
        dim = 0
        loc = 0
        score = 0
    else:
        print("provided label type is not implemented")
        sys.exit()

    label = {
        'label_name': label_name,
        'truncated': truncated,
        'occlusion_state': occ_state,
        'alpha_angle': alpha,
        'bbox': bbox,
        'dimensions': dim,
        'location': loc,
        'rotation': loc,
        'score': score
    }

    return label


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        """Convert png to jpeg image format
        Args:
          image_data: the image in png format
        Returns:
          the image in jpeg format
        """
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
      filename: string, path of the image file.
    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename


def process_image(filename, coder):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    print(filename)
    filename2 = filename.encode('utf-8')
    image_data = tf.gfile.FastGFile(filename2, 'rb').read()

    # Convert any PNG to JPEG's for consistency.
    if is_png(filename):
        # print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def read_plot_images_from_records(tf_records_filename, coder):
    """Process a single image file.
    Args:
      tf_records_filename: string, path to the records file
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # init reader and read data
    reader = tf.python_io.tf_record_iterator(tf_records_filename)
    dataset_entries = [tf.train.Example().FromString(example_str)
                       for example_str in reader]

    # plot the first x images from the .tfrecords file
    for i in range(10):
        # Create figure
        fig = plt.figure(frameon=False)
        axes = plt.Axes(fig, [0., 0., 1., 1.])
        axes.set_axis_off()
        fig.add_axes(axes)

        dataset_entry = dataset_entries[i]

        read_image_bytes = dataset_entry.features.feature['image/encoded'].bytes_list.value[0]
        read_image_height = list(
            dataset_entry.features.feature['image/height'].int64_list.value)[0]
        read_image_width = list(
            dataset_entry.features.feature['image/width'].int64_list.value)[0]

        # decode the image
        read_image = coder.decode_jpeg(read_image_bytes)

        bboxes = []
        for j in range(100):  # this is not optimal, but works
            try:
                bbox = {
                    'left': list(dataset_entry.features.feature
                                 ['image/object/bbox/xmin'].float_list.value)[j]*read_image_width,
                    'right': list(dataset_entry.features.feature
                                  ['image/object/bbox/xmax'].float_list.value)[j]*read_image_width,
                    'top': list(dataset_entry.features.feature
                                ['image/object/bbox/ymin'].float_list.value)[j]*read_image_height,
                    'bottom': list(dataset_entry.features.feature
                                   ['image/object/bbox/ymax'].float_list.value)[j]*read_image_height,
                }
                bboxes.append(bbox)
            except:
                #print('there were ', str(j), ' bounding boxes found')
                break

        # plot the image
        axes.imshow(read_image)

        # add each bounding box
        for bbox in bboxes:
            # Create the rectangle
            rect = patches.Rectangle((bbox['left'], bbox['bottom']), bbox['right'] - bbox['left'],
                                     bbox['top'] - bbox['bottom'], linewidth=0.5, edgecolor="b",
                                     facecolor='none')
            # Add the rectangle to the Axes
            axes.add_patch(rect)

        # Remove ticks from the plot.
        axes.set_xticks([])
        axes.set_yticks([])

        # render
        plt.show()


def write_images_to_records(output_file, coder):
    """Encode all dataset entries (images and .txt files) into one records file
    Args:
      output_file: path to where the records file should be saved to
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    """
    writer = tf.python_io.TFRecordWriter(output_file)

    print('writing tfRecords...')
    progress = 0
    # go through every image in the directory
    for filename in os.listdir(DatasetDirectory):
        if filename.endswith(".png"):
            filename_image = os.path.join(DatasetDirectory, filename)
            filename_label = os.path.join(DatasetDirectory, filename[0:-4] + ".txt")
        elif filename.endswith(".jpg"):
            filename_image = os.path.join(DatasetDirectory, filename)
            filename_label = os.path.join(DatasetDirectory, filename[0:-4] + ".txt")
        elif filename.endswith(".jpeg"):
            filename_image = os.path.join(DatasetDirectory, filename)
            filename_label = os.path.join(DatasetDirectory, filename[0:-5] + ".txt")
        else:
            continue

        # encode the image
        image_buffer, image_height, image_width = process_image(filename_image, coder)

        # load valid labels
        bboxes, labels = read_bboxes_from_label_txt(filename_label)

        xmins = []  # List of left x coordinates in bounding box (1 per box)
        xmaxs = []  # List of right x coordinates in bounding box (1 per box)
        ymins = []  # List of top y coordinates in bounding box (1 per box)
        ymaxs = []  # List of bottom y coordinates in bounding box (1 per box)
        classes_text = []  # List of string class name of bounding box (1 per box)
        classes = []  # List of integer class id of bounding box (1 per box)

        # go through every bounding box
        for i in enumerate(bboxes):
            xmins.append(bboxes[i[0]]['left']/image_width)
            xmaxs.append(bboxes[i[0]]['right']/image_width)
            ymins.append(bboxes[i[0]]['top']/image_height)
            ymaxs.append(bboxes[i[0]]['bottom']/image_height)
            classes_text.append(labels[i[0]]['label_name'])
            classes.append(ValidLabels.index(classes_text[i[0]]))  # this gets the index of the label as integer

        # save the data to the .records file
        tf_dataset_entry = create_tf_dataset_entry(filename, image_buffer, xmins, xmaxs, ymins, ymaxs,
                                                   classes_text, classes, image_height, image_width)
        writer.write(tf_dataset_entry.SerializeToString())

        progress += 1
        print_progress_bar(progress, len(os.listdir(DatasetDirectory))/2)
    writer.close()


def main(_):
    # some init stuff
    output_file = os.path.join(OutputDirectory, Output_filename)
    print ("data will be saved to: ", output_file)
    coder = ImageCoder()

    # encode and write all the data to records file
    write_images_to_records(output_file, coder)
    # read in the TFRecords file and show entries
    read_plot_images_from_records(output_file, coder)

    return


if __name__ == '__main__':
    tf.app.run()