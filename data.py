import multiprocessing
import tensorflow as tf
import tensorlayer as tl
# # enable debug logging
# tl.logging.set_verbosity(tl.logging.DEBUG)
# tl.logging.set_verbosity(tl.logging.DEBUG)

class FLAGS(object):
    def __init__(self):
        self.batch_size = 1
        self.n_epoch = 200
        self.lr_init = 0.0002
        self.beta_1 = 0.5
        self.shuffle_buffer_size = 128
        self.model_dir = 'models' # folder name to save models
        self.sample_dir = 'samples' # folder name to save visualized results

flags = FLAGS()
tl.files.exists_or_mkdir(flags.model_dir, verbose=False)
tl.files.exists_or_mkdir(flags.sample_dir, verbose=False)

def get_data(images):
    def generator_fn():
        for image in images:
            yield image
    # def prepro_fn(x):
    #
    #     # https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/train#L82 Hao: 需要和他一样吗？这个也不是官方的
    #
    #     M_rotate = tl.prepro.affine_rotation_matrix(angle=(-16, 16))
    #     M_flip = tl.prepro.affine_horizontal_flip_matrix(prob=0.5)
    #     M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.8, 1.2))
    #     h, w, _ = x.shape
    #     M_combined = M_zoom.dot(M_flip).dot(M_rotate)
    #     transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)
    #     x = tl.prepro.affine_transform_cv2(x, transform_matrix, border_mode='replicate')
    #         # x = tl.prepro.flip_axis(x, axis=1, is_random=True)
    #         # x = tl.prepro.rotation(x, rg=16, is_random=True, fill_mode='nearest')
    #     # x = tl.prepro.imresize(x, size=[int(h * 1.2), int(w * 1.2)], interp='bicubic', mode=None)
    #     # x = tl.prepro.crop(x, wrg=256, hrg=256, is_random=True)
    #     x = x / 127.5 - 1.
    #     return x
    def _map_fn(x):
        # x = tf.numpy_function(prepro_fn, [x], [tf.float32]) # slow
        # return x[0]
        x.set_shape([256, 256, 3])
        x = tf.image.resize(x, size=[int(256*1.12), int(256*1.12)])
        x = tf.image.random_crop(x, size=[256, 256, 3])
        x = tf.image.random_flip_left_right(x)
        x = x / 127.5 - 1.
        return x

    ds = tf.data.Dataset.from_generator(
        generator_fn, output_types=(tf.float32))
    ds = ds.shuffle(flags.shuffle_buffer_size)
    # ds = ds.repeat(n_epoch)
    ds = ds.map(_map_fn, num_parallel_calls=multiprocessing.cpu_count())
    ds = ds.batch(flags.batch_size)
    ds = ds.prefetch(buffer_size=20)
    return ds

im_train_A, im_train_B, im_test_A, im_test_B = tl.files.load_cyclegan_dataset(filename='horse2zebra', path='data') # horse2zebra apple2orange
print("num of A", len(im_train_A))
print("num of B", len(im_train_B))

n_step_per_epoch = min(len(im_train_A), len(im_train_B)) // flags.batch_size

data_A = get_data(im_train_A)
data_B = get_data(im_train_B)

# data_A_test = get_data(im_test_A)
# data_B_test = get_data(im_test_B)
