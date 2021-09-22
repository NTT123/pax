import os

import tensorflow as tf
import tensorflow_datasets as tfds

### load celeb_a dataset

# This is a hack to use a custom link to celeb-a dataset in tensorflow-datasets.
# replace the ``tfds.image.CelebA._split_generators`` method by the following method
# which uses our custom links.

IMG_ALIGNED_DATA = (
    "https://drive.google.com/uc?export=download&"
    "id=1iQRFaGXRiPBd-flIm0u-u8Jy6CfJ_q6j"
)

EVAL_LIST = (
    "https://drive.google.com/uc?export=download&"
    "id=1ab9MDLOblszbKKXoDe8jumFsSkn6lIX1"
)
# Landmark coordinates: left_eye, right_eye etc.
LANDMARKS_DATA = (
    "https://drive.google.com/uc?export=download&"
    "id=1y8qfK-jaq1QWl9v_n_mBNIMu5-h3UXK4"
)

# Attributes in the image (Eyeglasses, Mustache etc).
ATTR_DATA = (
    "https://drive.google.com/uc?export=download&"
    "id=1BPfcVuIqrAsJAgG40-XGWU7g2wmmQU30"
)


def _split_generators(self, dl_manager):
    downloaded_dirs = dl_manager.download(
        {
            "img_align_celeba": IMG_ALIGNED_DATA,
            "list_eval_partition": EVAL_LIST,
            "list_attr_celeba": ATTR_DATA,
            "landmarks_celeba": LANDMARKS_DATA,
        }
    )

    # Load all images in memory (~1 GiB)
    # Use split to convert: `img_align_celeba/000005.jpg` -> `000005.jpg`
    all_images = {
        os.path.split(k)[-1]: img
        for k, img in dl_manager.iter_archive(downloaded_dirs["img_align_celeba"])
    }

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "file_id": 0,
                "downloaded_dirs": downloaded_dirs,
                "downloaded_images": all_images,
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "file_id": 1,
                "downloaded_dirs": downloaded_dirs,
                "downloaded_images": all_images,
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                "file_id": 2,
                "downloaded_dirs": downloaded_dirs,
                "downloaded_images": all_images,
            },
        ),
    ]


img_mean = 0.5
img_scale = 0.5
image_size = 64  # size of input image: 64x64

tfds.image.CelebA._split_generators = _split_generators


def load_celeb_a():
    ds = tfds.load("celeb_a")

    def img_ops(x):
        img = tf.cast(x["image"], tf.float32) / 255.0
        img = tf.image.resize(
            img, (image_size * 2, image_size), preserve_aspect_ratio=True
        )
        img = tf.image.crop_to_bounding_box(img, 7, 0, 64, 64)
        img = (img - img_mean) / img_scale
        return img

    dataset = (
        ds["train"].concatenate(ds["validation"]).concatenate(ds["test"]).map(img_ops)
    )
    return dataset
