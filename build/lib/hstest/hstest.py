import logging

import tensorflow as tf

from brevettiai.data.image import ImagePipeline, ImageAugmenter
from brevettiai.interfaces import vue_schema_utils as vue
from brevettiai.interfaces.remote_monitor import RemoteMonitor
from brevettiai.platform import Job, get_image_samples
from brevettiai.data.sample_integrity import SampleSplit
from brevettiai.data.data_generator import DataGenerator, OneHotEncoder
from . import data

log = logging.getLogger(__name__)


class Hstest(Job):
    class Settings(vue.VueSettingsModule):
        def __init__(self, image: ImagePipeline, augmenter: ImageAugmenter,
                     classes: list = None, class_mapping: dict = None,
                     epochs=30, batch_size=8, augmenter_enable=True):
            self.image = image
            self.augmenter = augmenter
            self.classes = classes or []
            self.class_mapping = class_mapping or {}
            self.epochs = epochs
            self.batch_size = batch_size
            self.augmenter_enable = augmenter_enable

        @classmethod
        def get_schema(cls, namespace=None):
            builder = super().get_schema(namespace)
            # This is for user - other settings are set to default value
            builder.filter_fields([
                vue.label("Image Pipeline"),
                "image.input_width",
                "image.input_height",
                # "image.rois",
                "image.color_mode",

                vue.label("Image Augmenter"),
                "augmenter_enable",
                "augmenter.random_transformer.flip_up_down",
                "augmenter.random_transformer.flip_left_right",
                "augmenter.random_transformer.rotate",
                "augmenter.random_transformer.translate_horizontal",
                "augmenter.random_transformer.translate_vertical",

                vue.label("Training"),
                "classes",
                "class_mapping",
                "epochs",
                "batch_size",
            ])
            return builder

    settings: Settings

    def train(self):
        settings = self.settings

        # pd dataframe
        full_set = get_image_samples(self.datasets, category_map=settings.class_mapping,
                                     force_categories=settings.classes or False)
        full_set = SampleSplit().assign(full_set, id_path=self.artifact_path("sample_identification.csv"))

        # divide samples into train/test set
        train_set = full_set.loc[full_set.purpose == 'train']
        test_set = full_set.loc[full_set.purpose == 'devel']

        # Data
        if not settings.classes:
            class_space = set(full_set.category.unique())
            classes = set(item for sublist in class_space for item in sublist if item != "__UNLABELED__")
            classes = list(sorted(classes))
        else:
            classes = settings.classes

        if settings.augmenter_enable:
            train_ds = DataGenerator(train_set, output_structure=("img", "onehot"), batch_size=settings.batch_size,
                                     shuffle=True, repeat=True) \
                .map(settings.image) \
                .map(settings.augmenter) \
                .map(OneHotEncoder(classes=classes))

            test_ds = DataGenerator(test_set, output_structure=("img", "onehot"), batch_size=settings.batch_size,
                                    shuffle=True, repeat=True) \
                .map(settings.image) \
                .map(settings.augmenter) \
                .map(OneHotEncoder(classes=classes))
        else:
            train_ds = DataGenerator(train_set, output_structure=("img", "onehot"), batch_size=settings.batch_size,
                                     shuffle=True, repeat=True) \
                .map(settings.image) \
                .map(OneHotEncoder(classes=classes))

            test_ds = DataGenerator(test_set, output_structure=("img", "onehot"), batch_size=settings.batch_size,
                                    shuffle=True, repeat=True) \
                .map(settings.image) \
                .map(settings.augmenter) \
                .map(OneHotEncoder(classes=classes))

        # define the model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(len(classes), activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-7),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        tensorboard = self.temp_path("tensorboard", self.run_id, dir=True)
        model.fit(train_ds.get_dataset(),
                  epochs=settings.epochs,
                  steps_per_epoch=len(train_ds),
                  validation_data=test_ds.get_dataset().take(16),
                  callbacks=[RemoteMonitor(root=self.host_name, path=self.api_endpoints["remote"]),
                             tf.keras.callbacks.TensorBoard(log_dir=tensorboard, profile_batch=0)])

        # test accuracy
        loss, acc = model.evaluate(test_ds.get_dataset(), steps=len(test_set) // settings.batch_size,
                                   verbose=1)
        log.info('Accuracy:' + str(acc))

        # Export
        saved_model_dir = self.temp_path("export", "saved_model", dir=True)

        model.save(saved_model_dir, overwrite=True, include_optimizer=False,
                   signatures=data.get_serving_receiver(model, settings.image.target_size[:2],
                                                        **settings.image.get_rescaling()))

        output_path = data.package_saved_model(saved_model_dir)

        return output_path