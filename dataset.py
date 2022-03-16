# data aug and dataloader
import json
import numpy as np
import os

from transform import TransformFactory


class Dataset:
    def __init__(self, cfg):
        self.data_path = os.path.join(os.getcwd(), "data")
        self.cfg = cfg
        try:
            classes = [name for name in os.listdir(self.data_path) if not name.startswith('.')]
        except FileNotFoundError:
            print(f"{self.data_path} not found! Are you in the root directory?")
            raise

        classes = sorted(classes)
        self.idx2cls = {idx: cls for idx, cls in enumerate(classes)}
        self.cls2idx = {cls: idx for idx, cls in self.idx2cls.items()}
        self.transformations = []
        # for name, kwargs in self.cfg.DATASET.TRAIN_AUG:
        #     self.transformations.append(TransformFactory.get(name, **kwargs))
        self._load()

    def _load(self):
        self.dataset = []
        for cls in self.cls2idx.keys():
            class_path = os.path.join(self.data_path, cls)
            names = [name for name in os.listdir(os.path.join(class_path, "images")) if name.endswith(".png")]
            names = [name for name in names
                     if name.replace(".png", ".json") in os.listdir(os.path.join(class_path, "labels"))]
            image_paths = [os.path.join(class_path, "images", name) for name in names]
            annos = [json.load(open(os.path.join(class_path, "labels", name.replace(".png", ".json"))))
                     for name in names]
            annos = [anno["multi_hand_landmarks"] for anno in annos]

            def anno_to_ndarray(anno):
                rtn = np.zeros((len(anno), 3))
                for i in range(len(anno)):
                    rtn[i, 0] = anno[i]['x']
                    rtn[i, 1] = anno[i]['y']
                    rtn[i, 2] = anno[i]['z']
                return rtn

            annos = [anno_to_ndarray(anno) for anno in annos]
            self.dataset.extend([(img, anno, cls) for img, anno in zip(image_paths, annos)])

    def apply_transform(self, image, anno, cls):
        for tfm in self.transformations:
            image, anno, cls = tfm.apply(image, anno, cls)
        return image, anno, cls

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_image, anno, cls = self.dataset[idx]
        image, anno, cls = self.apply_transform(original_image, anno, cls)
        return {"original_image": original_image, "image": image, "anno": anno, "label": self.cls2idx[cls]}


if __name__ == "__main__":
    cfg = dict(
        DATASET=dict(
            TRAIN_AUG=[  # TODO: implement these data aug
                ("Resize", dict(w=224, h=224)),  # resize
                ("RandomCrop", dict(scale=[0.9, 1.5],)),  # random crop
                ("RandomFlip", dict(prob=0.5)),  # random flip, do we need to change the sequence of coords?
                ("CoordNoise", dict(std=0.3)),  # add Gaussian noise to the coords
            ],
            TEST_AUG=[],
            BATCH_SIZE=64,
        ),
    )
    dataset = Dataset(cfg)
    print(dataset[1])

