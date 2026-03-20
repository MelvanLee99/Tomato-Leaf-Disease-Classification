import os
import tensorflow_datasets as tfds

class BaseTomatoLeafDiseaseDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    DATA_DIR = None

    def _info(self):
        if self.DATA_DIR is None:
            raise ValueError("'DATA_DIR' must be set in a subclass.")
        
        return tfds.core.DatasetInfo(
            builder=self,
            description="Custom Tomato Leaf Disease Dataset from PlantVillage",
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(None, None, 3)),
                "label": tfds.features.ClassLabel(names=self._get_label_names()),
            }),
            supervised_keys=("image", "label"),
        )

    def _split_generators(self, dl_manager):
        if self.DATA_DIR is None:
            raise ValueError("'DATA_DIR' must be set in a subclass.")
        
        return {
            'train': self._generate_examples(self.DATA_DIR),
        }

    def _generate_examples(self, path):
        for label in os.listdir(path):
            class_dir = os.path.join(path, label)
            if not os.path.isdir(class_dir): continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('jpg', 'jpeg', 'png')):
                    yield f"{label}_{img_name}", {
                        "image": os.path.join(class_dir, img_name),
                        "label": label,
                    }

    def _get_label_names(self):
        if self.DATA_DIR is None:
            raise ValueError("'DATA_DIR' must be set in a subclass.")
        return sorted([d for d in os.listdir(self.DATA_DIR) if os.path.isdir(os.path.join(self.DATA_DIR, d))])
    
class BinaryTomatoLeafDiseaseDataset(BaseTomatoLeafDiseaseDataset):
    DATA_DIR = './dataset/tomato_leaf_disease_binary/'

class QuinaryTomatoLeafDiseaseDataset(BaseTomatoLeafDiseaseDataset):
    DATA_DIR = './dataset/tomato_leaf_disease_quinary/'

class MainTomatoLeafDiseaseDataset(BaseTomatoLeafDiseaseDataset):
    DATA_DIR = './dataset/tomato_leaf_disease_main/'