# Expects vtab codebase and datasets to be installed and configured
# See https://github.com/google-research/task_adaptation

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image
from task_adaptation.data.caltech import Caltech101
from task_adaptation.data.cifar import CifarData
from task_adaptation.data.clevr import CLEVRData
from task_adaptation.data.diabetic_retinopathy import RetinopathyData
from task_adaptation.data.dmlab import DmlabData
from task_adaptation.data.dsprites import DSpritesData
from task_adaptation.data.dtd import DTDData
from task_adaptation.data.eurosat import EurosatData
from task_adaptation.data.kitti import KittiData
from task_adaptation.data.oxford_flowers102 import OxfordFlowers102Data
from task_adaptation.data.oxford_iiit_pet import OxfordIIITPetData
from task_adaptation.data.patch_camelyon import PatchCamelyonData
from task_adaptation.data.resisc45 import Resisc45Data
from task_adaptation.data.smallnorb import SmallNORBData
from task_adaptation.data.sun397 import Sun397Data
from task_adaptation.data.svhn import SvhnData
from tqdm import tqdm


# Convert TensorFlow dataset to a Hugging Face compatible dictionary format
def tfds_to_dict(dataset, split_name):
    print(f"Working on split: {split_name}")
    data = dataset.get_tf_data(split_name, 1, for_eval=True, epochs=1)
    data_dict = {
        "image": [],
        "label": [],
    }  # Assuming there are "image" and "label" keys
    for example in tqdm(data, total=dataset.get_num_samples(split_name)):
        # Extract the image (convert it from tensor to a proper format)
        image = example["image"][0].numpy()  # Ensure it's a NumPy array

        # Convert the image to the right format (you may need to reshape based on the dataset)
        # Hugging Face supports images as arrays directly
        data_dict["image"].append(image)

        # Handle labels (assuming they are strings or class indices)
        label = example["label"][0].numpy()
        data_dict["label"].append(label)

    return data_dict


def convert_and_upload_with_dict(dataset, num_classes, upload_name):
    features = Features(
        {
            "image": Image(),  # This tells Hugging Face that the "image" field contains image data
            "label": ClassLabel(
                num_classes=num_classes
            ),  # Replace with actual number of classes
        }
    )
    splits = dataset.splits
    # splits = ["train800", "val200", "test"]

    data_dicts = {}

    for split in splits:
        # Convert all splits
        converted = tfds_to_dict(dataset, split)

        # Convert each split dictionary to a Hugging Face dataset
        hf_dataset = Dataset.from_dict(converted, features=features)
        data_dicts[split] = hf_dataset

    # Create a DatasetDict to handle multiple splits (you can extend it to handle all splits)
    hf_dataset = DatasetDict(data_dicts)

    # Push the dataset to the Hugging Face Hub
    # hf_dataset.push_to_hub(upload_name, "test", private=True)
    hf_dataset.push_to_hub(upload_name, private=True)


def convert_and_upload_with_gen(dataset, num_classes, upload_name):
    features = Features(
        {
            "image": Image(),  # This tells Hugging Face that the "image" field contains image data
            "label": ClassLabel(
                num_classes=num_classes
            ),  # Replace with actual number of classes
        }
    )
    splits = dataset.splits
    # splits = ["train800", "val200", "test"]

    data_dicts = {}

    for split in splits:
        # Convert all splits in batches or streaming mode
        def gen():
            data = dataset.get_tf_data(
                split, batch_size=1, for_eval=True, epochs=1, drop_remainder=False
            )
            for batch in data:
                images = batch[
                    "image"
                ].numpy()  # Convert the batch of images to NumPy arrays
                labels = batch[
                    "label"
                ].numpy()  # Convert the batch of labels to NumPy arrays

                # Yield one image and label at a time
                for img, lbl in zip(images, labels):
                    yield {"image": img, "label": lbl}

        hf_dataset = Dataset.from_generator(gen, features=features)
        data_dicts[split] = hf_dataset

    # Create a DatasetDict to handle multiple splits (you can extend it to handle all splits)
    hf_dataset = DatasetDict(data_dicts)

    # Push the dataset to the Hugging Face Hub
    hf_dataset.push_to_hub(upload_name, private=True)


# Caltech101  CIFAR-100  DTD  Flowers102  Pets  Sun397  SVHN
convert_and_upload_with_gen(
    Caltech101(), num_classes=102, upload_name="YOUR_HF_USERNAME/vtab_caltech101"
)

convert_and_upload_with_gen(
    CifarData(num_classes=100),
    num_classes=100,
    upload_name="YOUR_HF_USERNAME/vtab_cifar100",
)

convert_and_upload_with_gen(
    CifarData(num_classes=10),
    num_classes=10,
    upload_name="YOUR_HF_USERNAME/vtab_cifar10",
)

convert_and_upload_with_gen(
    DTDData(), num_classes=47, upload_name="YOUR_HF_USERNAME/vtab_dtd"
)

convert_and_upload_with_gen(
    OxfordFlowers102Data(),
    num_classes=102,
    upload_name="YOUR_HF_USERNAME/vtab_flowers102",
)

convert_and_upload_with_gen(
    OxfordIIITPetData(), num_classes=37, upload_name="YOUR_HF_USERNAME/vtab_pets"
)

convert_and_upload_with_gen(
    Sun397Data(), num_classes=397, upload_name="YOUR_HF_USERNAME/vtab_sun397"
)

convert_and_upload_with_gen(
    SvhnData(), num_classes=10, upload_name="YOUR_HF_USERNAME/vtab_svhn"
)


# Camelyon  EuroSAT  Resisc45  Retinopathy
convert_and_upload_with_gen(
    PatchCamelyonData(),
    num_classes=2,
    upload_name="YOUR_HF_USERNAME/vtab_patch_camelyon",
)

convert_and_upload_with_gen(
    EurosatData(), num_classes=10, upload_name="YOUR_HF_USERNAME/vtab_eurosat"
)

convert_and_upload_with_gen(
    Resisc45Data(), num_classes=45, upload_name="YOUR_HF_USERNAME/vtab_resisc45"
)

convert_and_upload_with_gen(
    RetinopathyData(), num_classes=5, upload_name="YOUR_HF_USERNAME/vtab_retinopathy"
)


# Clevr-Count  Clevr-Dist  DMLab  dSpr-Loc  dSpr-Ori  KITTI-Dist  sNORB-Azim  sNORB-Elev
convert_and_upload_with_gen(
    CLEVRData(task="count_all"),
    num_classes=8,
    upload_name="YOUR_HF_USERNAME/vtab_clevr_count",
)


convert_and_upload_with_gen(
    CLEVRData(task="closest_object_distance"),
    num_classes=6,
    upload_name="YOUR_HF_USERNAME/vtab_clevr_distance",
)


convert_and_upload_with_gen(
    DmlabData(), num_classes=6, upload_name="YOUR_HF_USERNAME/vtab_dmlab"
)


convert_and_upload_with_dict(
    DSpritesData("label_x_position", num_classes=16),
    num_classes=16,
    upload_name="YOUR_HF_USERNAME/vtab_dsprites_location",
)


convert_and_upload_with_dict(
    DSpritesData("label_orientation", num_classes=16),
    num_classes=16,
    upload_name="YOUR_HF_USERNAME/vtab_dsprites_orientation",
)


convert_and_upload_with_gen(
    KittiData(task="closest_vehicle_distance"),
    num_classes=4,
    upload_name="YOUR_HF_USERNAME/vtab_kitti_distance",
)


convert_and_upload_with_gen(
    SmallNORBData("label_azimuth"),
    num_classes=18,
    upload_name="YOUR_HF_USERNAME/vtab_smallnorb_azimuth",
)


convert_and_upload_with_gen(
    SmallNORBData("label_elevation"),
    num_classes=9,
    upload_name="YOUR_HF_USERNAME/vtab_smallnorb_elevation",
)
