import utils
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataset import CloudDataset
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
from torch.utils.data import DataLoader
import importlib
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from models import CloudModel


DATA_DIR = Path("/codeexecution")
TRAIN_FEATURES = DATA_DIR / "train_features"
TRAIN_LABELS = DATA_DIR / "train_labels"
BANDS = ["B02", "B03", "B04"]



transform = albu.Compose([
        albu.Resize(512, 512),
        albu.CLAHE(),
        albu.Normalize(),
        ToTensor()
    ])


if __name__ == "__main__":
    
    ### Load config
    config = utils.load_yaml("config.yaml")
    utils.seed_everything(config['SEED'])

    ### Prepare data
    data = utils.makeTrainCSV(DATA_DIR)
    train_data, val_data = train_test_split(data, test_size=0.25, stratify = data['segmentedArea'])
    print("Train size", len(train_data))
    print("Validation size", len(val_data))
    train_dataset = CloudDataset(data=train_data, bands=BANDS, label_exist=True, transforms=transform)
    val_dataset = CloudDataset(data=val_data, bands=BANDS, label_exist=True, transforms=transform)

    print(train_data.head())

    train_loader = DataLoader(train_dataset,
                                  batch_size=config["BATCH_SIZE"],
                                  shuffle=True,
                                  num_workers=config["NUM_WORKERS"])

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config["BATCH_SIZE"],
                                shuffle=False,
                                num_workers=config["NUM_WORKERS"])
    
    dataloaders = {"train": train_loader, "val": val_dataloader}

    ### Import model
    module = importlib.import_module(config["MODEL"]["PY"])
    model = getattr(module, config["MODEL"]["ARCH"])(
                    **config["MODEL"]["ARGS"])

    ### Import other things
    module = importlib.import_module(config["OPTIMIZER"]["PY"])
    optimizer = getattr(module, config["OPTIMIZER"]["CLASS"])(
        model.parameters(), **config["OPTIMIZER"]["ARGS"])

    module = importlib.import_module(config["SCHEDULER"]["PY"])
    scheduler = getattr(module, config["SCHEDULER"]["CLASS"])(
        optimizer, **config["SCHEDULER"]["ARGS"])
    callbacks = []
    if config["EARLY_STOPPING"]["ENABLE"]:
        early_stop_callback = EarlyStopping(
            **config["EARLY_STOPPING"]["ARGS"]
        )
        callbacks.append(early_stop_callback)

    checkpoint_callback = ModelCheckpoint(
        **config["CHECKPOINT"]["ARGS"]
    )
    callbacks.append(checkpoint_callback)

    logger = TensorBoardLogger(
            "logs/"+config["EXPERIMENT_NAME"], name=config["EXPERIMENT_NAME"])
    
    lightning_model = CloudModel(
        dataloaders,
        model = model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )

    grad_clip = config["GRADIENT_CLIPPING"]
    grad_acum = config["GRADIENT_ACCUMULATION_STEPS"]
    trainer = pl.Trainer(gpus=config["GPUS"],
                            max_epochs=config["EPOCHS"],
                            num_sanity_val_steps=0,
                        #  limit_train_batches=0.001,
                        #  limit_val_batches=0.2,
                            logger=logger,
                            gradient_clip_val=grad_clip,
                            accumulate_grad_batches=grad_acum,
                            precision=16,
                            callbacks=callbacks)
    trainer.fit(lightning_model)


