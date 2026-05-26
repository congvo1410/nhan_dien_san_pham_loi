import os

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

DATA_DIR = r'E:\dai hoc\project\archive'

product_list = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

for product_name in product_list:
    print(f"ĐANG HUẤN LUYỆN SẢN PHẨM: {product_name}")

    product_path = os.path.join(DATA_DIR, product_name)

    datamodule = MVTecAD(
        root = DATA_DIR,
        category = product_name,
        train_batch_size = 16,
        eval_batch_size = 16,
        num_workers= 0,
    )

    model = Patchcore(
        backbone="wide_resnet50_2", 
        coreset_sampling_ratio=0.1
    )

    engine = Engine(
        max_epochs = 1,
    )

    engine.fit(datamodule=datamodule, model=model)
    engine.test(datamodule=datamodule, model=model)

    engine.export(
        model=model
    )
    
    print(f"ĐÃ HOÀN THÀNH HUẤN LUYỆN SẢN PHẨM: {product_name}")