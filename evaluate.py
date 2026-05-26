import os
import pandas as pd
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

def main():

    DATA_DIR = r'E:\dai hoc\project\archive'
    
    product_list = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    BASE_CKPT_DIR = r"E:\dai hoc\project\results\Patchcore\MVTecAD" 

    all_results = []
    
    for product_name in product_list:

        ckpt_path = os.path.join(BASE_CKPT_DIR, product_name, "v0", "weights", "lightning", "model.ckpt")
        
        if not os.path.exists(ckpt_path):
            print(f"Bỏ qua {product_name}: Không tìm thấy file {ckpt_path}")
            continue

        print(f"Đang nạp mô hình và lấy điểm cho: {product_name}")
        
        datamodule = MVTecAD(
            root=DATA_DIR,
            category=product_name,
            train_batch_size=16,
            eval_batch_size=16,
            num_workers=0,
        )

        model = Patchcore(
            backbone="wide_resnet50_2", 
            coreset_sampling_ratio=0.1
        )

        engine = Engine(
            enable_progress_bar=False, 
            enable_model_summary=False
        )

        test_results = engine.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path 
        )

        if test_results:
            metrics = test_results[0]
            metrics['Sản phẩm'] = product_name 
            all_results.append(metrics)
            print(f"Đã trích xuất xong bảng điểm của {product_name}")

if __name__ == "__main__":
    main()