import os
from anomalib.engine import Engine
from anomalib.models import Patchcore

def main():

    BASE_CKPT_DIR = r"E:\dai hoc\project\results\Patchcore\MVTecAD"
    
    BASE_OUTPUT_DIR = r"E:\dai hoc\project\exported_models"

    product_list = [d for d in os.listdir(BASE_CKPT_DIR)]

    for product_name in product_list:

        ckpt_path = os.path.join(BASE_CKPT_DIR, product_name, "v0", "weights", "lightning", "model.ckpt")
        
        print(f"Đang xử lý sản phẩm: {product_name.upper()}")
        
        product_output_dir = os.path.join(BASE_OUTPUT_DIR, product_name)

        model = Patchcore(backbone="wide_resnet50_2", coreset_sampling_ratio=0.1)
        engine = Engine()

        print(f"Tiến hành xuất file .pt vào: {product_output_dir}")
        engine.export(
            model=model,
            export_type="torch",
            ckpt_path=ckpt_path,
            export_root=product_output_dir
        )
        print(f"Hoàn tất xuất mô hình cho {product_name}!")

if __name__ == "__main__":
    main()