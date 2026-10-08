import os
import shutil

# 복사할 기존 파일
o_path = "./keras/"
o_keras_num = "53"
o_keyword = "Reduce"

# 새로 생성할 파일
path = "./keras2/"
keras_num = "70"
keyword = "Conv1D_"

names = [
    "01_california",
    "02_diabetes",
    "03_boston",
    "04_dacon_ddarung",
    "05_kaggle_bike",
    "06_cancer",
    "07_santander",
    "08_wine",
    "09_fetch_covtype",
    "10_digits",
    "11_mnist",
    "12_fashion",
    "13_cifar10",
    "14_cifar100",
    "15_man_woman",
    "16_jena",
]

for name in names:
    # 복사할 기존 파일
    template = f"{o_path}keras{o_keras_num}_{o_keyword}{name}.py"

    # 새 파일
    filename = f"keras{keras_num}_{keyword}{name}.py"
    new_file = f"{path}{filename}"

    # 기존 파일이 없으면 건너뜀
    if not os.path.exists(template):
        print(f"{template} 없음 → 건너뜀")
        continue

    # 새 파일이 이미 있으면 건너뜀
    if os.path.exists(new_file):
        print(f"{filename} 이미 존재 → 건너뜀")
        continue

    shutil.copy(template, new_file)

    print(f"{filename} 생성 및 코드 복사")