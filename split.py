import os
import shutil
import random
import os.path


src_dir = r'C:\Users\user\Desktop\car_crush\091.차량 외관 영상 데이터\01.데이터\1.Training\brand\train\Sporatge20-21'
target_dir = r'C:\Users\user\Desktop\car_crush\091.차량 외관 영상 데이터\01.데이터\1.Training\brand\test\Sporatge20-21'
src_files = (os.listdir(src_dir))

def valid_path(dir_path, filename):
    full_path = os.path.join(dir_path, filename)
    return os.path.isfile(full_path)


files = [os.path.join(src_dir, f) for f in src_files if valid_path(src_dir, f)]
choices = random.sample(files, 20)
for files in choices:
    shutil.move(files, target_dir)
print ('Done')