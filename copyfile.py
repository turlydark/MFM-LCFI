import os
import shutil

source_path = os.path.abspath(r'F:\forgery_images_c40\NT2\test\0')
target_path = os.path.abspath(r'F:\forgery_images_c40\NT2\test\0')

if not os.path.exists(target_path):
    os.makedirs(target_path)

if os.path.exists(source_path):
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
    for root, dirs, files in os.walk(source_path):
        for file in files:
            src_file = os.path.join(root, file)

            print(src_file)
            tmp = src_file.split('.')
            # print(tmp)
            target_path = tmp[0] + '_nt.' + tmp[1]
            # print(target_path)
            # exit(0)

            shutil.copy(src_file, target_path)
            print(src_file)

print('copy files finished!')