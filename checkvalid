from PIL import Image
import os

def check_images(s_dir, ext_list):
    bad_images=[]
    bad_ext=[]
    for fldr in os.listdir(s_dir):
        sub_folder=os.path.join(s_dir, fldr)
        if os.path.isdir(sub_folder):
            print('processing folder ', fldr)
            for file in os.listdir(sub_folder):
                f_path=os.path.join(sub_folder, file)
                test, ext = os.path.splitext(f_path)
                if ext.lower() not in ext_list:
                    print('file ', f_path, ' has an invalid extension ', ext)
                    bad_ext.append(f_path)
                if os.path.isfile(f_path):
                    try:
                        img=Image.open(f_path)
                        img.verify()  # verify that it is, in fact an image
                    except (IOError, SyntaxError) as e:
                        print('Bad file:', f_path)
                        bad_images.append(f_path)
    return bad_images, bad_ext

src_dir = "RGB ArSL dataset"
exten_list = ['.jpg', '.jpeg', '.png', '.bmp']  # list of acceptable extensions
bad_file_list, bad_file_ext= check_images(src_dir, exten_list)
print('bad files:', bad_file_list)
