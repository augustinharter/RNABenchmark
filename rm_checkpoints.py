import os

def get_sub_folders(path, recurse=0):
    if recurse == 0:
        return [os.path.join(path, f.name) for f in os.scandir(path) if f.is_dir()]
    else:
        sub_folders = []
        for f in os.scandir(path):
            if f.is_dir():
                sub_folders.append(os.path.join(path, f.name))
                sub_folders.extend(get_sub_folders(os.path.join(path, f.name), recurse - 1))
        return sub_folders
    
all_folders = get_sub_folders('experiments', recurse=-1)
for folder in all_folders:
    if folder.split(os.path.sep)[-1].startswith('checkpoint'):
        print('deleting folder: ', folder)
        # rmdir even if not empty, since checkpoints are large and we want to free up space
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
        os.rmdir(folder)