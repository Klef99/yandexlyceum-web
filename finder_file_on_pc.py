import os
import re
import win32api


def find_file(root_folder, rex):
    way = ''
    for root, dirs, files in os.walk(root_folder):
        for f in files:
            result = rex.search(f)
            if result:
                way = os.path.join(root, f)
                break
    return way


def find_file_in_all_drives(file_name):
    rex = re.compile(file_name)
    for drive in win32api.GetLogicalDriveStrings().split('\000')[:-1]:
        way_res = find_file(drive, rex)
        return way_res


#print(find_file_in_all_drives('Canada.jpg'))