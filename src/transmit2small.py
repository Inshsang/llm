import pickle
import numpy as np
# with open("/media/kou/Data3/htc/dataset/Object/my_train_8192pts_fps.dat", 'rb') as f:
#     list_of_objpoints = pickle.load(f)
#
# for index,(point,label) in enumerate(zip(list_of_objpoints[0],list_of_objpoints[1])):
#     # new0 = point
#     # new1 = label
#     np.save("/media/kou/Data3/htc/Objects_8192_npy/points/"+str(index)+".npy",point)
#     np.save("/media/kou/Data3/htc/Objects_8192_npy/labels/" + str(index) + ".npy", label)
#     print(index)
#

with open("/media/kou/Data3/htc/dataset/cut_scene_500.dat", 'rb') as f:
    list_of_objpoints = pickle.load(f)
# with open("/media/kou/Data3/htc/dataset/cut_scene_train.dat", 'rb') as f:
#     list_of_objpoints = pickle.load(f)

for index,point in enumerate(list_of_objpoints):
    # new0 = point
    # new1 = label
    # np.save("/media/kou/Data3/htc/Detection_obj/" + str(index) + ".npy", point)
    np.save("/media/kou/Data3/htc/Detection_obj/"+str(index)+"_500.npy",point)
    print(index)

