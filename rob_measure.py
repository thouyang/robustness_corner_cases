from deepfool import deepfool
import numpy as np
import pandas as pd

def rob_measure(model, TestD, path):
    #### robustness measurement
    rob_r = []
    label_tru = []
    label_pre = []
    n = 0
    for images, labels in TestD:

        for i in range(len(images)):
            r, _, label_orig, label_pert, perimage = deepfool(images[i], model, num_classes=10)
            # IMG=torch.cat((IMG,perimage.data.cpu()),dim=0)
            # temp=np.sqrt(np.sum(r**2))
            temp2 = np.linalg.norm(np.reshape(r, [1, -1]), ord=2, keepdims=True)
            rob_r.append(np.array(temp2[0, 0]))
            label_pre.append(label_pert)

    dataframe = pd.DataFrame({'rob':rob_r,'pred_y': label_pre})
    dataframe.to_csv(path, index=False, sep=',')
    mean_radius=np.mean(rob_r)

    return mean_radius




