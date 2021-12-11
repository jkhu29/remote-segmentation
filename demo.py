import torch
import config
import cv2
import os
import time
import numpy as np
from models import DANet

opt = config.get_options()

CUDA_ENABLE = torch.cuda.is_available()
output_dir = "./data/results"


def semantic_to_mask(mask, labels: np.ndarray):
    x = np.argmax(mask, axis=0)
    x = labels[x.astype(np.uint8)]
    return x


@torch.no_grad()
def Demo():
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    device = torch.device('cuda:0' if CUDA_ENABLE and opt.cuda else 'cpu')
    model = DANet(n_classes=9).to(device)
    model.load_state_dict(torch.load('danet_epoch_29.pth'))
    model.eval()

    for img_name in os.listdir("test_images/images"):
        t0 = time.time()
        img_path = os.path.join("test_images/images", img_name)
        inputs = torch.FloatTensor(cv2.imread(img_path).transpose(2, 0, 1)).unsqueeze(0).cuda()

        out = model(inputs)[0].squeeze()
        # print(out.shape)
        pred = torch.softmax(out, dim=0).cpu().numpy()
        pred = semantic_to_mask(pred, labels=labels).squeeze().astype(np.uint8)
        t1 = time.time()
        # print(pred.shape)
        cv2.imwrite(os.path.join(output_dir, img_name.replace("tif", "png")), pred)
        print(img_path + " OK! cost: {}s".format(t1 - t0))
        # cv2.imshow("label_pred", pred*25)
        # cv2.waitKey()
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    Demo()
