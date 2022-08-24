import imageio.v2 as imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms

from utils.flow_utils import flow_to_image, resize_flow
from utils.torch_utils import restore_model
from models.pwclite import PWCLite

import glob
from pathlib import Path
import os
from tqdm import tqdm
import cv2
from utils.utils import InputPadder


class TestHelper():
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            # sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def init_model(self):
        model = PWCLite(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    def run(self, imgs):
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        padder = InputPadder(imgs[0].shape, divis_by=64)
        img1, img2 = padder.pad(imgs[0], imgs[1])

        img_pair = torch.cat([img1, img2], 1).to(self.device)
        # img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair), padder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='checkpoints/KITTI15/pwclite_raw.tar')
    parser.add_argument('-s', '--test_shape', default=[384, 640], type=int, nargs=2)
    parser.add_argument('-i', '--img_list', nargs='+',
                        default=['examples/img1.png', 'examples/img2.png'])
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
    #                     default="/media/insta360/新加卷/stereo_data/kitti15/training/image_2/*_10.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
    #                     default="/media/insta360/新加卷/stereo_data/kitti15/training/image_3/*_10.png")
    parser.add_argument('-gt', '--gt_imgs', help="path to all gt frames",
                        default="/media/insta360/新加卷/stereo_data/kitti15/training/disp_occ_0/*_10.png")
    parser.add_argument('--output_directory', help="directory to save output",
                        default="test/FlyingThings3D/result_pwclite_raw")
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
                        default="test/FlyingThings3D/frames_finalpass/TEST/left/*.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
                        default="test/FlyingThings3D/frames_finalpass/TEST/right/*.png")
    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': len(args.img_list),
            'reduce_dense': True
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    ts = TestHelper(cfg)
    left_images = sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))
    gt_images = sorted(glob.glob(args.gt_imgs, recursive=True))

    out_list_d1, out_list_d2, out_list_d3, epe_list = [], [], [], []

    for (imfile1, imfile2, gtimg1) in tqdm(list(zip(left_images, right_images, gt_images))):
        imgs = [imageio.imread(img).astype(np.float32) for img in [imfile1, imfile2]]
        img1 = imageio.imread(imfile1).astype(np.float32)
        h, w = imgs[0].shape[:2]
        gtimg1 = cv2.imread(gtimg1, cv2.IMREAD_ANYDEPTH) / 256.0
        valid_mask = (gtimg1 > 0.0)

        result, padder = ts.run(imgs)
        flow_12 = result['flows_fw'][0]

        # flow_12 = resize_flow(flow_12, (h, w))
        np_flow_12 = padder.unpad(flow_12)[0].detach().cpu().numpy().transpose([1, 2, 0])

        # vis_flow = flow_to_image(np_flow_12)
        vis_flow = -1 * np_flow_12[:, :, 0]


        # fig = plt.figure()
        # plt.imshow(vis_flow)
        # plt.show()
        #
        file_stem = imfile1.split('/')[-1]
        plt.imsave(output_directory / f"{os.path.splitext(file_stem)[0]}.png",
                   vis_flow, cmap='jet')

    #     mask_gt = gtimg1[valid_mask]
    #     mask_pre = vis_flow[valid_mask]
    #
    #     disp_error = np.absolute(np.subtract(mask_gt, mask_pre))
    #
    #     epe = np.average(disp_error)
    #
    #     bad1_pixels = np.count_nonzero(disp_error > 1)
    #     bad2_pixels = np.count_nonzero(disp_error > 2)
    #     bad3_pixels = np.count_nonzero(disp_error > 3)
    #
    #     d1 = 100 * (bad1_pixels / mask_pre.size)
    #     d2 = 100 * (bad2_pixels / mask_pre.size)
    #     d3 = 100 * (bad3_pixels / mask_pre.size)
    #
    #     epe_list.append(epe)
    #     out_list_d1.append(d1)
    #     out_list_d2.append(d2)
    #     out_list_d3.append(d3)
    #
    # epe_list = np.array(epe_list)
    # out_list_d1 = np.array(out_list_d1)
    # out_list_d2 = np.array(out_list_d2)
    # out_list_d3 = np.array(out_list_d3)
    #
    # epe = np.mean(epe_list)
    # d1 = np.mean(out_list_d1)
    # d2 = np.mean(out_list_d2)
    # d3 = np.mean(out_list_d3)
    # print(f"Validation KITTI: EPE {epe}, D1 {d1}, D2 {d2}, D3 {d3} ")


