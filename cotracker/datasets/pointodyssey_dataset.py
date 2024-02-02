import os
import torch

import imageio
import numpy as np
import pdb

from cotracker.datasets.utils import CoTrackerData
from torchvision.transforms import ColorJitter, GaussianBlur
from PIL import Image

import matplotlib.pyplot as plt

from cotracker.datasets.kubric_movif_dataset import CoTrackerDataset, save_img_with_tracks, save_track_vid

import matplotlib.pyplot as plt

class PointOdysseyDataset(CoTrackerDataset):
    def __init__(
        self,
        data_root,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        sample_vis_1st_frame=False,
        use_augs=False,
    ):
        super(PointOdysseyDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            sample_vis_1st_frame=sample_vis_1st_frame,
            use_augs=use_augs,
        )

        self.pad_bounds = [0, 25]
        # self.resize_lim = [0.75, 1.25]  # sample resizes from here
        self.resize_lim = [0.5, .75]  # sample resizes from here
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        self.strides=[3, 4]
        self.seq_names = [
            fname
            for fname in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, fname))
        ]
        print("found %d unique videos in %s" % (len(self.seq_names), self.data_root))

    def getitem_helper(self, index):
        gotit = True
        seq_name = self.seq_names[index]

        self.curr_seq_name = seq_name
        self.curr_index = index
        self.save_images = False

        if self.save_images:
            frame = 0
            # interm_save_dir = './assets/intermediate_ims'
            interm_save_dir = '/data/ilona/cotracker_on_point_odyssey'
            save_path = os.path.join(interm_save_dir, 'intermediate_ims', f'{seq_name}_rbs_{frame}')

        npy_path = os.path.join(self.data_root, seq_name, "annot.npz")
        rgb_path = os.path.join(self.data_root, seq_name, "rgbs")

        img_paths = sorted(os.listdir(rgb_path))
        rgbs = []
        for i, img_path in enumerate(img_paths):
            image = imageio.v2.imread(os.path.join(rgb_path, img_path))
            rgbs.append(image)
        pdb.set_trace()
        

        rgbs = np.stack(rgbs) # [num_frames, height, width, 3]
        annotations = np.load(npy_path, allow_pickle=True)
        trajs = annotations['trajs_2d'].astype(np.float32)
        visibs = annotations['visibs'].astype(np.float32)
        # TODO - verify what visibs flag is doing (how many categories etc.)
        valids = (visibs<2).astype(np.float32) # S,N
        visibs = (visibs==1).astype(np.float32) # S,N

        if self.seq_len < len(rgbs):
            start_ind = np.random.choice(len(rgbs) - self.seq_len, 1)[0]
            seq_stride = np.random.choice(self.strides)

        # ensure that the point is good at start_ind
        vis_and_val = valids * visibs
        vis0 = vis_and_val[start_ind] > 0
        trajs = trajs[:,vis0]
        visibs = visibs[:,vis0]
        valids = valids[:,vis0]

        S,N,D = trajs.shape

        # get rid of infs and nans
        valids_xy = np.ones_like(trajs)
        inf_idx = np.where(np.isinf(trajs))
        trajs[inf_idx] = 0
        valids_xy[inf_idx] = 0
        nan_idx = np.where(np.isnan(trajs))
        trajs[nan_idx] = 0
        valids_xy[nan_idx] = 0
        inv_idx = np.where(np.sum(valids_xy, axis=2)<2) # S,N
        visibs[inv_idx] = 0
        valids[inv_idx] = 0

        # ensure that the point is good in frame0
        # TODO: might want to do this for the crop? 
        vis_and_val = valids * visibs
        vis0 = vis_and_val[start_ind] > 0
        trajs = trajs[:,vis0]
        visibs = visibs[:,vis0]
        valids = valids[:,vis0]

        # ensure that the point is good in frame1
        vis_and_val = valids * visibs
        vis1 = vis_and_val[start_ind+1] > 0
        trajs = trajs[:,vis1]
        visibs = visibs[:,vis1]
        valids = valids[:,vis1]

        H,W,C = rgbs[start_ind].shape
        trajs = np.minimum(np.maximum(trajs, np.array([-64,-64])), np.array([W+64, H+64])) # S,N,2
        
        # rename variables to cotracker dataloader names
        traj_2d = trajs
        visibility = visibs

        # reshape to same dimensions as kubric:
        visibility = np.transpose(visibility, (1, 0)) # [num_points, num_frames]
        traj_2d = np.transpose(traj_2d, (1, 0, 2)) # [num_points, num_frames, 2]

        # random crop
        assert self.seq_len <= len(rgbs)
        if self.seq_len < len(rgbs):
            # pdb.set_trace()
            rgbs = rgbs[start_ind : start_ind + self.seq_len*seq_stride:seq_stride] # crop sequence to certain length
            traj_2d= traj_2d[:, start_ind : start_ind + self.seq_len*seq_stride:seq_stride]
            visibility = visibility[:, start_ind : start_ind + self.seq_len*seq_stride:seq_stride]

        traj_2d = np.transpose(traj_2d, (1, 0, 2))
        visibility = np.transpose(visibility, (1, 0))
        # visibility = np.transpose(np.logical_not(visibility), (1, 0))

        if self.save_images:
            save_img_with_tracks(rgbs[frame]/255.0, traj_2d[frame], visibility[frame], save_path+"_01_seq_crop.png")
        if self.use_augs:
            rgbs, traj_2d, visibility = self.add_photometric_augs(rgbs, traj_2d, visibility)
            if self.save_images:
                save_img_with_tracks(rgbs[frame]/255.0, traj_2d[frame], visibility[frame], save_path+"_02_photometric_augs.png")
            rgbs, traj_2d = self.add_spatial_augs(rgbs, traj_2d, visibility)
            if self.save_images:
                save_img_with_tracks(rgbs[frame]/255.0, traj_2d[frame], visibility[frame], save_path+"_03_spatial_augs.png")
        else:
            rgbs, traj_2d = self.crop(rgbs, traj_2d)

        # update visibilities after augmentations
        visibility[traj_2d[:, :, 0] > self.crop_size[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > self.crop_size[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False

        # save images if you want
        if self.save_images:
            save_img_with_tracks(rgbs[0]/255.0, traj_2d[0], visibility[0], save_path+"_07_visibility_tracks.png")

        visibility = torch.from_numpy(visibility)
        traj_2d = torch.from_numpy(traj_2d)

        visibile_pts_first_frame_inds = (visibility[0]).nonzero(as_tuple=False)[:, 0]

        if self.sample_vis_1st_frame:
            visibile_pts_inds = visibile_pts_first_frame_inds
        else:
            visibile_pts_mid_frame_inds = (visibility[self.seq_len // 2]).nonzero(as_tuple=False)[
                :, 0
            ]
            visibile_pts_inds = torch.cat(
                (visibile_pts_first_frame_inds, visibile_pts_mid_frame_inds), dim=0
            )
        point_inds = torch.randperm(len(visibile_pts_inds))[: self.traj_per_sample]
        if len(point_inds) < self.traj_per_sample:
            # if false then they pad the points
            # print("len(point_inds) < self.traj_per_sample")
            # print(f"{len(point_inds)} < {self.traj_per_sample}")
            gotit = False
            

        visible_inds_sampled = visibile_pts_inds[point_inds]

        trajs = traj_2d[:, visible_inds_sampled].float()
        visibles = visibility[:, visible_inds_sampled]
        valids = torch.ones((self.seq_len, self.traj_per_sample))

        if self.save_images:
            save_track_vid(rgbs, trajs, visibles, os.path.join(interm_save_dir, 'intermediate_vids',f'{seq_name}_all_points'), f'{seq_name}_all_points')
            pts = [[31, 20]]
            for pt in pts:
                # pdb.set_trace()
                if visibles.shape[1]>31:
                    save_track_vid(rgbs, trajs[:,pt,:], visibles[:,pt], os.path.join(interm_save_dir, 'intermediate_vids',f'{seq_name}_{pt}'), f'{pt}', 15)
            # pdb.set_trace()

        # permute rgbs to be num_frames x 3 x H x w
        rgbs = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float()
        
        sample = CoTrackerData(
            video=rgbs,
            trajectory=trajs,
            visibility=visibles,
            valid=valids,
            seq_name=seq_name,
        )
        return sample, gotit

    def __len__(self):
        return len(self.seq_names)