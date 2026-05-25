import random
import pandas as pd
import torch
import torch.utils.data as data
from scipy import ndimage
import numpy as np
import cv2


class PublicDataset(data.Dataset):
    def __init__(self, csv_data_path, root, is_trans, img_size):
        self.img_size = img_size
        self.is_trans = is_trans
        self.root = root
        self.data = pd.read_csv(csv_data_path, encoding='utf-8')

        self.folder = self.data['folder'].values.tolist()
        self.patient = self.data['patient'].values.tolist()
        self.npz_name = self.data['npz_name'].values.tolist()
        self.label = self.data['label'].values.tolist()
        self.size = len(self.data)

        self.clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))

    def __len__(self):
        return self.size

    def normalization(self, img):
        img_normal = (img - img.min()) / (img.max() - img.min())
        return img_normal

    def normalize_to_255(self, img, min_raw, max_raw):
        return ((img - min_raw) / (max_raw - min_raw) * 255).clip(0, 255).astype(np.uint8)

    def apply_window(self, img_norm, ww=200, wl=128):
        lower = wl - ww / 2
        upper = wl + ww / 2
        return np.clip((img_norm - lower) / ww * 255, 0, 255).astype(np.uint8)

    def __getitem__(self, index):
        data = np.load(self.root + '/' + self.folder[index] + '/' + self.npz_name[index])

        cine_image = data['cine']
        lge_image = data['lge']
        t2_image = data['t2']
        try:
            seg_mask = data['label']
            seg_mask = np.where(seg_mask == 2, 4, seg_mask)
            seg_mask = np.where(seg_mask == 1, 2, seg_mask)
            seg_mask = np.where(seg_mask == 4, 1, seg_mask)
        except:
            seg_mask = data['seg_mask_all']

        cine_right = cine_image

        nor_cine_right = self.normalization(cine_right)

        t2_A_norm = self.normalize_to_255(t2_image, t2_image.min(), t2_image.max())  # GE
        t2_A_final = self.apply_window(t2_A_norm, ww=180, wl=t2_A_norm.mean() + 50)
        t2_A_final = self.normalization(t2_A_final)
        lge_A_final = self.normalization(lge_image)

        nor_cine_right = nor_cine_right * 2 - 1
        nor_lge_image = lge_A_final * 2 - 1
        nor_t2_image = t2_A_final * 2 - 1

        lge = torch.tensor(nor_lge_image.copy()).unsqueeze(0)  # [1,128,128]
        t2 = torch.tensor(nor_t2_image.copy()).unsqueeze(0)
        cine_right = torch.tensor(nor_cine_right.copy()).unsqueeze(0)

        seg_mask = torch.tensor((2 * seg_mask.copy() / 3) - 1).unsqueeze(0)
        x = torch.randn_like(seg_mask)
        label = self.label[index]

        return {'cine_right': cine_right.float(), 'lge': lge.float(), 't2': t2.float(),
                'seg_mask': seg_mask.float(), 'npz_name': self.npz_name[index], 'label': torch.tensor(label)}


class ZSDataset(data.Dataset):
    def __init__(self, csv_data_path, root, is_trans, img_size):
        self.img_size = img_size
        self.is_trans = is_trans
        self.root = root
        self.data = pd.read_csv(csv_data_path, encoding='utf-8')

        self.folder = self.data['folder'].values.tolist()
        self.patient = self.data['patient'].values.tolist()
        self.npz_name = self.data['npz_name'].values.tolist()
        self.Pcine_index = self.data['Pcine_index'].values.tolist()
        self.Mf = self.data['Mf'].values.tolist()
        self.Mfs = self.data['Mfs'].values.tolist()
        self.size = len(self.data)

        self.clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))

    def __len__(self):
        return self.size

    def normalization(self, img):
        img_normal = (img - img.min()) / (img.max() - img.min())
        return img_normal

    def normalize_to_255(self, img, min_raw, max_raw):
        return ((img - min_raw) / (max_raw - min_raw) * 255).clip(0, 255).astype(np.uint8)

    def apply_window(self, img_norm, ww=200, wl=128):
        lower = wl - ww / 2
        upper = wl + ww / 2
        return np.clip((img_norm - lower) / ww * 255, 0, 255).astype(np.uint8)

    def get_style_label(self, mf, mfs):
        label = 0
        if mf == 'PHI':
            label = 3
        elif mf == 'SIE':
            label = 2
        elif mf == 'GE':
            label = 1
        elif mf == 'UIH':
            label = 3
        return label

    def __getitem__(self, index):
        data = np.load(self.root + '/' + self.folder[index] + '/' + self.npz_name[index])

        cine_image = data['cine']
        lge_image = data['lge']
        t2_image = data['t2']
        seg_mask = data['seg_mask_all']

        cine_right = cine_image[self.Pcine_index[index]]
        nor_cine_right = self.normalization(cine_right)

        t2_A_norm = self.normalize_to_255(t2_image, t2_image.min(), t2_image.max())  # GE
        t2_A_final = self.apply_window(t2_A_norm, ww=180, wl=t2_A_norm.mean() + 50)
        t2_A_final = self.normalization(t2_A_final)
        lge_A_final = self.normalization(lge_image)

        nor_cine_right = nor_cine_right * 2 - 1
        nor_lge_image = lge_A_final * 2 - 1
        nor_t2_image = t2_A_final * 2 - 1

        lge = torch.tensor(nor_lge_image.copy()).unsqueeze(0)  # [1,128,128]
        t2 = torch.tensor(nor_t2_image.copy()).unsqueeze(0)
        cine_right = torch.tensor(nor_cine_right.copy()).unsqueeze(0)

        seg_mask = torch.tensor((2 * seg_mask.copy() / 3) - 1).unsqueeze(0)

        x = torch.randn_like(seg_mask)
        label = self.get_style_label(self.Mf[index], self.Mfs[index])

        return {'cine_right': cine_right.float(), 'lge': lge.float(), 't2': t2.float(),
                'seg_mask': seg_mask.float(), 'npz_name': self.npz_name[index], 'label': torch.tensor(label)}


class Cine2LGEDataset(data.Dataset):
    def __init__(self, csv_data_path, root, is_trans, img_size):
        self.img_size = img_size
        self.is_trans = is_trans
        self.root = root
        self.data = pd.read_csv(csv_data_path, encoding='utf-8')

        self.folder = self.data['folder'].values.tolist()
        self.patient = self.data['patient'].values.tolist()
        self.npz_name = self.data['npz_name'].values.tolist()
        self.Pcine_index = self.data['Pcine_index'].values.tolist()
        self.Mf = self.data['Mf'].values.tolist()
        self.Mfs = self.data['Mfs'].values.tolist()
        self.size = len(self.data)

        self.clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))

    def __len__(self):
        return self.size

    def normalization(self, img):
        img_normal = (img - img.min()) / (img.max() - img.min())
        return img_normal  # Image.fromarray(np.uint8(img_normal * 255))

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def get_fft_image(self, img_gray):
        f = np.fft.fft2(img_gray)

        fshift = np.fft.fftshift(f)
        rows, cols = fshift.shape
        mid_x, mid_y = int((rows) / 2), (int((cols) / 2))

        mask1 = np.ones((rows, cols), dtype=np.uint8)
        mask1[mid_x - 1:mid_x + 1, mid_y - 1:mid_y + 1] = 0
        fshift1 = mask1 * fshift
        isshift1 = np.fft.ifftshift(fshift1)

        mask2 = np.zeros((rows, cols), dtype=np.uint8)
        mask2[mid_x - 30:mid_x + 30, mid_y - 30:mid_y + 30] = 1
        fshift2 = mask2 * fshift
        isshift2 = np.fft.ifftshift(fshift2)

        high = np.fft.ifft2(isshift1)
        low = np.fft.ifft2(isshift2)

        img_high = np.abs(high)
        img_low = np.abs(low)
        return img_high, img_low

    def get_transformer_enhance(self, data, angle):
        def rotate_image(yuan_img, angle=90):
            rotated_array = ndimage.rotate(yuan_img, angle=angle, reshape=False)
            return rotated_array

        def rotate_image_multi(crop_cine, angle=90):
            all_cine = []
            x, y, z = crop_cine.shape
            min_idx = min(x, y, z)
            if min_idx == z:
                crop_cine = np.transpose(crop_cine, (2, 0, 1))
            for i in range(min_idx):
                nor = rotate_image(crop_cine[i], angle)
                all_cine.append(nor)
            crop_cine = np.stack(all_cine, axis=0)
            if min_idx == z:
                crop_cine = np.transpose(crop_cine, (1, 2, 0))
            return crop_cine

        keys = data.files
        # angle = 180  # (-90,0,90,180)
        all_image = []
        for i in range(len(keys)):
            temp_image = data[keys[i]]
            new_image = None
            if len(temp_image.shape) == 2:
                new_image = rotate_image(temp_image, angle=angle)
            if len(temp_image.shape) == 3:
                new_image = rotate_image_multi(temp_image, angle=angle)
            all_image.append(new_image)
        ro_data = {}
        for j in range(len(keys)):
            ro_data[keys[j]] = all_image[j]
        return ro_data

    def get_style_label(self, mf, mfs):
        label = 0
        if mf == 'PHI':
            if mfs == float(1.5):
                label = 0
            else:
                label = 3
        elif mf == 'SIE':
            label = 2
        elif mf == 'GE':
            label = 1
        elif mf == 'UIH':
            label = 3
        return label

    def normalize_to_255(self, img, min_raw, max_raw):
        return ((img - min_raw) / (max_raw - min_raw) * 255).clip(0, 255).astype(np.uint8)

    def apply_window(self, img_norm, ww=200, wl=128):
        lower = wl - ww / 2
        upper = wl + ww / 2
        return np.clip((img_norm - lower) / ww * 255, 0, 255).astype(np.uint8)

    def __getitem__(self, index):
        data = np.load(self.root + '/' + self.folder[index] + '/' + self.npz_name[index])
        if self.is_trans:
            angle = random.choice([-90, 0, 90, 180, 0, 0, 0])
            data = self.get_transformer_enhance(data, angle)
        cine_image = data['cine']
        lge_image = data['lge']
        t2_image = data['t2']
        seg_mask = data['seg_mask']

        label = self.get_style_label(self.Mf[index], self.Mfs[index])
        cine_right = cine_image[self.Pcine_index[index]]

        nor_lge_image = self.normalization(lge_image)
        nor_cine_right = self.normalization(cine_right)

        img_A = t2_image
        img_A_norm = self.normalize_to_255(img_A, img_A.min(), img_A.max())
        img_A_final = self.apply_window(img_A_norm, ww=180, wl=img_A_norm.mean() + 50)
        img_A_final = self.normalization(img_A_final)

        nor_cine_right = nor_cine_right * 2 - 1
        nor_lge_image = nor_lge_image * 2 - 1
        nor_t2_image = img_A_final * 2 - 1

        cine_right = torch.tensor(nor_cine_right.copy()).unsqueeze(0)
        lge = torch.tensor(nor_lge_image.copy()).unsqueeze(0)
        t2 = torch.tensor(nor_t2_image.copy()).unsqueeze(0)
        seg_mask = torch.tensor((2 * seg_mask.copy() / 3) - 1).unsqueeze(0)

        return {'cine_right': cine_right.float(), 'lge': lge.float(), 't2': t2.float(),
                'seg_mask': seg_mask.float(), 'npz_name': self.npz_name[index], 'label': torch.tensor(label)}


def divide_data(all_data_path, save_train, save_val, save_test):
    dataframe = pd.read_csv(all_data_path, encoding="utf-8")
    image_info = dataframe["patient"].values.tolist()
    image_info = list(set(image_info))
    random.seed(42)
    random.shuffle(image_info)
    scale = [8, 0, 2]
    fen = 10
    train_ID = image_info[:int(len(image_info) * (scale[0] / fen))]
    val_ID = image_info[int(len(image_info) * (scale[0] / fen)):int(len(image_info) * ((scale[0] + scale[1]) / fen))]
    test_ID = image_info[int(len(image_info) * ((scale[0] + scale[1]) / fen)):]

    train_info = dataframe[dataframe["patient"].isin(train_ID)]
    val_info = dataframe[dataframe["patient"].isin(val_ID)]
    test_info = dataframe[dataframe["patient"].isin(test_ID)]

    train_info.to_csv(save_train, index=None)
    val_info.to_csv(save_val, index=None)
    test_info.to_csv(save_test, index=None)


if __name__ == '__main__':
    divide_data(all_data_path="E:/2025Project/AMI_Multitask_lynx/data_excel/data_PLA_Lynx.csv",
                save_train="E:/2025Project/AMI_Multitask_lynx/data_excel/data_PLA_Lynx_train.csv",
                save_val="E:/2025Project/AMI_Multitask_lynx/data_excel/data_PLA_Lynx_val.csv",
                save_test="E:/2025Project/AMI_Multitask_lynx/data_excel/data_PLA_Lynx_test.csv")


