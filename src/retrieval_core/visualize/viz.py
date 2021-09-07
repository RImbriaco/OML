import os
from tqdm import tqdm
import numpy as np
import cv2
from PIL import ImageDraw, Image
from src.common.io.tiff.load_tiff import load_rgb
from src.retrieval_core.dataloader.loaders.lmdb_loader import BigEarthLMDB
from src.retrieval_core.dataloader.helpers.image_readers import cv_loader


def load_img(data_loader, idx, img_path):
    """
    Load image from disk.
    :param data_loader: Dataloader object.
    :param idx: Image index.
    :param img_path: Path to images.
    :return:
    NumPy array of image.
    """
    if isinstance(data_loader.dataset, BigEarthLMDB):
        data = data_loader.dataset[idx]
        try:
            path = data['name']
        except KeyError:
            path = data['patch_name']
        return cv2.resize(load_rgb(os.path.join(img_path,path, path)), (256, 256))
    else:
        return cv_loader(os.path.join(img_path,
                                      data_loader.dataset.img_list[idx] + '.jpg'))


def save_matches(res_dict, data_loader, img_path, save_path):
    """
    Check the retrieved images and creates a mosaic depicting queries
    and matches.
    :param res_dict: Result dictionary.
    :param data_loader: Datalodaer object.
    :param img_path: Path to images.
    :param save_path: Output path.
    """
    data_loader.dataset.imgTransform = None

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    indices = res_dict['indices']
    dist_top = res_dict['dist_top']
    qr_idx = res_dict['qr_idx']
    db_idx = res_dict['db_idx']

    for en, idx in enumerate(tqdm(qr_idx)):
        query = load_img(data_loader, idx, img_path)
        if query.dtype == np.float64:
            query = (query * 255).astype(np.uint8)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        top10_img = np.zeros((11, query.shape[0], query.shape[1], query.shape[2]))
        top10_img[0] = query

        top_matches = indices[:10, en]
        distances = dist_top[:10, en]

        for en_match, tm in enumerate(top_matches):
            match = load_img(data_loader, db_idx[tm], img_path)
            match = cv2.rectangle(match, (256 - 64, 256 - 64), (256, 256),
                                  (0, 0, 0), -1)

            if match.dtype == np.float64:
                match = (match*255).astype(np.uint8)

            fmt = '{:.2f}'
            fmt = fmt.format(distances[en_match])
            pil_im = Image.fromarray(match)
            draw = ImageDraw.Draw(pil_im)
            draw.text((256 - 60, 256 - 48), fmt)
            cv2_im_processed = np.array(pil_im)
            top10_img[en_match+1] = cv2_im_processed

        cv2.imwrite(os.path.join(save_path, '{}_{:.2f}.png'.format(en,np.sum(distances))),
                    np.swapaxes(top10_img, 0, 1).reshape(256, 256 * 11, 3))





