#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# some function for augmentations


def joints_to_point8(joints, num_p=18):
    
    segment = np.zeros((num_p * len(joints), 2), dtype=float)

    for idx_all, j_list in enumerate(joints):
        for idx, k in enumerate(j_list):
            if k:
                segment[idx_all * num_p + idx, 0] = k[0]
                segment[idx_all * num_p + idx, 1] = k[1]
            else:
                segment[idx_all * num_p + idx, 0] = -1000000
                segment[idx_all * num_p + idx, 1] = -1000000

    return segment



def point8_to_joints(points, num_p=18):
    l = points.shape[0] // num_p

    all = []
    for i in range(l):
        skel = []
        for j in range(num_p):
            idx = i * num_p + j
            x = points[idx, 0]
            y = points[idx, 1]

            if x <= 0 or y <= 0 or x > 2000 or y > 2000:
                skel.append(None)
            else:
                skel.append((x, y))

        all.append(skel)
    return all


def gen_mask__(masks_segments,dim ):

    if masks_segments:
        mask_miss = np.ones((dim[0], dim[1]), dtype=np.uint8)
        for seg in masks_segments:
            bin_mask  = maskUtils.decode(seg)
            bin_mask  = np.logical_not(bin_mask)
            mask_miss = np.bitwise_and(mask_miss, bin_mask)

        mask = mask_miss
        return mask

    return np.ones((dim[0], dim[1]), dtype=np.uint8)


def apply_mask__(img , mask):

  img[:, :, 0] = img[:, :, 0] * mask
  img[:, :, 1] = img[:, :, 1] * mask
  img[:, :, 2] = img[:, :, 2] * mask
  return img


def Rotation__(img , all_joints , mask):
  (h, w)               = img.shape[:2]
  (center_x, center_y) = (w // 2, h // 2)
  R                    = cv2.getRotationMatrix2D((center_x, center_y), np.random.uniform(low=-1 , high= 1 , size = 1)[0] * 40 , 1.0)
  (h, w)               = img.shape[:2]
  cos                  = np.abs(R[0, 0])
  sin                  = np.abs(R[0, 1])
  new_w                = int((h * sin) + (w * cos))
  new_h                = int((h * cos) + (w * sin))
  R[0, 2]             += (new_w / 2) - center_x
  R[1, 2]             += (new_h / 2) - center_y

  image                = cv2.warpAffine( img , R ,(new_w,new_h))
  mask                 = cv2.warpAffine( mask , R ,(new_w,new_h))

  aug_joints           = np.array(joints_to_point8(all_joints))
  aug_joints           = np.concatenate((aug_joints , np.ones((aug_joints.shape[0] , 1))) , axis = 1)
  aug_joints           = np.matmul(R ,aug_joints.T ).T

  return image , point8_to_joints(aug_joints) , mask

def flip__(img , aug_joints ,mask):
  (h, w) = img.shape[:2]
  if np.random.random() > .5:
    img               = cv2.flip(img , 1)
    mask              = cv2.flip(mask , 1)
    aug_joints        = np.array(joints_to_point8(aug_joints))
    aug_joints[:, 0]  = w - aug_joints[:, 0]
    
    return img , point8_to_joints(aug_joints) , mask
  return img , aug_joints , mask

def scale__(img , aug_joints , mask ):
  sx = np.random.uniform(low= .9 , high=1.1 , size=1)[0]
  sy = np.random.uniform(low= .9 , high=1.1 , size=1)[0]
  M  = np.array([[sx , 0  , 0] ,
                [0  , sy , 0]])
  
  (h, w)               = img.shape[:2]
  new_h                = int(sy * h)
  new_w                = int(sx * w)
  image = cv2.warpAffine(img , M , (new_w , new_h))
  mask  = cv2.warpAffine(mask , M , (new_w , new_h))

  aug_joints        = np.array(joints_to_point8(aug_joints))
  aug_joints        = np.concatenate((aug_joints , np.ones((aug_joints.shape[0] , 1))) , axis = 1)
  aug_joints        = np.matmul(M ,aug_joints.T ).T
  
  return image , point8_to_joints(aug_joints) , mask


def resize__(img , aug_joints , mask):

  (h, w) = img.shape[:2]
  image  = cv2.resize(img , (368 , 368))
  mask   = cv2.resize( mask , (368 , 368))

  sy = 368 / h
  sx = 368 / w
  M  = np.array([[sx , 0  , 0] ,
                [0  , sy , 0]])
  
  aug_joints        = np.array(joints_to_point8(aug_joints))
  aug_joints        = np.concatenate((aug_joints , np.ones((aug_joints.shape[0] , 1))) , axis = 1)
  aug_joints        = np.matmul(M ,aug_joints.T ).T
  
  return image , point8_to_joints(aug_joints) , mask


def build_sample_(image , mask , aug_joints):
    if mask.min() == 1 :
        mask_paf     = ALL_PAF_MASK
        mask_heatmap = ALL_HEATMAP_MASK
    else:
        mask_paf     = create_all_mask(mask, 38, stride=8)
        mask_heatmap = create_all_mask(mask, 19, stride=8)

    heatmap = create_heatmap(num_joints_and_bkg, 46, 46,
                             aug_joints, 7.0, stride=8)

    pafmap  = create_paf(num_connections, 46, 46,
                        aug_joints, 1, stride=8)

    return [image.astype(np.uint8), mask_paf, mask_heatmap, pafmap, heatmap]

