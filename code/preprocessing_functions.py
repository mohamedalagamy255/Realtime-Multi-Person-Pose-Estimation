#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# some functions for preprocessing 


def create_all_mask(mask, num, stride):

    scale_factor = 1.0 / stride
    small_mask   = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    small_mask   = small_mask[:, :, np.newaxis]
    return np.repeat(small_mask, num, axis=2)



class Meta(object):
  def __init__(self, img_path, height, width, center, bbox,
                 area, scale, num_keypoints):

    self.img_path       = img_path
    self.height         = height
    self.width          = width
    self.center         = center
    self.bbox           = bbox
    self.area           = area
    self.scale          = scale
    self.num_keypoints  = num_keypoints

    # updated after iterating over all persons
    self.masks_segments = None
    self.all_joints     = None

    # updated during augmentation
    self.img            = None
    self.mask           = None
    self.aug_center     = None
    self.aug_joints     = None


def _get_neck(coco_parts, idx1, idx2):

  p1 = coco_parts[idx1]
  p2 = coco_parts[idx2]
  if p1 and p2:
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
  else:
    return None


def from_coco_keypoints(all_keypoints, w ,h):

  all_joints = []
  for keypoints in all_keypoints:
    kp = np.array(keypoints)
    xs = kp[0::3]
    ys = kp[1::3]
    vs = kp[2::3]


    keypoints_list = []
    for idx, (x, y, v) in enumerate(zip(xs, ys, vs)):
      # only visible and occluded keypoints are used
      if v >= 1 and x >=0 and y >= 0 and x < w and y < h:
        keypoints_list.append((x, y))
      else:
        keypoints_list.append(None)

    joints = []
    for part_idx in range(len(idx_in_coco)):
      coco_kp_idx = idx_in_coco[part_idx]

      if callable(coco_kp_idx):
        p = coco_kp_idx(keypoints_list)
      else:
        p = keypoints_list[coco_kp_idx]

      joints.append(p)
    all_joints.append(joints)

  return all_joints





def prepare(annotations ,target_size ):
  all_meta = []
  for i,image_ in enumerate(annotations["images"]):
    h               = image_['height']
    w               = image_['width']
    file_name       = image_['file_name']
    img_path        = os.path.join("/content/train2017/train2017", file_name)

    image_anns      = [ann__ for ann__ in annotations["annotations"] if ann__['image_id'] == image_['id']]
    total_keypoints = sum([ann.get('num_keypoints', 0) for ann in image_anns])

    if total_keypoints == 0:
      continue

    persons     = []
    prev_center = []
    masks       = []
    keypoints   = []

    persons_ids = np.argsort([-a['area'] for a in image_anns], kind='mergesort')

    for id in list(persons_ids):
      person_meta = image_anns[id]
      if person_meta["iscrowd"]:
        masks.append(coco.annToRLE(person_meta))
        continue

      if person_meta["num_keypoints"] < 5 or person_meta["area"] < 32 * 32:
        masks.append(coco.annToRLE(person_meta))
        continue
      

      person_center = [person_meta["bbox"][0] + person_meta["bbox"][2] / 2,
                      person_meta["bbox"][1] + person_meta["bbox"][3] / 2]
      
      too_close = False
      for pc in prev_center:
        a    = np.expand_dims(pc[:2], axis=0)
        b    = np.expand_dims(person_center, axis=0)
        dist = cdist(a, b)[0]
        if dist < pc[2]*0.3:
          too_close = True
          break
      
      if too_close:
        masks.append(coco.annToRLE(person_meta))
        continue
      
      pers = Meta(
                  img_path=img_path,
                  height=h,
                  width=w,
                  center=np.expand_dims(person_center, axis=0),
                  bbox=person_meta["bbox"],
                  area=person_meta["area"],
                  scale=person_meta["bbox"][3] / target_size[0],
                  num_keypoints=person_meta["num_keypoints"])
      

      keypoints.append(person_meta["keypoints"])
      persons.append(pers)
      prev_center.append(np.append(person_center, max(person_meta["bbox"][2], person_meta["bbox"][3])))
    
    if len(persons) > 0:
      main_person                = persons[0]
      main_person.masks_segments = masks
      main_person.all_joints     = from_coco_keypoints(keypoints, w, h)
      all_meta.append(main_person)
    if (i+1) % 100 == 0:
      print("images  : %d"%i)
  return all_meta



def create_heatmap(num_maps, height, width, all_joints, sigma, stride):
 
    heatmap = np.zeros((height, width, num_maps), dtype=np.float64)

    for joints in all_joints:
        for plane_idx, joint in enumerate(joints):
            if joint:
                _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride)

    # background
    heatmap[:, :, -1] = np.clip(1.0 - np.amax(heatmap, axis=2), 0.0, 1.0)

    return heatmap


def create_paf(num_maps, height, width, all_joints, threshold, stride):
    
    vectormap = np.zeros((height, width, num_maps*2), dtype=np.float64)
    countmap  = np.zeros((height, width, num_maps), dtype=np.uint8)

    for joints in all_joints:
        for plane_idx, (j_idx1, j_idx2) in enumerate(joint_pairs):
            center_from = joints[j_idx1]
            center_to   = joints[j_idx2]

            # skip if no valid pair of keypoints
            if center_from is None or center_to is None:
                continue

            x1, y1 = (center_from[0] / stride, center_from[1] / stride)
            x2, y2 = (center_to[0] / stride, center_to[1] / stride)

            _put_paf_on_plane(vectormap, countmap, plane_idx, x1, y1, x2, y2,
                              threshold, height, width)

    return vectormap
  
def _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride):
    start = stride / 2.0 - 0.5

    center_x, center_y = joint

    for g_y in range(height):
        for g_x in range(width):
            x        = start + g_x * stride
            y        = start + g_y * stride
            d2       = (x-center_x) * (x-center_x) + (y-center_y) * (y-center_y)
            exponent = d2 / 2.0 / sigma / sigma
            if exponent > 4.6052:
                continue

            heatmap[g_y, g_x, plane_idx] += math.exp(-exponent)
            if heatmap[g_y, g_x, plane_idx] > 1.0:
                heatmap[g_y, g_x, plane_idx] = 1.0



def _put_paf_on_plane(vectormap, countmap, plane_idx, x1, y1, x2, y2,
                     threshold, height, width):

    min_x = max(0, int(round(min(x1, x2) - threshold)))
    max_x = min(width, int(round(max(x1, x2) + threshold)))

    min_y = max(0, int(round(min(y1, y2) - threshold)))
    max_y = min(height, int(round(max(y1, y2) + threshold)))

    vec_x = x2 - x1
    vec_y = y2 - y1

    norm  = math.sqrt(vec_x ** 2 + vec_y ** 2)
    if norm < 1e-8:
        return

    vec_x /= norm
    vec_y /= norm

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - x1
            bec_y = y - y1
            dist  = abs(bec_x * vec_y - bec_y * vec_x)

            if dist > threshold:
                continue

            cnt = countmap[y][x][plane_idx]

            if cnt == 0:
                vectormap[y][x][plane_idx * 2 + 0] = vec_x
                vectormap[y][x][plane_idx * 2 + 1] = vec_y
            else:
                vectormap[y][x][plane_idx*2+0] = (vectormap[y][x][plane_idx*2+0] * cnt + vec_x) / (cnt + 1)
                vectormap[y][x][plane_idx*2+1] = (vectormap[y][x][plane_idx*2+1] * cnt + vec_y) / (cnt + 1)

            countmap[y][x][plane_idx] += 1





