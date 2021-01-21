#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# download data

os.environ['KAGGLE_USERNAME'] = "***" # username from the json file
os.environ['KAGGLE_KEY'] = "***" # key from the json file
get_ipython().system('kaggle datasets download -d aishwr/coco2017')

zip__file = zipfile.ZipFile("/content/coco2017.zip")
zip__file.extract('annotations_trainval2017/annotations/person_keypoints_train2017.json')
with open("/content/annotations_trainval2017/annotations/person_keypoints_train2017.json" , "r") as f:
    annotations  = json.load(f)

image_paths = []
for val in annotations["images"]:
    file_name       = val['file_name']
    img_path        = os.path.join("train2017/train2017", file_name)
    image_paths.append(img_path)

for name in image_paths:
    zip__file.extract(name)


coco = COCO("/content/annotations_trainval2017/annotations/person_keypoints_train2017.json")

