def annotations_by_image(data):
  dataset = {}
  annotations = {}
  names = {}

  id_list = []
  for dico in data["images"]:
    im_id = dico["id"]
    id_list.append(im_id)
    names[im_id] = dico["file_name"][:-4]+".npy"

  for dico in data["annotations"]:
    im_id = dico["image_id"]
    annotations[im_id]=annotations.get(im_id,[])+[dico.copy()]   

  for im_id, lst in annotations.items():
    kps = []
    area = []

    for ls in lst:
      kps_car = []
      if ls["iscrowd"] == 0:
        for i in range(24):
          x,y,z = tuple(ls["keypoints"][i*3:(i+1)*3])
          if( z == 2.0):
            # the point is visible
            cls = i+1
            kps_car.append((cls,x,y))
        if len(kps_car)>0:
          kps.append(kps_car)
          area.append(ls["area"])
    nb=len(kps)
    name = names[im_id]
    dataset[name]=(kps, area, nb)
  return dataset

def get_nb_car_distribution(annotations):
  annot_by_img = annotations_by_image(annotations)
  lst = []
  for k, v in annot_by_img.items():
    lst.append(v[2])

  return lst

def get_nb_keypoints_car_distribution(annotations):
  annot_by_img = annotations_by_image(annotations)
  lst = []
  for k, v in annot_by_img.items():
    lst.extend([len(l) for l in v[0]])

  return lst

def get_keypoint_class_distrib(annotations):
  annot_by_img = annotations_by_image(annotations)
  lst = []
  for k, v in annot_by_img.items():
    lst.extend([l[0] for j in v[0] for l in j])

  return lst

def get_area_distribution(annotations):
  annot_by_img = annotations_by_image(annotations)
  lst = []
  for k, v in annot_by_img.items():
    lst.extend(v[1])

  return lst
