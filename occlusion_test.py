import torch
from mmf.common.registry import registry
import numpy 
from PIL import Image
import torchvision.datasets.folder as tv_helpers
import json
import jsonlines
import os
from occlusion_functions import *

model_cls = registry.get_model_class('mmbt')
model = model_cls.from_pretrained('mmbt.hateful_memes.images')
model.cuda()

dataset_dir = '/panasas/scratch/grp-hongxinh/datasets/data/'

data = [json.loads(i) for i in open(os.path.join(dataset_dir, 'dev_unseen.jsonl')).readlines()]

count = 0

for d in data:
  #if count > 20:
  #  break
  img_path = os.path.join(dataset_dir, d['img'])
  image = tv_helpers.default_loader(img_path)
  image = model.processor_dict["image_processor"](image)
  text = d['text']
  output = model.classify(image, text, image)
  pred_conf, pred_label, z = output['confidence'], output['label'], output['pooled_output']

  if (d['label'] != 1) or (pred_label != d['label']):
    continue    

  # occlude all pixels
  x_prime = image.detach().clone().cuda()
  x_prime[:, :, :] = 0. 

  # select only those where image has some effect on the fusion embedding
  with torch.no_grad():
    output_prime = model.classify(x_prime, text, x_prime)
    pred_conf_prime, pred_label_prime, z_prime = output_prime['confidence'], output_prime['label'], output_prime['pooled_output']

  is_img_important = True
  if torch.argmax(z, 1).item() == torch.argmax(z_prime, 1).item():
    print('Image not important')
    is_img_important = False

  print('Image Name:', img_path)
  print('Text:', text)

  # search for critical points in the image
  C_i = []
  find_img_candidates(224, (0, 0), img_path, image, C_i, model, text, z) 
  print(C_i)
  

  # search for critical points in the text
  C_t = []
  find_txt_candidates(C_t, img_path, image, model, text, z)
  print(C_t)

  #success, points, success_img_point, success_text_point, text_fool_label, img_fool_label = pointwise_occlusion(C_i, C_t, img_path, image, model, text, z, pred_label)
  success, points, success_points, fool_label = pointwise_occlusion_combination(C_i, C_t, img_path, image, model, text, z, pred_label)

  #json_obj = {'img_name': img_path, 'text': text, 'tgt': pred_label,'is_img_important': is_img_important, 'success': success, 'points': points, 'success_img_point': success_img_point, 'success_text_point': success_text_point, 'text_fool_label': text_fool_label, 'img_fool_label': img_fool_label}

  json_obj = {'img_name': img_path, 'text': text, 'tgt': pred_label,'is_img_important': is_img_important, 'success': success, 'points': points, 'success_point': success_points, 'fool_label': fool_label}

  print('-' * 50) 

  count += 1
  
  with jsonlines.open('output.jsonl', mode = 'a') as writer:
    writer.write(json_obj)
 

