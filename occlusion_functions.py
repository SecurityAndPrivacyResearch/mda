import torch
import math
from torchvision.transforms.functional import erase
from functools import cmp_to_key
import imageio
import os
import re
import itertools

def find_txt_candidates(C, img_name, img, model, text, z):

  tokens = text.split()

  for k in range(len(tokens)):

    temp = list(tokens)
    del temp[k]
    temp_text = ' '.join(temp)

    output = model.classify(img, temp_text, img)
    pred_conf_o, pred_label_o, z_o = output['confidence'], output['label'], output['pooled_output']

    if torch.argmax(z, 1).item() != torch.argmax(z_o, 1).item():
      C.append(k)

def occlude_quadrants(img, orig, i):
  occluded_img_1 = erase(img, orig[0] + 0, orig[1] + 0, i, i, torch.tensor(0).cuda())
  occluded_img_2 = erase(img, orig[0] + i, orig[1] + 0, i, i, torch.tensor(0).cuda())
  occluded_img_3 = erase(img, orig[0] + 0, orig[1] + i, i, i, torch.tensor(0).cuda())
  occluded_img_4 = erase(img, orig[0] + i, orig[1] + i, i, i, torch.tensor(0).cuda())

  return occluded_img_1, occluded_img_2, occluded_img_3, occluded_img_4

def find_img_candidates(i, orig, img_name, img, C, model, text, z):

  if i <= 10:
    return

  i = int(math.ceil(i / 2))

  # occlude quandrants
  occluded_img_1, occluded_img_2, occluded_img_3, occluded_img_4 = occlude_quadrants(img, orig, i)

  with torch.no_grad():
    output1 = model.classify(occluded_img_1, text, occluded_img_1)
    pred_conf_o1, pred_label_o1, z_o1 = output1['confidence'], output1['label'], output1['pooled_output']

    output2 = model.classify(occluded_img_2, text, occluded_img_2)
    pred_conf_o2, pred_label_o2, z_o2 = output2['confidence'], output2['label'], output2['pooled_output']

    output3 = model.classify(occluded_img_3, text, occluded_img_3)
    pred_conf_o3, pred_label_o3, z_o3 = output3['confidence'], output3['label'], output3['pooled_output']

    output4 = model.classify(occluded_img_4, text, occluded_img_4)
    pred_conf_o4, pred_label_o4, z_o4 = output4['confidence'], output4['label'], output4['pooled_output']

  # drill down to the relevant parts
  if torch.argmax(z, 1).item() != torch.argmax(z_o1, 1).item():
    new_orig = (orig[0] + 0, orig[1] + 0)
    C.append((orig, 0, 0, i, i))
    find_img_candidates(i, new_orig, img_name, img, C, model, text, z)

  if torch.argmax(z, 1).item() != torch.argmax(z_o2, 1).item():
    new_orig = (orig[0] + i, orig[1] + 0)
    C.append((orig, i, 0, i, i))
    find_img_candidates(i, new_orig, img_name, img, C, model, text, z)

  if torch.argmax(z, 1).item() != torch.argmax(z_o3, 1).item():
    new_orig = (orig[0] + 0, orig[1] + i)
    C.append((orig, 0, i, i, i))
    find_img_candidates(i, new_orig, img_name, img, C, model, text, z)

  if torch.argmax(z, 1).item() != torch.argmax(z_o4, 1).item():
    new_orig = (orig[0] + i, orig[1] + i)
    C.append((orig, i, i, i, i))
    find_img_candidates(i, new_orig, img_name, img, C, model, text, z)


def compare(item1, item2):
  if item1[3] * item1[4] < item2[3] * item2[4]:
    return -1
  elif item1[3] * item1[4] > item2[3] * item2[4]:
    return 1
  else:
    return 0 
 
def pointwise_occlusion(C_i, C_t, img_name, img, model, text, z, tgt):
  # order salience of img regions by area
  salient_C_i = sorted(C_i, key = cmp_to_key(compare))

  img_h = img.size(1)
  img_w = img.size(2)
  success = False

  tokens = text.split()

  total_points = len(tokens) + img_h * img_w
  points = 0

  # text data points
  success_text_point = []
  text_fool_label = None

  # compute all word combinations
  combos = []
  for l in range(1, len(C_t) + 1):
    combo = itertools.combinations(C_t, l)
    combos.extend(list(combo))

  for combo in combos:
    # occlude all the points in this combo
    temp = list(tokens)
    for ct in combo:
      #del temp[ct]
      temp[ct] = ''
    temp_text = ' '.join(temp)

    output_o = model.classify(img, temp_text, img)
    pred_conf_o, pred_label_o, z_o = output_o['confidence'], output_o['label'], output_o['pooled_output'] 

    print(combo, 'TEXTPOINT', '\t', torch.argmax(z, 1).item(), torch.argmax(z_o, 1).item(), '\t', tgt, pred_label_o, pred_conf_o)

    #if success == False and pred_label_o != tgt:
    if pred_label_o != tgt:
      success = True
      points = (len(combo) / total_points) * 100
      success_text_point.append(combo)
      text_fool_label = pred_label_o
      break # no need to try other combos.

  # make copies of img and im
  occluded_img = img.detach().clone().cuda()

  #img_occlusion_success = False
  img_fool_label = None
  success_img_point = None

  # image data points
  for ci in salient_C_i:
    orig, i, j, h, w = ci
    occluded_img = erase(occluded_img, orig[0] + i, orig[1] + j, h, w, torch.tensor(0).cuda()) 

    output_o = model.classify(occluded_img, text, occluded_img)
    pred_conf_o, pred_label_o, z_o = output_o['confidence'], output_o['label'], output_o['pooled_output']

    print(ci, 'IMGPOINT', '\t', torch.argmax(z, 1).item(), torch.argmax(z_o, 1).item(), '\t', tgt, pred_label_o, pred_conf_o)

    if tgt != pred_label_o:
      success = True
      points = ((1 + h * w) / total_points) * 100

    # save this image
    #if tgt != pred_label_o and img_occlusion_success == False:
    if tgt != pred_label_o:
      #img_occlusion_success = True
      save_img = occluded_img.squeeze().cpu().detach().numpy().transpose(1, 2, 0) * 255 
      img_file_name = img_name.split('/')[-1].replace('.jpg', '')
      t_save = re.sub('[^A-Za-z0-9 ]+', '', text)
      t_save = t_save.replace(' ', '_')
      imageio.imwrite(os.path.join('occluded_samples', img_file_name + '_' + t_save + '.png'), save_img)
      success_img_point = ci
      img_fool_label = pred_label_o
      success = True
      break

  print('Result:', success, ' % Points occluded:', points)
  return success, points, success_img_point, success_text_point, text_fool_label, img_fool_label
 
# ---------------------------------------------------------------------------------------------------------

def pointwise_occlusion_combination(C_i, C_t, img_name, img, model, text, z, tgt, max_iter = 500):

  if len(C_i) == 0 and len(C_t) == 0:
    print('Result:', False, ' % Points occluded:', None)
    return False, 0, None, None 

  # order salience of img regions by area
  salient_C_i = sorted(C_i, key = cmp_to_key(compare))

  img_h = img.size(1)
  img_w = img.size(2)
  success = False

  # text data points
  fool_label = None
  tokens = text.split()
  total_points = len(tokens) + img_h * img_w
  success_points = None

  # Combining text and image
  C_c = []
  for i in salient_C_i:
    C_c.append(('img', i))
  for i in C_t:
    C_c.append(('text', i))

  # compute all image and word combinations
  #combos = []
  #for l in range(1, len(C_c) + 1):
  #  combo = itertools.combinations(C_c, l)
  #  combos.extend(list(combo))

  #import IPython; IPython.embed(); exit(1)
  #for count, combo in enumerate(combos):
  count = 0

  for l in range(1, len(C_c) + 1):
    combos = list(itertools.combinations(C_c, l))
    #combos = [j[0] for j in combo]
    #print(combo)
    #import IPython; IPython.embed(); exit(1)
    #for count, combo in enumerate(combos):
    for combo in combos:  
      occluded_img = img.detach().clone().cuda()
      temp_tokens = list(tokens)
      count += 1
      points = 0.

      for ind_combo, cc in enumerate(combo):
        # Apply img occlusion
        #import IPython; IPython.embed(); exit(1)
        if cc[0] == 'img':
          orig, i, j, h, w = cc[1]
          occluded_img = erase(occluded_img, orig[0] + i, orig[1] + j, h, w, torch.tensor(0).cuda())
          if ind_combo == len(combo) - 1:
            points += h * w

        # Apply text occlusion
        if cc[0] == 'text':
          temp_tokens[cc[1]] = ''  
          if ind_combo == len(combo) - 1:
            points += 1

      temp_text = ' '.join(temp_tokens) 
      with torch.no_grad():
        output_o = model.classify(occluded_img, temp_text, occluded_img)
      pred_conf_o, pred_label_o, z_o = output_o['confidence'], output_o['label'], output_o['pooled_output']
      
      #import IPython; IPython.embed(); exit(1)
      if tgt != pred_label_o:
        success = True
        save_img = occluded_img.squeeze().cpu().detach().numpy().transpose(1, 2, 0) * 255
        img_file_name = img_name.split('/')[-1].replace('.jpg', '')
        t_save = re.sub('[^A-Za-z0-9 ]+', '', text)
        t_save = t_save.replace(' ', '_')
        imageio.imwrite(os.path.join('occluded_samples', img_file_name + '_' + t_save + '.png'), save_img)
        success_points = cc
        fool_label = pred_label_o
        points = (points / total_points) * 100
        #import IPython; IPython.embed(); exit(1)
        print('Result:', success, ' % Points occluded:', points)
        return success, points, success_points, fool_label

      if count == max_iter:
        print('Result:', success, ' % Points occluded:', points)
        return success, points, success_points, fool_label
      #print(count)  

  # if its come here then we failed
  print('Result:', success, ' % Points occluded:', points)
  return success, points, success_points, fool_label

 






