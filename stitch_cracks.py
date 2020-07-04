import cv2
import numpy as np
from collections import Counter
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import tqdm
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('out_dir', type=str, default='data/stitched_cracks',
                    help='the otput directory where the final stitched image will be stored')
parser.add_argument('phone_dir', type=str, default='data/phones',
                    help='the input directory of the phone images')
parser.add_argument('crack_dir', type=str, default='data/cracks',
                    help='the input directory of the cracks')
parser.add_argument('out_csv_file_path', type=str, default='labelled_cracks_stitched_2.csv',
                    help='the output csv file where the labels will be written')

args = parser.parse_args()

out_dir = args.out_dir
phone_dir = args.phone_dir
crack_dir = args.crack_dir

if not os.path.exists(out_dir):
  os.makedirs(out_dir)

NUM_PHONES_PER_CRACK = 1
ROTATE_CRACK = False
RESIZE_CRACK = True
# IMAGES_TO_SAMPLE = 10
RESIZE_PHONE_SIZE = 300

CLASS = "crack"
headers = "filename,width,height,class,xmin,ymin,xmax,ymax"
out_file_path = args.out_csv_file_path
out_file = open(out_file_path, 'w')
out_file.write("{}\n".format(headers))
out_file.flush()


def stitch_image(img, img_to_be_stiched):
  print(img.shape,img_to_be_stiched.shape)

  img2 = img_to_be_stiched
  if img.shape[0]<img2.shape[0]:
    h = img.shape[0]-10
    w = int(img2.shape[1]/img2.shape[0]*h)
    img2 = cv2.resize(img2,(w,h))
  elif img.shape[1]<img2.shape[1]:
    w = img.shape[1]-10
    h = int(img2.shape[0]/img2.shape[1]*w)
    img2 = cv2.resize(img2,(w,h))

  PATCH_SIZE_y = int(img2.shape[0])
  PATCH_SIZE_x = int(img2.shape[1])

  if PATCH_SIZE_x>img.shape[1]-12:
    PATCH_SIZE_x = img.shape[1]-24
    PATCH_SIZE_y = int(img2.shape[0]/img2.shape[1]*PATCH_SIZE_x)
  elif PATCH_SIZE_y>img.shape[0]-12:
    PATCH_SIZE_y = img.shape[0]-24
    PATCH_SIZE_x = int(img2.shape[1]/img2.shape[0]*PATCH_SIZE_y)
  print(PATCH_SIZE_x,PATCH_SIZE_y)

  img2 = cv2.resize(img2,(PATCH_SIZE_x,PATCH_SIZE_y))
  img = np.array(img,dtype=np.float32)
  img2 = np.array(img2,dtype=np.float32)
  # print(img.shape)
  # print(img2.shape)
  kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
  kernel_1 = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

  window_name = ('Sobel')
  scale = 1
  delta = 0
  ddepth = cv2.CV_16S

  # Remove noise
  # img_blur = cv2.GaussianBlur(img, (3, 3), 0)
  img_blur = cv2.filter2D(img,-1,kernel_1)
  # img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

  # Apply Laplacian
  # _img_laplacian1 = cv2.filter2D(img_gray,-1,kernel)
  # _img_laplacian1 = np.expand_dims(_img_laplacian1,axis=-1)
  # img_laplacian1 = np.concatenate((_img_laplacian1,_img_laplacian1,_img_laplacian1),axis=-1)
  img_laplacian1 = cv2.filter2D(img,-1,kernel,borderType=cv2.BORDER_REPLICATE)

  # img_laplacian2 = cv2.filter2D(img2,-1,kernel)
  img_laplacian2 = np.zeros(img.shape)
  margin_x = np.ceil(img.shape[1]*0.1)
  margin_y = np.ceil(img.shape[0]*0.1)
  print('==')
  print(img.shape[0],img.shape[1],PATCH_SIZE_x,margin_x,img.shape[1]-PATCH_SIZE_x-margin_x-2)
  x1 = np.random.randint(margin_x,img.shape[1]-PATCH_SIZE_x-margin_x-2)
  x2 = x1 + PATCH_SIZE_x
  print('==')
  print(margin_y,img.shape[0]-PATCH_SIZE_y-margin_y-2)
  y1 = np.random.randint(margin_y,img.shape[0]-PATCH_SIZE_y-margin_y-2)
  y2 = y1 + PATCH_SIZE_y
  img_laplacian2[y1:y2,x1:x2,:] = cv2.filter2D(img2,-1,kernel,borderType=cv2.BORDER_REPLICATE)

  bbox = [x1, y1, x2, y2]
  print(x1,x2,y1,y2)

  # img_laplacian = np.array(img_laplacian1,dtype=np.uint8) + np.array(img_laplacian2,dtype=np.uint8)
  # img_laplacian = np.array(img_laplacian1[y1:y2,x1:x2,:].copy(),dtype=np.uint8) + np.array(img_laplacian2[y1:y2,x1:x2,:].copy(),dtype=np.uint8)
  img_laplacian = np.array(img_laplacian1[y1:y2,x1:x2,:].copy(),dtype=np.float32) + np.array(img_laplacian2[y1:y2,x1:x2,:].copy(),dtype=np.float32)

  # cv2.imshow('img2',np.array(img2,dtype=np.uint8))
  # cv2.imshow('Blur',np.array(img_blur,dtype=np.uint8))
  # cv2.imshow('Laplacian1',np.array(img_laplacian1,dtype=np.uint8))
  # cv2.imshow('Laplacian2',np.array(img_laplacian2,dtype=np.uint8))
  # cv2.imshow('Laplacian',np.array(img_laplacian,dtype=np.uint8))
  # print(np.max(img_blur),np.max(img_laplacian1),np.max(img_laplacian2))
  # img_blur = img_blur//2
  # img_laplacian1 = img_laplacian1//4
  # img_laplacian2 = img_laplacian2//4
  # img_laplacian2 = np.array(img_laplacian2,dtype=np.uint8)
  # print(np.max(img_blur),np.max(img_laplacian1),np.max(img_laplacian2))
  # img_m = img_blur+img_laplacian1+img_laplacian2
  # print(np.max(img_m))
  # cv2.imshow('original',img_m)

  # print(img_laplacian.shape)

  # vars_to_solve_a = np.zeros((img_laplacian.shape[0]*img_laplacian.shape[1]*img_laplacian.shape[2],img_laplacian.shape[0]*img_laplacian.shape[1]*img_laplacian.shape[2]))
  # vars_to_solve_b = np.zeros((img_laplacian.shape[0]*img_laplacian.shape[1]*img_laplacian.shape[2]))
  # img_laplacian_tmp = np.zeros((img_laplacian.shape[0]+2,img_laplacian.shape[1]+2,img_laplacian.shape[2]))
  # # img_laplacian_tmp[:,:,:] = img[y1-1:y2+1,x1-1:x2+1,:].copy()
  # # img_laplacian_tmp[0,1:-1,:] = img[1,:,:].copy()
  # # img_laplacian_tmp[-1,1:-1,:] = img[-2,:,:].copy()
  # # img_laplacian_tmp[1:-1,0,:] = img[:,1,:].copy()
  # # img_laplacian_tmp[1:-1,-1,:] = img[:,-2,:].copy()
  # img_laplacian_tmp[0,1:-1,:] = img[y1-1,x1:x2,:].copy()
  # img_laplacian_tmp[-1,1:-1,:] = img[y2+1,x1:x2,:].copy()
  # img_laplacian_tmp[1:-1,0,:] = img[y1:y2,x1-1,:].copy()
  # img_laplacian_tmp[1:-1,-1,:] = img[y1:y2,x2+1,:].copy()

  # cv2.imshow('Laplacian tmp',np.array(img_laplacian_tmp,dtype=np.uint8))

  def get_var_num(r, c, channel, s0, s1):
    return (s0*s1)*channel + (r*s1+c)

  s0 = img_laplacian.shape[0]
  s1 = img_laplacian.shape[1]

  img_stitched_arr = []
  # count = 0
  for channel in tqdm.tqdm(range(img.shape[2])):
    vars_to_solve_a_data = []
    vars_to_solve_a_row = []
    vars_to_solve_a_col = []
    vars_to_solve_b = np.zeros((img_laplacian.shape[0]*img_laplacian.shape[1]))
    img_laplacian_tmp = np.zeros((img_laplacian.shape[0]+2,img_laplacian.shape[1]+2,1))
    # img_laplacian_tmp[:,:,:] = img[y1-1:y2+1,x1-1:x2+1,:].copy()
    # img_laplacian_tmp[0,1:-1,:] = img[1,:,:].copy()
    # img_laplacian_tmp[-1,1:-1,:] = img[-2,:,:].copy()
    # img_laplacian_tmp[1:-1,0,:] = img[:,1,:].copy()
    # img_laplacian_tmp[1:-1,-1,:] = img[:,-2,:].copy()
    img_laplacian_tmp[0,1:-1,0] = img[y1-1,x1:x2,channel].copy()
    img_laplacian_tmp[-1,1:-1,0] = img[y2+1,x1:x2,channel].copy()
    img_laplacian_tmp[1:-1,0,0] = img[y1:y2,x1-1,channel].copy()
    img_laplacian_tmp[1:-1,-1,0] = img[y1:y2,x2+1,channel].copy()

    # cv2.imshow('Laplacian tmp {}'.format(channel),np.array(img_laplacian_tmp,dtype=np.uint8))

    l_channel = 0
    count = 0
    for r in tqdm.tqdm(range(1,img_laplacian_tmp.shape[0]-1)):
      for c in tqdm.tqdm(range(1,img_laplacian_tmp.shape[1]-1)):
        R = r-1
        C = c-1
        if r>1 and r<img_laplacian_tmp.shape[0]-2 and c>1 and c<img_laplacian_tmp.shape[1]-2:
          vars_to_solve_b[get_var_num(R,C,l_channel,s0,s1)] = img_laplacian[r-1,c-1,channel]
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R-1,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R-1,C,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R+1,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R+1,C,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C-1,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C-1,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C+1,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C+1,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(-4)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C,l_channel,s0,s1)] = -4
        elif r==1 and c>1 and c<img_laplacian_tmp.shape[1]-2:
          vars_to_solve_b[get_var_num(R,C,l_channel,s0,s1)] = img_laplacian[r-1,c-1,channel]-img_laplacian_tmp[r-1,c,l_channel]
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R+1,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R+1,C,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C-1,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C-1,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C+1,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C+1,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(-4)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C,l_channel,s0,s1)] = -4
        elif r==img_laplacian_tmp.shape[0]-2 and c>1 and c<img_laplacian_tmp.shape[1]-2:
          vars_to_solve_b[get_var_num(R,C,l_channel,s0,s1)] = img_laplacian[r-1,c-1,channel]-img_laplacian_tmp[r+1,c,l_channel]
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R-1,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R-1,C,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C-1,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C-1,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C+1,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C+1,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(-4)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C,l_channel,s0,s1)] = -4
        elif c==1 and r>1 and r<img_laplacian_tmp.shape[0]-2:
          vars_to_solve_b[get_var_num(R,C,l_channel,s0,s1)] = img_laplacian[r-1,c-1,channel]-img_laplacian_tmp[r,c-1,l_channel]
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R-1,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R-1,C,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R+1,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R+1,C,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C+1,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C+1,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(-4)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C,l_channel,s0,s1)] = -4
        elif c==img_laplacian_tmp.shape[1]-2 and r>1 and r<img_laplacian_tmp.shape[0]-2:
          vars_to_solve_b[get_var_num(R,C,l_channel,s0,s1)] = img_laplacian[r-1,c-1,channel]-img_laplacian_tmp[r,c+1,l_channel]
          # print(vars_to_solve_a[count,shape,r+1,c,l_channel,s0,s1,get_var_num(r+1,c,l_channel,s0,s1)))
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R-1,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R-1,C,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R+1,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R+1,C,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C-1,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C-1,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(-4)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C,l_channel,s0,s1)] = -4
        elif r==1 and c==1:
          vars_to_solve_b[get_var_num(R,C,l_channel,s0,s1)] = img_laplacian[r-1,c-1,channel]-img_laplacian_tmp[r-1,c,l_channel]-img_laplacian_tmp[r,c-1,l_channel]
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R+1,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R+1,C,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C+1,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C+1,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(-4)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C,l_channel,s0,s1)] = -4
        elif r==1 and c==img_laplacian_tmp.shape[1]-2:
          vars_to_solve_b[get_var_num(R,C,l_channel,s0,s1)] = img_laplacian[r-1,c-1,channel]-img_laplacian_tmp[r-1,c,l_channel]-img_laplacian_tmp[r,c+1,l_channel]
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R+1,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R+1,C,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C-1,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C-1,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(-4)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C,l_channel,s0,s1)] = -4
        elif r==img_laplacian_tmp.shape[0]-2 and c==1:
          vars_to_solve_b[get_var_num(R,C,l_channel,s0,s1)] = img_laplacian[r-1,c-1,channel]-img_laplacian_tmp[r+1,c,l_channel]-img_laplacian_tmp[r,c-1,l_channel]
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R-1,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R-1,C,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C+1,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C+1,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(-4)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C,l_channel,s0,s1)] = -4
        elif r==img_laplacian_tmp.shape[0]-2 and c==img_laplacian_tmp.shape[1]-2:
          vars_to_solve_b[get_var_num(R,C,l_channel,s0,s1)] = img_laplacian[r-1,c-1,channel]-img_laplacian_tmp[r+1,c,l_channel]-img_laplacian_tmp[r,c+1,l_channel]
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R-1,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R-1,C,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(1)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C-1,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C-1,l_channel,s0,s1)] = 1
          vars_to_solve_a_data.append(-4)
          vars_to_solve_a_row.append(count)
          vars_to_solve_a_col.append(get_var_num(R,C,l_channel,s0,s1))
          # vars_to_solve_a[count,get_var_num(R,C,l_channel,s0,s1)] = -4

        count += 1

    data = np.array(vars_to_solve_a_data)
    row = np.array(vars_to_solve_a_row)
    col = np.array(vars_to_solve_a_col)
    shape = (img_laplacian.shape[0]*img_laplacian.shape[1], img_laplacian.shape[0]*img_laplacian.shape[1])
    print(data.shape,row.shape,col.shape,shape)
    A = csr_matrix((data, (row, col)), shape=shape)
    # print('== A ',A)
    x = spsolve(A, vars_to_solve_b)
    s = img_laplacian.shape[0]*img_laplacian.shape[1]
    img_stitched_c = np.reshape(x,(img_laplacian.shape[0],img_laplacian.shape[1],1))
    # img_stitched_c = np.abs(img_stitched_c)
    img_stitched_c = np.clip(img_stitched_c,0,255)
    # img_stitched_c = np.mod(img_stitched_c,255)
    img_stitched_arr.append(img_stitched_c)

    # img_stitched_1 = np.reshape(x[s:2*s],(img_laplacian.shape[0],img_laplacian.shape[1],1))
    # img_stitched_2 = np.reshape(x[2*s:3*s],(img_laplacian.shape[0],img_laplacian.shape[1],1))
    # img_stitched = np.concatenate((img_stitched_0,img_stitched_1,img_stitched_2),axis=-1)
    # cv2.imshow('Stitched {}'.format(channel),np.array(img_stitched_c,dtype=np.uint8))

    img_l2 = cv2.filter2D(img_stitched_c,-1,kernel,borderType=cv2.BORDER_REPLICATE)
    # cv2.imshow('lap 2 {}'.format(channel),np.array(img_l2,dtype=np.uint8))

  if len(img_stitched_arr)>1:
    img_stitched = np.concatenate(img_stitched_arr,axis=-1)
  else:
    img_stitched = img_stitched_arr[0]
  # img_stitched_c = np.abs(img_stitched_c)
  # img_stitched_c = np.clip(img_stitched_c,0,255)

  # cv2.imshow('Stitched patch',np.array(img_stitched,dtype=np.uint8))

  img_final = img.copy()
  img_final[y1:y2,x1:x2,:] = img_stitched.copy()
  # cv2.imshow('Stitched final',np.array(img_final,dtype=np.uint8))

  # img_diff = img_stitched - img[y1:y2,x1:x2,:]
  # img_diff = np.abs(img_diff)
  # cv2.imshow('diff',np.array(img_diff,dtype=np.uint8))


  return img_final, bbox

# ref: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
def rotateImage(image, angle, pivot=None, scale=1.0):
  print(image.shape,angle,pivot)
  # grab the dimensions of the image and then determine the
  # center
  (h, w) = image.shape[:2]
  (cX, cY) = (w // 2, h // 2)
  # (cX, cY) = (int(pivot[0]), int(pivot[1]))

  # grab the rotation matrix (applying the negative of the
  # angle to rotate clockwise), then grab the sine and cosine
  # (i.e., the rotation components of the matrix)
  M = cv2.getRotationMatrix2D((cX, cY), angle, scale)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])

  # compute the new bounding dimensions of the image
  nW = int((h * sin) + (w * cos))
  nH = int((h * cos) + (w * sin))

  # adjust the rotation matrix to take into account translation
  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY

  # perform the actual rotation and return the image
  return cv2.warpAffine(image, M, (nW, nH))

def augment_crack(crack, phone_shape, rotate=True, resize=True):
  # crack[:,:2,:] = 0
  # crack[:,-2:,:] = 0
  # crack[:2,:,:] = 0
  # crack[-2:,:,:] = 0
  if rotate:
    angle = 360*np.random.random()
    crack = rotateImage(crack,angle)      
  if resize:
    max_height = phone_shape[0]
    max_width = phone_shape[1]
    margin_x = np.ceil(max_width*0.1)
    margin_y = np.ceil(max_height*0.1)
    w = int((0.40 + 0.30*np.random.random())*max_width)
    # h = int((0.10 + 0.35*np.random.random())*max_height)
    h = w
    crack = cv2.resize(crack,(w,h))

  return crack




# img_diff = img_stitched - img[y1:y2,x1:x2,:]
# img_diff = np.abs(img_diff)
# cv2.imshow('diff',np.array(img_diff,dtype=np.uint8))

# img_stitched = stitch_image(img,img2,loc_to_stitch='top')
# cv2.imshow('Stitched final',img_stitched)
# cv2.waitKey(0)

crack_files = []
for crack_file in os.listdir(crack_dir):
  if ".DS_Store" in crack_file:
    continue
  p1 = os.path.join(crack_dir,crack_file)
  crack_files.append(p1)


phones = []
for file in os.listdir(phone_dir):
  if ".DS_Store" in file:
    continue
  p1 = os.path.join(phone_dir,file)
  phones.append(p1)

np.random.shuffle(phones)
phone_count = 0
for crack_file in crack_files:
  for i in tqdm.tqdm(range(NUM_PHONES_PER_CRACK)):
    # print(crack_file)
    # cur_phone_file = phones[int(np.random.random()*len(phones))]
    cur_phone_file = phones[phone_count%len(phones)]

    phone = cv2.imread(cur_phone_file)
    if RESIZE_PHONE_SIZE:
      rh = RESIZE_PHONE_SIZE
      rw = int(phone.shape[1]*1.0/phone.shape[0]*rh)
      phone = cv2.resize(phone,(rw,rh))
    crack = cv2.imread(crack_file)
    phone_shape = phone.shape
    crack = augment_crack(crack, phone_shape, rotate=ROTATE_CRACK, resize=RESIZE_CRACK)

    if crack.shape[0]==0 or crack.shape[1]==0:
      continue
    img_stitched, bbox = stitch_image(phone,crack)

    crack_file_name = crack_file.split('/')[-1].split('.')[0]
    out_path = os.path.join(out_dir,'{}_{}.jpg'.format(crack_file_name,i))
    cv2.imwrite(out_path,img_stitched)

    final_file_name = '{}_{}'.format(crack_file_name,i)
    width = img_stitched.shape[1]
    height = img_stitched.shape[0]
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
    out_file.write("{},{},{},{},{},{},{},{}\n".format(final_file_name,width,height,CLASS,xmin,ymin,xmax,ymax))
    out_file.flush()
    phone_count += 1

out_file.close()
