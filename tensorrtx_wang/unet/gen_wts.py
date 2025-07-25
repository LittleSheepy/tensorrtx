import torch
import sys
import struct

def main(pth_path, wts_path):
  device = torch.device('cpu')
  state_dict = torch.load(pth_path, map_location=device)

  f = open(wts_path, 'w')
  f.write("{}\n".format(len(state_dict.keys())))
  for k, v in state_dict.items():
    print('key: ', k)
    print('value: ', v.shape)
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))
    for vv in vr:
      f.write(" ")
      f.write(struct.pack(">f", float(vv)).hex())
    f.write("\n")
  f.close()

if __name__ == '__main__':
  pth_path = r"D:\03GitHub\00myGitHub\tensorrtx\tensorrtx_wang\unet/unet_carvana_scale1.0_epoch2.pth"
  wts_path = r"D:\03GitHub\00myGitHub\tensorrtx\tensorrtx_wang\unet/unet_carvana_scale1.0_epoch2.wts"
  main(pth_path, wts_path)

