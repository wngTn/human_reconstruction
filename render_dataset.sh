cd taichi_render_gpu
# python render_multi.py --data_root ../assets/obj --texture_root ../assets/tex --save_path ../dataset/example
# python render_smpl.py --dataroot ../dataset/example --obj_path ../assets/smplx --faces_path ../lib/data/smplx_fine.obj





python render_multi.py --data_root ../dataset/MultiHuman/single/obj --texture_root ../../MHDataset/MultiHumanDataset/multihuman_single_raw/multihuman_single --save_path ../dataset/mh_single
python render_smpl.py --dataroot ../dataset/mh_single_6view --obj_path ../dataset/MultiHuman/single/smplx --faces_path ../lib/data/smplx_fine.obj --yaw_list 0 60 120 180 240 300
# 0 90 180 270
