cd taichi_render_gpu

python render_smpl.py --dataroot ../data/demo \
    --obj_path ../data/demo/smplx \
    --faces_path ../lib/data/smplx_multi.obj --yaw_list 0 90 180 270

