# 机器A
export MASTER_ADDR=11.204.14.164
export MASTER_PORT=29500
export RANK=0 WORLD_SIZE=2
python test_scr.py
# 机器B
# export MASTER_ADDR=<机器A的IP>
# export MASTER_PORT=29500
# export RANK=1 WORLD_SIZE=2
# python test_script.py