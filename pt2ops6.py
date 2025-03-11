"""
程序说明：2024-11-20：本程序在pt2ops4的基础上进行了改进，
包括将跨数，每跨长度，层数、每层高度等设为变量，方便更改
本模型是基于--古泉的教材建立的，与ETABS模型（3层框架2）结果一致
"""

from openseespy.opensees import *
import numpy as np
import matplotlib.pyplot as plt
import opsvis

# 定义单位
mm = 1
m = 1000 * mm
N = 1
kN = 1000 * N

# 移除现有模型
wipe()

# 设置模型构建器
model('basic', '-ndm', 3, '-ndf', 6)

# 定义框架尺寸
floor_z = np.ones(3) * [3.6576 * m]  # 层高
bx = np.ones(1) * [6.096 * m]  # 框架宽
by = np.ones(1) * [6.096 * m]  # 框架长

# 计算累加数列
floor_z = np.cumsum(floor_z)
bx = np.cumsum(bx)
by = np.cumsum(by)

# 在数列的前面插入一个0
floor_z = np.insert(floor_z, 0, 0)
bx = np.insert(bx, 0, 0)
by = np.insert(by, 0, 0)

# 生成节点坐标
node_coor = []
for i in floor_z:
    for j in by:
        for k in bx:
            node_coor.append([k, j, i])

# 创建nodes
node_tags = []
for i, (x, y, z) in enumerate(node_coor):
    node(i + 1, x, y, z)
    node_tags.append(i + 1)

# 定义边界条件:找出z坐标为0的点的索引,将其自由度固定
z_zero_indices = [node_tag for index, node_tag in enumerate(node_tags) if node_coor[node_tag - 1][2] == 0]

for i in z_zero_indices:
    fix(i, 1, 1, 1, 1, 1, 1)

# 定义梁和柱的截面尺寸
beam_sections = [(200 * mm, 500 * mm),
                 (200 * mm, 450 * mm),
                 (200 * mm, 500 * mm),
                 (200 * mm, 550 * mm),
                 (250 * mm, 400 * mm),
                 (250 * mm, 450 * mm)]
column_sections = [(500 * mm, 500 * mm),
                   (450 * mm, 450 * mm),
                   (500 * mm, 500 * mm),
                   (550 * mm, 550 * mm),
                   (600 * mm, 600 * mm)]

# 定义材料
# uniaxialMaterial('Concrete01', matTag, fpc, epsc0, fpcu, epsU)
uniaxialMaterial('Concrete01', 1, -34473.8, -0.005, -24131.66, -0.02)
uniaxialMaterial('Concrete01', 2, -27579.04, -0.002, 0.0, -0.006)
# uniaxialMaterial('Steel01', matTag,      Fy,     E0,    b, a1, a2, a3, a4)
uniaxialMaterial('Steel01', 3, 248200., 2.1e8, 0.02)
# uniaxialMaterial('Elastic', matTag, E, eta=0.0, Eneg=E)
uniaxialMaterial('Elastic', 10, 68947600000000)

# 定义截面
beam_sec_tags = []
# section('Elastic', secTag, E_mod,        A,         Iz,       Iy,         G_mod, Jxx, alphaY=None, alphaZ=None)
for i, (b, h) in enumerate(beam_sections):
    section('Elastic', i + 1, 30000 * N / mm ** 2, b * h * mm ** 2, b * h ** 3 / 12, b ** 3 * h / 12,
            30000 * N / mm ** 2 / (2 * (1 + 0.2)), b * h ** 2 / 6)
    beam_sec_tags.append(i + 1)

column_sec_tags = []
for i, (b, h) in enumerate(column_sections):
    section('Elastic', i + len(beam_sections) + 1, 30000 * N / mm ** 2, b * h * mm ** 2, b * h ** 3 / 12,
            b ** 3 * h / 12, 30000 * N / mm ** 2 / (2 * (1 + 0.2)), b * h ** 2 / 6)
    column_sec_tags.append(i + len(beam_sections) + 1)

# 定义几何变换
geomTransf('Linear', 1, 1, 0, 0)

# 定义几何变换
geomTransf('Linear', 2, 1, 1, 0)

# 定义梁柱单元
column_ele_tags = []
column_ele_length = []
# element('elasticBeamColumn', eleTag, *eleNodes, secTag, transfTag, <'-mass', mass>, <'-cMass'> <'-releasez', releaseCode>, <'-releasey', releaseCode>)
for i in range(len(node_tags) - len(z_zero_indices)):
    element('elasticBeamColumn', i + 1, i + 1, i + 1 + len(z_zero_indices), column_sec_tags[0], 1)
    column_ele_tags.append(i + 1)
    # length = sqrt(node_coor[i][0]-node_coor[i + len(z_zero_indices)][0])
    
    # column_ele_length.append(node_coor[i]-node_coor[i + len(z_zero_indices)])
    
beam_ele_tags = []
length_beam_ele_tags = []
index = len(column_ele_tags)
for i in range(len(floor_z))[1::]:
    for j in range(len(by)):
        for k in range(len(bx))[1::]:
            index = index + 1
            nodei = i * len(z_zero_indices) + len(bx) * j + k
            nodej = i * len(z_zero_indices) + len(bx) * j + k + 1
            element('elasticBeamColumn', index, nodei, nodej, beam_sec_tags[0], 2)
            beam_ele_tags.append(index)

for i in range(len(floor_z))[1::]:
    for j in range(len(by))[1::]:
        for k in range(len(bx)):
            index = index + 1
            nodei = i * len(z_zero_indices) + (k + 1) + (j - 1) * len(bx)
            nodej = i * len(z_zero_indices) + (k + 1) + (j - 1) * len(bx) + len(bx)
            element('elasticBeamColumn', index, nodei, nodej, beam_sec_tags[0], 2)
            beam_ele_tags.append(index)

# 定义刚性楼板：先定义主节点，后将每层的节点链接到该主节点上
index = len(node_tags)
# ndof = [1, 1, 1, 0, 0, 0]
node_rigidDiaphragm_tags = []
for i in range(len(floor_z) - 1):
    node(index + i + 1, np.mean(bx), np.mean(by), floor_z[i + 1])
    fix(index + i + 1, 0, 0, 1, 1, 1, 0)
    # fix(index + i + 1, 1, 1, 1, 1, 1, 1)
    start_index = ((i + 1) * len(bx) * len(by) + 1)
    end_index = start_index + len(bx) * len(by) - 1
    print(index + i + 1)
    if end_index <= len(node_tags):
        rigidDiaphragm(3, index + i + 1, *node_tags[start_index:end_index])
    else:
        print(f"Index out of range for rigidDiaphragm at floor {i + 1}")
    node_rigidDiaphragm_tags.append(index + i + 1)

opsvis.plot_model()

# 创建荷载模式
timeSeries('Linear', 1)
pattern('Plain', 1, 1)

# # Create the nodal load - command: load nodeID xForce yForce
# for i in range(len(node_rigidDiaphragm_tags)):
#     load(node_rigidDiaphragm_tags[i], 10, 0.0, 0.0, 0.0, 0.0, 0.0)
# load(node_rigidDiaphragm_tags[-1], 0, 10, 0, 0, 0, 0)
# load(node_rigidDiaphragm_tags[-2], 0, 10, 0, 0, 0, 0)
load(13, 0, 10*kN, 0, 0, 0, 0)
load(14, 0, 10*kN, 0, 0, 0, 0)

# 可视化模型荷载
opsvis.plot_load()

# 分析
system('BandSPD')
numberer('RCM')
constraints('Transformation')
integrator('LoadControl', 1.0)
algorithm('Newton')
analysis('Static')
result = analyze(1)

if result < 0:
    print("Analysis failed with error code:", result)
else:
    print("Analysis successful")

# 获取位移
# print(node_rigidDiaphragm_tags)
# disp_x = nodeDisp(node_rigidDiaphragm_tags[-1], 1)
# disp_y = nodeDisp(node_rigidDiaphragm_tags[-1], 2)
disp_x = nodeDisp(node_rigidDiaphragm_tags[-1], 1)
disp_y = nodeDisp(node_rigidDiaphragm_tags[-1], 2)

print(f"Displacement at top node in x-direction: {disp_x} mm")
print(f"Displacement at top node in y-direction: {disp_y} mm")

# 计算梁和柱的体积
total_beam_volume = sum([beam_sections[i // 4][0] * beam_sections[i // 4][1] * h for i in range(4)]) * len(
    beam_sections)
total_column_volume = sum([column_sections[i // 4][0] * column_sections[i // 4][1] * h for i in range(4)]) * len(
    column_sections)

print(f"Total beam volume: {total_beam_volume / mm ** 3} mm^3")
print(f"Total column volume: {total_column_volume / mm ** 3} mm^3")
opsvis.plot_defo()
plt.show()
# exit()
