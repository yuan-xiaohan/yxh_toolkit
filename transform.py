from obj_process import *


obj_path = [
            r"Z:\XiaohanYuan\obj_rigid\0001312654_010_LV.obj",
            r"Z:\XiaohanYuan\obj_rigid\0001312654_020_LV.obj",
            ]



# obj_path = [
#             r"Z:\XiaohanYuan\obj_rigid\0008101542_010_LV.obj",
#             r"Z:\XiaohanYuan\obj_rigid\0008101542_020_LV.obj",
#             r"Z:\XiaohanYuan\obj_rigid\0008101542_030_LV.obj",
#             r"Z:\XiaohanYuan\obj_rigid\0008101542_040_LV.obj",
#             r"Z:\XiaohanYuan\obj_rigid\0008101542_050_LV.obj",
#             r"Z:\XiaohanYuan\obj_rigid\0008101542_060_LV.obj",
#             r"Z:\XiaohanYuan\obj_rigid\0008101542_070_LV.obj",
#             r"Z:\XiaohanYuan\obj_rigid\0008101542_080_LV.obj",
#             r"Z:\XiaohanYuan\obj_rigid\0008101542_090_LV.obj",
#             r"Z:\XiaohanYuan\obj_rigid\0008101542_100_LV.obj",
#             ]

RT_path = r"Z:\XiaohanYuan\obj_rigid\RT\0001312654_040_LV.aln"
# get matrix RT
RT = np.loadtxt(RT_path, dtype=np.float32)
print(RT.shape)


# get obj
for path in obj_path:
    vertices, faces = obj_read(path)

    vertices_T = vertices.T

    vertices_T = np.row_stack((vertices_T, np.ones([1, vertices.shape[0]])))


    vertices_T_new = np.dot(RT, vertices_T)
    vertices_new = vertices_T_new.T[:, :-1]
    # print(vertices_new.shape)

    obj_write(path, vertices_new, faces)


