import os
import numpy as np

def obj_read(obj_path):
    # read obj file and get vertices and faces
    with open(obj_path) as file:
        vertices = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                vertices.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break
            if strs[0] == "f":
                faces.append(
                    (int(strs[1].split('//')[0]), int(strs[2].split('//')[0]), int(strs[3].split('//')[0])))
    vertices = np.array(vertices)  # in matrix form
    faces = np.array(faces)
    return vertices, faces


def obj_write(obj_path, vertices, faces):
    with open(obj_path, "w") as file:
        file.write("# " + str(vertices.shape[0]) + " vertices, " + str(faces.shape[0]) + " faces" + "\n")
        # write vertices
        for i in range(vertices.shape[0]):
            file.write("v ")
            file.write(str(float("{0:.6g}".format(vertices[i, 0]))) + " ")
            file.write(str(float("{0:.6g}".format(vertices[i, 1]))) + " ")
            file.write(str(float("{0:.6g}".format(vertices[i, 2]))) + " ")
            file.write("\n")
        # write faces
        for i in range(faces.shape[0]):
            file.write("f ")
            file.write(str(int(faces[i, 0])) + " ")
            file.write(str(int(faces[i, 1])) + " ")
            file.write(str(int(faces[i, 2])) + " ")
            file.write("\n")

'''
# simple
# objFilePath = 'D:\\XiaohanYuan\\registration\\registration_code\\vs\\template_rigid003_010_LV.obj'
objFilePath ="LV.obj"
with open(objFilePath) as file:
    points = []
    faces = []
    while 1:
        line = file.readline()
        if not line:
            break
        strs = line.split(" ")
        if strs[0] == "v":
            points.append((float(strs[1]), float(strs[2]), float(strs[3])))
        if strs[0] == "vt":
            break
        if strs[0] == "f":
            faces.append((float(strs[1].split('//')[0]), float(strs[2].split('//')[0]), float(strs[3].split('//')[0])))

# points原本为列表，需要转变为矩阵，方便处理
points = np.array(points)
faces = np.array(faces)
# print(vertices.shape)
print(faces)
'''

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads m Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = map(float, values[1:4])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = map(float, values[1:4])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))


