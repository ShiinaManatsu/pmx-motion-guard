"""Numerical worker entry point. Runs with ordinary Python and NumPy."""
import importlib
import json
import os
from pathlib import Path
import sys
import types
import struct

import numpy as np


def solve_surfaces(directory, data, meta, api, coordinate, parents):
    from motion_guard_worker.vendor.retarget_guard.streaming import frozen_settings
    from motion_guard_worker.vendor.retarget_guard.rotations import matrix_to_axis_angle, as_matrices
    from motion_guard_worker.vendor.retarget_guard import vertex_normals
    vertices = np.load(directory/'vertices.npy',mmap_mode='r')
    bases = np.load(directory/'bases.npy',mmap_mode='r')
    with np.load(directory/'surface.npz',allow_pickle=False) as packed:
        weights, faces = packed['weights'], packed['faces']
    ids = np.unique(faces)
    remap = np.full(vertices.shape[1],-1)
    remap[ids] = np.arange(len(ids))
    faces = remap[faces]
    identity = np.tile(np.eye(3),(22,1,1))
    enabled = np.r_[True,data['enabled']]
    previous = None
    transforms = []
    temporary = directory/'motion.pc2.partial'
    with temporary.open('wb') as stream:
        stream.write(struct.pack('<12siiffi',b'POINTCACHE2\0',1,vertices.shape[1],float(meta['start']),1.,len(vertices)))
        for index, vertices_frame in enumerate(vertices):
            source = vertices_frame @ coordinate.T
            character = api.build_character(source[ids],faces,weights[ids],parents,
                data['positions'][index],np.arange(22),rest_basis=bases[index])
            with api.RetargetGuard(character,character,fps=meta['fps'],
                    settings=frozen_settings(source_allow=0,rho=meta['compression'])) as guard:
                import ctypes
                mask = np.ascontiguousarray(data['enabled'],dtype=np.int32)
                native = guard._guard
                native.lib.guard_set_mask.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
                native.lib.guard_set_mask.restype = None
                native.lib.guard_set_mask(native.handle,mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
                pose = guard.step(identity) if meta['mode']=='BOTH' else identity
                angles = matrix_to_axis_angle(pose)
                angles[~enabled] = 0
                if previous is not None:
                    angles = previous + meta['smoothing']*(angles-previous)
                    angles[~enabled] = 0
                previous = angles
                pose, _ = as_matrices(angles,22)
                pack = character.data
                world, joints = np.empty_like(pose), np.empty((22,3))
                for j,parent in enumerate(parents):
                    local = pack['local_basis'][j] @ pose[j]
                    if parent<0:
                        world[j],joints[j] = local,pack['local_offset'][j]
                    else:
                        world[j] = world[parent] @ local
                        joints[j] = joints[parent]+world[parent] @ pack['local_offset'][j]
                rotation = world @ pack['rest_basis'].swapaxes(-1,-2)
                translation = joints-(rotation @ pack['rest_joints'][...,None])[...,0]
                output = np.zeros_like(source)
                for j in range(22):
                    output += (source @ rotation[j].T+translation[j])*weights[:,j,None]
                output[ids] = guard.deform(output[ids],vertex_normals(output[ids],faces))
                stream.write(np.asarray(output @ coordinate,dtype='<f4').tobytes())
                transform = np.tile(np.eye(4),(22,1,1))
                transform[:,:3,:3] = coordinate.T @ rotation @ coordinate
                transform[:,:3,3] = translation @ coordinate
                transforms.append(transform)
            progress = directory/'progress.tmp.json'
            progress.write_text(json.dumps(dict(completed=index+1)),encoding='utf-8')
            try:
                progress.replace(directory/'progress.json')
            except PermissionError:
                pass
    temporary.replace(directory/'motion.pc2')
    return np.array(transforms)


def quaternion_channels(matrices):
    if not np.all(np.isfinite(matrices)):
        raise ValueError('动画矩阵包含无效数值')
    scale = np.linalg.norm(matrices[..., :3, :3], axis=-2)
    if np.any(scale < 1e-9):
        raise ValueError('骨骼矩阵包含零缩放')
    rotation = matrices[..., :3, :3] / scale[..., None, :]
    # Signed scale keeps reflected transforms representable.
    sign = np.where(np.linalg.det(rotation) < 0, -1., 1.)
    scale[..., 0] *= sign
    rotation[..., :, 0] *= sign[..., None]
    if np.max(np.abs(rotation.swapaxes(-1, -2) @ rotation - np.eye(3))) > 1e-4:
        raise ValueError('骨骼局部矩阵包含剪切，请检查骨架缩放')
    flat = rotation.reshape(-1, 3, 3)
    candidates = np.stack((1 + np.trace(flat, axis1=1, axis2=2),
                           1+flat[:,0,0]-flat[:,1,1]-flat[:,2,2],
                           1-flat[:,0,0]+flat[:,1,1]-flat[:,2,2],
                           1-flat[:,0,0]-flat[:,1,1]+flat[:,2,2]), axis=1)
    choices = candidates.argmax(axis=1)
    q = np.empty((len(flat), 4))
    for k in range(4):
        selected = choices == k
        m = flat[selected]
        v = np.sqrt(np.maximum(candidates[selected, k], 0)) * .5
        d = np.maximum(4*v, 1e-12)
        if k == 0:
            values = [v, (m[:,2,1]-m[:,1,2])/d, (m[:,0,2]-m[:,2,0])/d, (m[:,1,0]-m[:,0,1])/d]
        elif k == 1:
            values = [(m[:,2,1]-m[:,1,2])/d, v, (m[:,0,1]+m[:,1,0])/d, (m[:,0,2]+m[:,2,0])/d]
        elif k == 2:
            values = [(m[:,0,2]-m[:,2,0])/d, (m[:,0,1]+m[:,1,0])/d, v, (m[:,1,2]+m[:,2,1])/d]
        else:
            values = [(m[:,1,0]-m[:,0,1])/d, (m[:,0,2]+m[:,2,0])/d, (m[:,1,2]+m[:,2,1])/d, v]
        q[selected] = np.stack(values, axis=1)
    q = q.reshape(matrices.shape[:-2] + (4,))
    for frame in range(1, len(q)):
        flip = np.sum(q[frame]*q[frame-1], axis=-1) < 0
        q[frame, flip] *= -1
    return matrices[..., :3, 3], q, scale


def run(directory):
    directory = Path(directory)
    root = Path(__file__).resolve().parent
    package = types.ModuleType('motion_guard_worker')
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    solver = importlib.import_module(package.__name__ + '.sequence_solver')
    api = importlib.import_module(package.__name__ + '.vendor.retarget_guard')
    coordinate = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    parents = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
    meta = json.loads((directory/'input.json').read_text(encoding='utf-8'))
    os.environ['RETARGET_GUARD_CACHE'] = meta['native']
    with np.load(directory/'input.npz', allow_pickle=False) as z:
        data = dict(z)
    character = api.Character.load(directory/'character.npz')
    surface = meta.get('mode','JOINTS') != 'JOINTS'
    if surface:
        transforms = solve_surfaces(directory,data,meta,api,coordinate,parents)
        report = dict(frames=len(data['poses']),mode=meta['mode'])
    else:
        solved, report = solver.solve_sequence(character, data['poses'], fps=meta['fps'],
                            enabled=data['enabled'], screen=False, lookahead=5)
    pack = character.data
    def forward(poses):
        world = np.empty_like(poses)
        joints = np.empty(poses.shape[:-2]+(3,))
        for j, parent in enumerate(parents):
            local = pack['local_basis'][j] @ poses[:, j]
            if parent < 0:
                world[:, j] = local
                joints[:, j] = pack['local_offset'][j]
            else:
                world[:, j] = world[:, parent] @ local
                joints[:, j] = joints[:, parent] + (world[:, parent] @ pack['local_offset'][j,:,None])[...,0]
        return world, joints
    if not surface:
        world0, joints0 = forward(data['poses'])
        world1, joints1 = forward(solved)
        rotations = world1 @ world0.swapaxes(-1,-2)
        actual = data['positions']
        translations = actual+joints1-joints0-(rotations @ actual[...,None])[...,0]
        transforms = np.broadcast_to(np.eye(4), rotations.shape[:-2]+(4,4)).copy()
        transforms[..., :3,:3] = coordinate.T @ rotations @ coordinate
        transforms[..., :3,3] = translations @ coordinate
    transforms = data['object_to_arm'][:,None] @ transforms @ np.linalg.inv(data['object_to_arm'])[:,None]
    corrected = transforms[:,data['groups']] @ data['bone_world']
    local = np.empty_like(corrected)
    for j, parent in enumerate(data['parents']):
        if parent < 0:
            local[:,j] = np.linalg.inv(data['rest'][j]) @ corrected[:,j]
        else:
            rest_local = np.linalg.inv(data['rest'][parent]) @ data['rest'][j]
            local[:,j] = np.linalg.inv(rest_local) @ np.linalg.inv(corrected[:,parent]) @ corrected[:,j]
    location, quaternion, scale = quaternion_channels(local)
    temporary = directory/'result.tmp.npz'
    np.savez(temporary, location=location.astype(np.float32), quaternion=quaternion.astype(np.float32),
             scale=scale.astype(np.float32), corrected_world=corrected.astype(np.float32))
    temporary.replace(directory/'result.npz')
    (directory/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    run(sys.argv[1])
