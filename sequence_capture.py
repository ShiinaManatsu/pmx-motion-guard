"""Main-thread extraction for the fixed-pack skeletal solver prototype."""
import time

import bpy
import numpy as np

from .engine import matrices, weights_and_groups
from .mapping import COORDINATES as C, PARENTS
from .vendor.retarget_guard import build_character


def basis_vertices(obj, use_shape_keys):
    """Sample the current shape-key mix in mesh space before modifiers."""
    values = np.empty(len(obj.data.vertices) * 3, np.float32)
    if not use_shape_keys or obj.data.shape_keys is None:
        obj.data.vertices.foreach_get('co', values)
        return values.reshape(-1, 3)
    temporary = obj.copy()
    temporary.name = 'Guard shape sampling'
    temporary.modifiers.clear()
    bpy.context.scene.collection.objects.link(temporary)
    evaluated = None
    try:
        graph = bpy.context.evaluated_depsgraph_get()
        evaluated = temporary.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        if len(mesh.vertices) != len(obj.data.vertices):
            raise ValueError('形态键基础形状的顶点数量发生变化')
        mesh.vertices.foreach_get('co', values)
        return values.reshape(-1, 3)
    finally:
        if evaluated is not None:
            evaluated.to_mesh_clear()
        bpy.data.objects.remove(temporary, do_unlink=True)


def capture_steps(obj, mapping, slots, start, end, use_shape_keys=False):
    """Capture rotations once and measure the reduced rig's FK mismatch.

    This validation gate is necessary because PMX independent lower/upper
    body translations and helper chains can violate the 22-joint FK model.
    No Blender object or pointer is returned to numerical worker threads.
    """
    scene = bpy.context.scene
    original_frame = scene.frame_current
    arm = obj.find_armature()
    original_pose = arm.data.pose_position
    weights, _ = weights_and_groups(obj, mapping)
    space = obj.matrix_world.inverted() @ arm.matrix_world
    rest_matrices = []
    for role, name in mapping.items():
        if name:
            matrix = space @ arm.data.bones[name].matrix_local
        else:
            matrix = space @ arm.data.bones[mapping['spine2']].matrix_local
            neck = space @ arm.data.bones[mapping['neck']].matrix_local
            matrix.translation = matrix.translation.lerp(neck.translation, .5)
        rest_matrices.append(matrix)
    # Always follow semantic role order, independent of dictionary insertion.
    from .mapping import ROLES
    rest_matrices = dict(zip(mapping, rest_matrices))
    rest_matrices = [rest_matrices[role] for role in ROLES]
    joints = np.array([list(m.translation) for m in rest_matrices]) @ C.T
    basis = C @ np.array([np.array(m.to_quaternion().to_matrix()) for m in rest_matrices])
    obj.data.calc_loop_triangles()
    faces = np.array([list(t.vertices) for t in obj.data.loop_triangles if t.material_index in slots], np.int32)
    if not len(faces):
        raise ValueError('No selected collision surfaces')
    ids = np.unique(faces)
    remap = np.full(len(weights), -1)
    remap[ids] = np.arange(len(ids))
    vertices = basis_vertices(obj, use_shape_keys) @ C.T
    pack_started = time.perf_counter()
    character = build_character(vertices[ids], remap[faces], weights[ids], PARENTS,
                                joints, np.arange(22), rest_basis=basis)
    pack_seconds = time.perf_counter() - pack_started
    poses, positions = [], []
    capture_started = time.perf_counter()
    try:
        for frame in range(start, end + 1):
            scene.frame_set(frame)
            current = matrices(obj, mapping)
            world = C @ np.array([np.array(m.to_quaternion().to_matrix()) for m in current])
            pose = np.empty_like(world)
            for joint, parent in enumerate(PARENTS):
                local = world[joint] if parent < 0 else world[parent].T @ world[joint]
                pose[joint] = character.data['local_basis'][joint].T @ local
            poses.append(pose)
            positions.append(np.array([list(m.translation) for m in current]) @ C.T)
            yield frame
    finally:
        scene.frame_set(original_frame)
        arm.data.pose_position = original_pose
    poses, positions = np.asarray(poses), np.asarray(positions)
    maximum = 0.
    squared = 0.
    for pose, actual in zip(poses, positions):
        world = np.empty_like(pose)
        predicted = np.empty_like(joints)
        for j, parent in enumerate(PARENTS):
            local = character.data['local_basis'][j] @ pose[j]
            if parent < 0:
                world[j] = local
                predicted[j] = character.data['local_offset'][j]
            else:
                world[j] = world[parent] @ local
                predicted[j] = predicted[parent] + world[parent] @ character.data['local_offset'][j]
        # Global translation is irrelevant to self collision and can be copied.
        predicted += actual[0] - predicted[0]
        error = np.linalg.norm(predicted - actual, axis=1)
        maximum = max(maximum, float(error.max()))
        squared += float(np.sum(error * error))
    report = dict(pack_seconds=pack_seconds, capture_seconds=time.perf_counter() - capture_started,
                  frames=len(poses), fk_max_error_mm=maximum * 1000,
                  fk_rms_error_mm=(squared / (len(poses) * 22)) ** .5 * 1000)
    return character, poses, positions, report


def capture_sequence(obj, mapping, slots, start, end):
    iterator = capture_steps(obj, mapping, slots, start, end)
    while True:
        try:
            next(iterator)
        except StopIteration as finished:
            return finished.value
