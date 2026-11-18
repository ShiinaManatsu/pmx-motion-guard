"""Evaluated PMX correction, sampled geometry caches and visual skeleton baking."""
import hashlib
import json
from pathlib import Path
import struct
import time
import uuid
import ctypes

import bpy
from mathutils import Matrix
import numpy as np

from .mapping import COORDINATES as C, PARENTS, ROLES
from .vendor.retarget_guard import RetargetGuard, build_character, vertex_normals
from .vendor.retarget_guard.streaming import frozen_settings


def matrices(obj, mapping):
    arm = obj.find_armature()
    space = obj.matrix_world.inverted() @ arm.matrix_world
    result = []
    for role in ROLES:
        name = mapping[role]
        if name:
            matrix = space @ arm.pose.bones[name].matrix
        else:
            matrix = space @ arm.pose.bones[mapping["spine2"]].matrix
            end = space @ arm.pose.bones[mapping["neck"]].matrix
            matrix.translation = matrix.translation.lerp(end.translation, .5)
        result.append(matrix)
    return result


def weights_and_groups(obj, mapping):
    arm = obj.find_armature()
    semantic = {name: i for i, name in enumerate(mapping.values()) if name}
    for side in ("L", "R"):
        for base, role in (("足", "hip"), ("ひざ", "knee"), ("足首", "ankle"), ("つま先", "foot")):
            if base + "." + side in arm.data.bones:
                semantic[base + "." + side] = ROLES.index(side.lower() + "_" + role)
    bone_groups = {}
    for bone in arm.data.bones:
        ancestor = bone
        while ancestor is not None and ancestor.name not in semantic:
            ancestor = ancestor.parent
        bone_groups[bone.name] = semantic[ancestor.name] if ancestor else 0
    group_map = {g.index: bone_groups[g.name] for g in obj.vertex_groups if g.name in bone_groups}
    weights = np.zeros((len(obj.data.vertices), 22))
    for vertex in obj.data.vertices:
        for group in vertex.groups:
            if group.group in group_map:
                weights[vertex.index, group_map[group.group]] += group.weight
    weights[weights.sum(1) == 0, 0] = 1
    weights /= weights.sum(1, keepdims=True)
    return weights, bone_groups


def sample_mesh(obj):
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh()
    try:
        values = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get("co", values)
        mesh.calc_loop_triangles()
        faces = np.array([list(t.vertices) for t in mesh.loop_triangles], np.int32)
        material = np.array([t.material_index for t in mesh.loop_triangles])
        return values.reshape(-1, 3).astype(float), faces, material
    finally:
        evaluated.to_mesh_clear()


def enabled_indices(settings):
    indices = []
    for enabled, group in ((settings.left_arm, [13, 16, 18, 20]),
                           (settings.right_arm, [14, 17, 19, 21]),
                           (settings.left_leg, [1, 4, 7, 10]),
                           (settings.right_leg, [2, 5, 8, 11]),
                           (settings.spine, [3, 6, 9]), (settings.neck, [12, 15])):
        if enabled:
            indices.extend(group)
    return indices


def corrected_frame(obj, mapping, body_slots, weights, settings, directory, previous=None):
    vertices, triangles, materials = sample_mesh(obj)
    if len(vertices) != len(weights):
        raise ValueError("当前修改器改变了顶点数量，请先固定拓扑")
    faces = triangles[np.isin(materials, body_slots)]
    if not len(faces):
        raise ValueError("请选择身体碰撞材质")
    ids = np.unique(faces)
    remap = np.full(len(vertices), -1)
    remap[ids] = np.arange(len(ids))
    faces = remap[faces]
    source = vertices @ C.T
    bone_matrices = matrices(obj, mapping)
    joints = np.array([list(m.translation) for m in bone_matrices]) @ C.T
    basis = C @ np.array([np.array(m.to_quaternion().to_matrix()) for m in bone_matrices])
    # A pose-local character pack retains arbitrary evaluated PMX deformation.
    # Identical evaluated frames reuse their numerical pack across runs.
    digest = hashlib.sha256()
    for array in (source[ids], faces, weights[ids], joints, basis):
        digest.update(np.ascontiguousarray(array).tobytes())
    pack_path = Path(directory) / "packs" / (digest.hexdigest() + ".npz")
    from .vendor.retarget_guard import Character
    if pack_path.exists():
        character = Character.load(pack_path)
    else:
        character = build_character(source[ids], faces, weights[ids], PARENTS, joints,
                                    np.arange(22), rest_basis=basis)
        pack_path.parent.mkdir(parents=True, exist_ok=True)
        temp = pack_path.with_suffix(".tmp")
        character.save(temp)
        temp.replace(pack_path)
    from .vendor.retarget_guard.rotations import matrix_to_axis_angle, as_matrices
    identity = np.tile(np.eye(3), (22, 1, 1))
    with RetargetGuard(character, character, fps=bpy.context.scene.render.fps / bpy.context.scene.render.fps_base,
                       settings=frozen_settings(source_allow=0, rho=settings.compression)) as guard:
        mask = np.zeros(22, bool)
        mask[enabled_indices(settings)] = True
        enabled = np.ascontiguousarray(mask[1:], dtype=np.int32)
        native = guard._guard
        native.lib.guard_set_mask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        native.lib.guard_set_mask.restype = None
        native.lib.guard_set_mask(native.handle, enabled.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        pose = guard.step(identity) if settings.mode != "SURFACE" else identity
        # Project the solution onto the explicitly selected output joints.
        angles = matrix_to_axis_angle(pose)
        angles[~mask] = 0
        if previous is not None:
            angles = previous + settings.smoothing * (angles - previous)
            angles[~mask] = 0
        pose, _ = as_matrices(angles, 22)
        data = character.data
        world = np.empty_like(pose)
        positions = np.empty((22, 3))
        for j, parent in enumerate(PARENTS):
            local = data["local_basis"][j] @ pose[j]
            if parent < 0:
                world[j], positions[j] = local, data["local_offset"][j]
            else:
                world[j] = world[parent] @ local
                positions[j] = positions[parent] + world[parent] @ data["local_offset"][j]
        rotation = world @ data["rest_basis"].swapaxes(-1, -2)
        translation = positions - np.einsum("jik,jk->ji", rotation, data["rest_joints"])
        output = np.zeros_like(source)
        for j in range(22):
            output += (source @ rotation[j].T + translation[j]) * weights[:, j, None]
        if settings.mode != "JOINTS":
            output[ids] = guard.deform(output[ids], vertex_normals(output[ids], faces))
        transforms = []
        for r, t in zip(rotation, translation):
            matrix = np.eye(4)
            matrix[:3, :3] = C.T @ r @ C
            matrix[:3, 3] = C.T @ t
            transforms.append(Matrix(matrix.tolist()))
    return output @ C, transforms, float(np.linalg.norm(output - source, axis=1).max()), angles


class AnimationJob:
    def __init__(self, context, obj, profile, settings, directory, start, end):
        self.scene = context.scene
        self.obj = obj
        self.arm = obj.find_armature()
        self.mapping = profile["mapping"]
        self.settings = settings
        self.directory = directory
        self.start, self.end = start, end
        self.frame = start
        self.original_frame = self.scene.frame_current
        self.original_hidden = obj.hide_get()
        self.original_arm_hidden = self.arm.hide_get()
        self.original_render_hidden = obj.hide_render
        self.source_vertex_count = len(obj.data.vertices)
        self.started = time.perf_counter()
        self.weights, self.bone_groups = weights_and_groups(obj, self.mapping)
        self.slots = [i for i, item in enumerate(settings.materials) if item.enabled]
        self.output_dir = Path(directory) / "animations" / uuid.uuid4().hex
        self.output_dir.mkdir(parents=True)
        self.scene.frame_set(start)
        graph = context.evaluated_depsgraph_get()
        mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(graph), preserve_all_data_layers=True, depsgraph=graph)
        self.preview = bpy.data.objects.new(obj.name + " · Guard", mesh)
        context.collection.objects.link(self.preview)
        self.preview.matrix_world = obj.matrix_world
        self.preview["pmx_guard_result"] = True
        self.preview["source"] = obj.name
        self.preview["cache"] = str(self.output_dir / "motion.pc2")
        self.rig = self.arm.copy()
        self.rig.data = self.arm.data.copy()
        self.rig.name = self.arm.name + " · Guard baked"
        self.rig.animation_data_clear()
        self.rig.parent = None
        self.rig.matrix_world = self.arm.matrix_world
        context.collection.objects.link(self.rig)
        self.rig.hide_render = True
        self.rig["pmx_guard_result"] = True
        self.preview["skeleton"] = self.rig.name
        for bone in self.rig.pose.bones:
            for constraint in list(bone.constraints):
                bone.constraints.remove(constraint)
            bone.rotation_mode = "QUATERNION"
        self.rig.animation_data_create()
        self.action = bpy.data.actions.new(self.rig.name + " motion")
        self.rig.animation_data.action = self.action
        self.handle = (self.output_dir / "motion.pc2.partial").open("wb")
        self.handle.write(struct.pack("<12siiffi", b"POINTCACHE2\0", 1, len(mesh.vertices), float(start), 1., end - start + 1))
        self.maximum_displacement = 0.
        self.previous_quaternions = {}
        self.previous_correction = None
        self.finished = False

    def step(self):
        self.scene.frame_set(self.frame)
        if len(self.obj.data.vertices) != self.source_vertex_count:
            raise ValueError("处理期间模型拓扑发生变化")
        vertices, transforms, displacement, angles = corrected_frame(self.obj, self.mapping, self.slots,
                                     self.weights, self.settings, self.directory, self.previous_correction)
        self.previous_correction = angles.copy()
        self.maximum_displacement = max(self.maximum_displacement, displacement)
        self.handle.write(vertices.astype("<f4").tobytes())
        self.preview.data.vertices.foreach_set("co", vertices.astype(np.float32).ravel())
        self.preview.data.update()
        self.preview.matrix_world = self.obj.matrix_world
        for prop in ("location", "rotation_quaternion", "scale"):
            self.preview.rotation_mode = "QUATERNION"
            self.preview.keyframe_insert(prop, frame=self.frame)
        self.rig.matrix_world = self.arm.matrix_world
        for prop in ("location", "rotation_quaternion", "scale"):
            self.rig.rotation_mode = "QUATERNION"
            self.rig.keyframe_insert(prop, frame=self.frame)
        object_to_arm = self.arm.matrix_world.inverted() @ self.obj.matrix_world
        arm_to_object = object_to_arm.inverted()
        for source in self.arm.pose.bones:
            target = self.rig.pose.bones[source.name]
            group = self.bone_groups[source.name]
            target.matrix = object_to_arm @ transforms[group] @ arm_to_object @ source.matrix
            quaternion = target.rotation_quaternion
            previous = self.previous_quaternions.get(target.name)
            if previous is not None:
                quaternion.make_compatible(previous)
            self.previous_quaternions[target.name] = quaternion.copy()
            for prop in ("location", "rotation_quaternion", "scale"):
                target.keyframe_insert(prop, frame=self.frame, group=target.name)
        self.frame += 1
        return self.frame > self.end

    def finish(self):
        self.handle.close()
        (self.output_dir / "motion.pc2.partial").replace(self.output_dir / "motion.pc2")
        modifier = self.preview.modifiers.new("Guard animation cache", "MESH_CACHE")
        modifier.cache_format = "PC2"
        modifier.filepath = str(self.output_dir / "motion.pc2")
        modifier.frame_start = self.start
        modifier.play_mode = "SCENE"
        track = self.rig.animation_data.nla_tracks.new()
        track.name = "Guard correction result"
        strip = track.strips.new("Guard result", self.start, self.action)
        strip.blend_type = "REPLACE"
        strip.extrapolation = "NOTHING"
        self.rig.animation_data.action = None
        self.obj.hide_set(True)
        self.obj.hide_render = True
        self.preview.hide_render = False
        self.arm.hide_set(True)
        self.rig.hide_set(False)
        self.settings.result = self.preview
        self.preview["frame_start"], self.preview["frame_end"] = self.start, self.end
        self.preview["mode"] = self.settings.mode
        report = dict(start=self.start, end=self.end, frames=self.end - self.start + 1,
                      seconds=time.perf_counter() - self.started,
                      maximum_displacement_mm=self.maximum_displacement * 1000,
                      solver="pose-local evaluated geometry", source=self.obj.name)
        (self.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.scene.frame_set(self.start)
        self.finished = True
        return report

    def cancel(self):
        if not self.handle.closed:
            self.handle.close()
        for obj in (self.preview, self.rig):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data.users == 0:
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                else:
                    bpy.data.armatures.remove(data)
        if self.action.users == 0:
            bpy.data.actions.remove(self.action)
        (self.output_dir / "motion.pc2.partial").unlink(missing_ok=True)
        self.obj.hide_set(self.original_hidden)
        self.obj.hide_render = self.original_render_hidden
        self.arm.hide_set(self.original_arm_hidden)
        self.scene.frame_set(self.original_frame)
