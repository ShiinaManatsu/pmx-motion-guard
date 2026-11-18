"""Main-thread capture and bulk import around a standalone numerical worker."""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import bpy
import numpy as np

from .engine import enabled_indices, weights_and_groups, validate_surfaces, sample_mesh, matrices
from .mapping import COORDINATES
from .sequence_capture import capture_steps


class OfflineBoneJob:
    def __init__(self, context, obj, profile, settings, directory, start, end):
        self.scene, self.obj, self.settings = context.scene, obj, settings
        self.arm = obj.find_armature()
        self.start, self.end, self.frame = start, end, start
        self.started = time.perf_counter()
        self.original_frame = self.scene.frame_current
        self.original_hidden = obj.hide_get()
        self.original_render_hidden = obj.hide_render
        self.original_arm_hidden = self.arm.hide_get()
        self.stage, self.stage_progress = '采集骨骼', 0.
        self.directory = Path(directory)/'animations'/uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.names = [b.name for b in self.arm.data.bones]
        self.baseline = []
        for bone in self.arm.pose.bones:
            for prop in ('location','rotation_quaternion','scale'):
                for component,value in enumerate(getattr(bone,prop)):
                    self.baseline.append((bone.path_from_id(prop),component,float(value)))
            for constraint in bone.constraints:
                self.baseline.append((constraint.path_from_id('influence'),0,constraint.influence))
        self.parents = [self.names.index(b.parent.name) if b.parent else -1 for b in self.arm.data.bones]
        unsupported = [b.name for b in self.arm.data.bones if b.inherit_scale != 'FULL' or not b.use_inherit_rotation or not b.use_local_location]
        if unsupported:
            raise ValueError('离线处理需要标准骨骼继承设置，请检查 '+ '、'.join(unsupported[:5]))
        weights, groups = weights_and_groups(obj, profile['mapping'])
        slots = [i for i,m in enumerate(settings.materials) if m.enabled]
        validate_surfaces(obj, weights, slots)
        self.groups = np.array([groups[name] for name in self.names])
        self.surface = settings.mode != 'JOINTS'
        self.bases = []
        self.vertices = None
        self.mapping = profile['mapping']
        if self.surface:
            vertices, triangles, materials = sample_mesh(obj)
            if len(vertices) != len(weights):
                raise ValueError('当前修改器改变了顶点数量，请先固定拓扑')
            self.vertices = np.lib.format.open_memmap(self.directory/'vertices.npy', mode='w+',
                dtype=np.float32, shape=(end-start+1,len(vertices),3))
            np.savez(self.directory/'surface.npz', weights=weights,
                     faces=triangles[np.isin(materials,slots)])
        self.capture_iterator = capture_steps(obj, profile['mapping'], slots, start, end)
        self.world, self.object_world, self.object_to_arm = [], [], []
        self.process = None
        self.rig = self.preview = None
        self.log = None
        self.curves = None
        self.actions = []

    def step(self):
        total = self.end-self.start+1
        if self.capture_iterator is not None:
            try:
                frame = next(self.capture_iterator)
                self.world.append(np.array([np.array(self.arm.pose.bones[name].matrix) for name in self.names], dtype=np.float32))
                self.object_world.append(np.array(self.obj.matrix_world))
                self.object_to_arm.append(np.array(self.arm.matrix_world.inverted() @ self.obj.matrix_world))
                if self.surface:
                    vertices, _, _ = sample_mesh(self.obj)
                    if len(vertices) != self.vertices.shape[1]:
                        raise ValueError('动画改变了顶点数量，请先固定拓扑')
                    self.vertices[frame-self.start] = vertices
                    self.bases.append(COORDINATES @ np.array([
                        np.array(matrix.to_quaternion().to_matrix()) for matrix in matrices(self.obj,self.mapping)]))
                self.stage_progress = (frame-self.start+1)/total*.2
            except StopIteration as completed:
                character, poses, positions, report = completed.value
                self.capture_iterator = None
                character.save(self.directory/'character.npz')
                enabled = np.zeros(22, bool)
                enabled[enabled_indices(self.settings)] = True
                np.savez(self.directory/'input.npz', poses=poses, positions=positions,
                         bone_world=np.asarray(self.world), object_world=self.object_world,
                         object_to_arm=self.object_to_arm, groups=self.groups, parents=self.parents,
                         rest=np.array([np.array(b.matrix_local) for b in self.arm.data.bones]), enabled=enabled[1:])
                if self.surface:
                    self.vertices.flush()
                    self.vertices = None
                    np.save(self.directory/'bases.npy',np.array(self.bases))
                    self.bases.clear()
                self.world.clear()
                meta = dict(fps=self.scene.render.fps/self.scene.render.fps_base,
                            native=os.environ['RETARGET_GUARD_CACHE'], capture=report,
                            mode=self.settings.mode, compression=self.settings.compression,
                            smoothing=self.settings.smoothing, start=self.start)
                (self.directory/'input.json').write_text(json.dumps(meta),encoding='utf-8')
                prefs = bpy.context.preferences.addons.get(__package__)
                python = prefs.preferences.worker_python if prefs else ''
                if not python:
                    python = str(Path(bpy.app.binary_path).parent/f'{bpy.app.version[0]}.{bpy.app.version[1]}'/'python'/'bin'/'python.exe')
                if not Path(python).is_file():
                    raise ValueError('请在插件偏好设置指定 Python 可执行文件')
                self.log = (self.directory/'worker.log').open('w',encoding='utf-8')
                self.process = subprocess.Popen([python, str(Path(__file__).with_name('offline_worker.py')),str(self.directory)],
                    stdout=self.log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                self.stage = '离线求解' if self.surface else '连续求解'
            return False
        if self.process is not None:
            code = self.process.poll()
            if code is None:
                progress = self.directory/'progress.json'
                if progress.exists():
                    try:
                        value = json.loads(progress.read_text(encoding='utf-8'))
                        self.stage_progress = .2 + .1*value['completed']/total
                    except (OSError, ValueError, KeyError):
                        pass
                return False
            self.log.close()
            self.process = None
            if code:
                raise RuntimeError('离线求解失败，日志 '+str(self.directory/'worker.log'))
            with np.load(self.directory/'result.npz',allow_pickle=False) as z:
                self.result = dict(z)
            self.prepare_output()
            self.stage = '写入动画'
            self.stage_progress = .3
            return False
        if self.curves is not None:
            for _ in range(16):
                try:
                    path, component, values = next(self.curves)
                except StopIteration:
                    self.curves = None
                    self.frame = self.end+1
                    self.stage_progress = 1.
                    return True
                curve = self.channels.fcurves.new(data_path=path,index=component)
                frames = np.arange(self.start,self.end+1)
                if np.all(values == values[0]):
                    frames, values = frames[:1], values[:1]
                curve.keyframe_points.add(len(values))
                coordinates = np.column_stack((frames,values)).astype(np.float32)
                curve.keyframe_points.foreach_set('co',coordinates.ravel())
                for point in curve.keyframe_points:
                    point.interpolation = 'LINEAR'
                curve.update()
                self.written += 1
            self.stage_progress = .3+.7*self.written/self.curve_count
            self.frame = self.start + round(total*self.written/self.curve_count)
        return False

    def prepare_output(self):
        if not self.surface:
            self.rig = self.arm
            self.action = bpy.data.actions.new(self.arm.name+' · Guard correction')
            self.actions.append(self.action)
            self.slot = self.action.slots.new(id_type='OBJECT',name=self.arm.name)
            self.channels = self.action.layers.new('Guard motion').strips.new(type='KEYFRAME').channelbag(self.slot,ensure=True)
            self.object_channels = []
            self.prepare_curves()
            return
        self.rig = self.arm.copy()
        self.rig.data = self.arm.data.copy()
        self.rig.name = self.arm.name+' · Guard offline'
        self.rig.parent = None
        for constraint in list(self.rig.constraints):
            self.rig.constraints.remove(constraint)
        self.rig.matrix_world = self.arm.matrix_world
        self.rig.animation_data_clear()
        self.scene.collection.objects.link(self.rig)
        for bone in self.rig.pose.bones:
            for constraint in list(bone.constraints):
                bone.constraints.remove(constraint)
            bone.rotation_mode = 'QUATERNION'
        self.rig.animation_data_create()
        self.action = bpy.data.actions.new(self.rig.name+' motion')
        self.actions.append(self.action)
        slot = self.action.slots.new(id_type='OBJECT',name=self.rig.name)
        layer = self.action.layers.new('Guard motion')
        strip = layer.strips.new(type='KEYFRAME')
        self.channels = strip.channelbag(slot,ensure=True)
        self.rig.animation_data.action = self.action
        self.rig.animation_data.action_slot = slot
        self.slot = slot
        if self.surface:
            graph = bpy.context.evaluated_depsgraph_get()
            mesh = bpy.data.meshes.new_from_object(self.obj.evaluated_get(graph),
                preserve_all_data_layers=True,depsgraph=graph)
            self.preview = bpy.data.objects.new(self.obj.name+' · Guard offline',mesh)
        else:
            self.preview = self.obj.copy()
            self.preview.data = self.obj.data.copy()
        self.preview.name = self.obj.name+' · Guard offline'
        self.preview.parent = None
        for constraint in list(self.preview.constraints):
            self.preview.constraints.remove(constraint)
        self.preview.matrix_world = self.obj.matrix_world
        self.preview.animation_data_clear()
        self.scene.collection.objects.link(self.preview)
        for modifier in self.preview.modifiers:
            if modifier.type == 'ARMATURE':
                modifier.object = self.rig
        if self.surface:
            modifier = self.preview.modifiers.new('Guard surface cache','MESH_CACHE')
            modifier.cache_format = 'PC2'
            modifier.filepath = str(self.directory/'motion.pc2')
            modifier.frame_start = self.start
            modifier.play_mode = 'SCENE'
            self.preview['cache'] = modifier.filepath
        self.preview['skeleton'] = self.rig.name
        self.preview['source'] = self.obj.name
        self.preview['mode'] = self.settings.mode
        self.preview['frame_start'], self.preview['frame_end'] = self.start,self.end
        self.preview['pmx_guard_result'] = self.rig['pmx_guard_result'] = True
        # Object transforms are captured separately for animated attachment roots.
        self.object_channels = []
        arm_world = np.array(self.object_world) @ np.linalg.inv(np.array(self.object_to_arm))
        from .offline_worker import quaternion_channels
        for obj, world in ((self.rig,arm_world),(self.preview,np.array(self.object_world))):
            obj.rotation_mode = 'QUATERNION'
            self.object_channels.append((obj,quaternion_channels(world[:,None])))
        self.prepare_curves()

    def prepare_curves(self):
        def curves():
            for j,name in enumerate(self.names):
                bone = self.rig.pose.bones[name]
                if bone.rotation_mode != 'QUATERNION':
                    raise ValueError('骨骼 NLA 输出需要四元数旋转模式，请检查 '+name)
                for prop,key in [('location','location'),('rotation_quaternion','quaternion'),('scale','scale')]:
                    values = self.result[key][:,j]
                    for component in range(values.shape[1]):
                        yield bone.path_from_id(prop),component,values[:,component]
        self.curves = curves()
        self.curve_count, self.written = len(self.names)*10,0

    def finish(self):
        if not self.surface:
            return self.finish_on_source()
        for obj,arrays in self.object_channels:
            action = self.action if obj==self.rig else bpy.data.actions.new(obj.name+' transform')
            if obj != self.rig:
                self.actions.append(action)
            if obj==self.rig:
                channels = self.channels
            else:
                obj.animation_data_create()
                slot = action.slots.new(id_type='OBJECT',name=obj.name)
                channels = action.layers.new('Transform').strips.new(type='KEYFRAME').channelbag(slot,ensure=True)
                obj.animation_data.action = action
                obj.animation_data.action_slot = slot
            for prop,values in zip(('location','rotation_quaternion','scale'),arrays):
                for component in range(values.shape[-1]):
                    curve = channels.fcurves.new(data_path=prop,index=component)
                    curve.keyframe_points.add(len(values))
                    curve.keyframe_points.foreach_set('co',np.column_stack((np.arange(self.start,self.end+1),values[:,0,component])).astype(np.float32).ravel())
                    for point in curve.keyframe_points:point.interpolation='LINEAR'
                    curve.update()
        track = self.rig.animation_data.nla_tracks.new()
        track.name = 'Guard offline result'
        strip = track.strips.new('Guard result',self.start,self.action)
        strip.action_slot = self.slot
        strip.extrapolation='NOTHING'
        self.rig.animation_data.action=None
        self.obj.hide_set(True)
        self.obj.hide_render = True
        self.preview.hide_set(False)
        self.preview.hide_render = False
        self.arm.hide_set(True)
        self.rig.hide_set(False)
        self.settings.result=self.preview
        self.scene.frame_set(self.start)
        report = dict(seconds=time.perf_counter()-self.started,offline=True,
                      frames=self.end-self.start+1,mode=self.settings.mode,
                      cache=str(self.directory))
        (self.directory/'job_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return report

    def finish_on_source(self):
        animation = self.arm.animation_data_create()
        source_actions = [strip.action for track in animation.nla_tracks for strip in track.strips if strip.action]
        if animation.action:
            source_actions.append(animation.action)
        animated = set()
        for action in source_actions:
            for layer in action.layers:
                for strip in layer.strips:
                    if strip.type == 'KEYFRAME':
                        for bag in strip.channelbags:
                            animated.update((curve.data_path,curve.array_index) for curve in bag.fcurves)
        animated.update((curve.data_path,curve.array_index) for curve in animation.drivers)
        base_action = bpy.data.actions.new(self.arm.name+' · Guard baseline')
        base_slot = base_action.slots.new(id_type='OBJECT',name=self.arm.name)
        base_channels = base_action.layers.new('Baseline').strips.new(type='KEYFRAME').channelbag(base_slot,ensure=True)
        for path,component,value in self.baseline:
            if (path,component) in animated:
                continue
            curve = base_channels.fcurves.new(data_path=path,index=component)
            curve.keyframe_points.insert(self.start,value)
        base_track = animation.nla_tracks.new()
        base_track.name = 'Motion Guard · baseline'
        base_strip = base_track.strips.new('Original defaults',self.start,base_action)
        base_strip.action_slot = base_slot
        base_strip.extrapolation = 'HOLD'
        # Bake evaluated constraints into the clip and disable their second pass
        # only while the correction strip contributes to the animation.
        for bone in self.arm.pose.bones:
            for constraint in bone.constraints:
                curve = self.channels.fcurves.new(data_path=constraint.path_from_id('influence'))
                curve.keyframe_points.insert(self.start,0.)
        if animation.action:
            original, slot = animation.action, animation.action_slot
            baseline = animation.nla_tracks.new()
            baseline.name = 'Original motion'
            begin, end = original.frame_range
            strip = baseline.strips.new(original.name,int(begin),original)
            strip.action_slot = slot
            strip.blend_type = animation.action_blend_type
            strip.influence = animation.action_influence
            strip.extrapolation = animation.action_extrapolation
            animation.action = None
        track = animation.nla_tracks.new()
        track.name = 'Motion Guard · correction'
        strip = track.strips.new('Guard correction',self.start,self.action)
        strip.action_slot = self.slot
        strip.blend_type = 'REPLACE'
        strip.extrapolation = 'NOTHING'
        if self.end > self.start:
            strip.frame_end = self.end
            strip.action_frame_end = self.end
        self.obj['pmx_guard_track'] = track.name
        self.obj['skeleton'] = self.arm.name
        self.obj['frame_start'],self.obj['frame_end'] = self.start,self.end
        self.obj['mode'] = 'JOINTS'
        self.settings.result = self.obj
        self.scene.frame_set(self.start)
        return dict(seconds=time.perf_counter()-self.started,offline=True)

    def cancel(self):
        self.vertices = None
        if self.capture_iterator is not None:
            self.capture_iterator.close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        if self.log and not self.log.closed:
            self.log.close()
        for obj in (self.preview,self.rig):
            if obj is not None and obj != self.arm:
                data = obj.data
                bpy.data.objects.remove(obj,do_unlink=True)
                if data.users == 0:
                    collection = bpy.data.armatures if isinstance(data,bpy.types.Armature) else bpy.data.meshes
                    collection.remove(data)
        for action in self.actions:
            if action.users == 0:
                bpy.data.actions.remove(action)
        (self.directory/'motion.pc2.partial').unlink(missing_ok=True)
        self.obj.hide_set(self.original_hidden)
        self.obj.hide_render = self.original_render_hidden
        self.arm.hide_set(self.original_arm_hidden)
        self.scene.frame_set(self.original_frame)
