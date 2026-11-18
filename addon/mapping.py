"""Character semantic mapping and persistent, content-addressed profiles."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROLES = ("pelvis l_hip r_hip spine1 l_knee r_knee spine2 l_ankle r_ankle "
         "spine3 l_foot r_foot neck l_collar r_collar head l_shoulder "
         "r_shoulder l_elbow r_elbow l_wrist r_wrist").split()
NAMES = ["下半身", "足D.L", "足D.R", "上半身", "ひざD.L", "ひざD.R", "上半身2",
         "足首D.L", "足首D.R", "上半身3", "足先EX.L", "足先EX.R", "首", "肩.L", "肩.R",
         "頭", "腕.L", "腕.R", "ひじ.L", "ひじ.R", "手首.L", "手首.R"]
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
COORDINATES = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])


def skeleton_data(arm):
    height = max((b.head_local.z for b in arm.data.bones), default=1) - min((b.head_local.z for b in arm.data.bones), default=0)
    rows = []
    for bone in arm.data.bones:
        if bone.name.startswith(("_shadow_", "_dummy_")):
            continue
        prop = getattr(arm.pose.bones[bone.name], "mmd_bone", None)
        rows.append(dict(name=bone.name, japanese=getattr(prop, "name_j", ""),
                         parent=bone.parent.name if bone.parent else None,
                         position=[round(v / max(height, .001), 5) for v in bone.head_local]))
    return rows


def fingerprint(arm):
    return hashlib.sha256(json.dumps(skeleton_data(arm), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate(mapping, arm):
    if set(mapping) != set(ROLES):
        raise ValueError("映射需要包含全部 22 个语义关节")
    used = []
    for role in ROLES:
        name = mapping[role]
        if name is None and role == "spine3":
            continue
        if not isinstance(name, str) or name not in arm.data.bones:
            raise ValueError(f"请指定 {role} 的骨骼")
        used.append(name)
    if len(used) != len(set(used)):
        raise ValueError("多个语义关节使用了同一根骨骼")
    for side in ("l", "r"):
        chain = [mapping[side + suffix] for suffix in ("_shoulder", "_elbow", "_wrist")]
        for parent, child in zip(chain, chain[1:]):
            if parent not in [b.name for b in arm.data.bones[child].parent_recursive]:
                raise ValueError(f"手臂骨骼层级不一致 {parent} / {child}")
    for left, right in (("l_shoulder", "r_shoulder"), ("l_hip", "r_hip")):
        if arm.data.bones[mapping[left]].head_local.x <= arm.data.bones[mapping[right]].head_local.x:
            raise ValueError(f"请检查左右映射 {left} / {right}")
    return mapping


def automatic(arm):
    aliases = {}
    for row in skeleton_data(arm):
        aliases[row["name"]] = row["name"]
        if row["japanese"]:
            aliases[row["japanese"]] = row["name"]
    result = {}
    for role, expected in zip(ROLES, NAMES):
        candidates = [expected]
        if expected.endswith((".L", ".R")):
            base, side = expected.rsplit(".", 1)
            candidates.append(("左" if side == "L" else "右") + base)
            if "D" in base:
                candidates.extend([base.replace("D", "") + "." + side,
                                   ("左" if side == "L" else "右") + base.replace("D", "")])
            if base == "足先EX":
                candidates.extend(["つま先." + side, ("左" if side == "L" else "右") + "つま先"])
        result[role] = next((aliases[n] for n in candidates if n in aliases), None)
    return result


def load_profile(obj, directory):
    arm = obj.find_armature()
    key = fingerprint(arm)
    saved = obj.get("pmx_guard_profile", "")
    candidates = []
    if saved:
        candidates.append(json.loads(saved))
    path = Path(directory) / "profiles" / (key + ".json")
    if path.exists():
        candidates.append(json.loads(path.read_text(encoding="utf-8")))
    for profile in candidates:
        if profile.get("fingerprint") == key and profile.get("version") == 1:
            validate(profile["mapping"], arm)
            return profile, True
    return dict(version=1, fingerprint=key, mapping=automatic(arm)), False


def save_profile(obj, profile, directory):
    validate(profile["mapping"], obj.find_armature())
    profile = dict(profile, version=1, fingerprint=fingerprint(obj.find_armature()))
    text = json.dumps(profile, ensure_ascii=False, indent=2)
    obj["pmx_guard_profile"] = text
    path = Path(directory) / "profiles" / (profile["fingerprint"] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return profile
