"""Keypoint schemas used by the real-time pipeline.

Each skeleton lists joint names, display bones, left/right flip pairs (for
flip test-time augmentation and left/right swap repair) and a rigid-limb tree
used by the bone-length constraint. Indices follow the model output order.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Skeleton:
    name: str
    joints: tuple[str, ...]
    bones: tuple[tuple[int, int], ...]
    flip_pairs: tuple[tuple[int, int], ...]
    # (parent, child) edges in parent-before-child order. Moving a child also
    # moves its descendants, so a limb keeps its shape when a bone is resized.
    limb_tree: tuple[tuple[int, int], ...]
    # Joint groups that a 2D model can confuse between body sides.
    swap_groups: tuple[tuple[tuple[int, int], ...], ...] = field(default=())

    @property
    def size(self) -> int:
        return len(self.joints)

    def flip_index(self) -> list[int]:
        index = list(range(self.size))
        for a, b in self.flip_pairs:
            index[a], index[b] = b, a
        return index

    def descendants(self) -> dict[int, list[int]]:
        children: dict[int, list[int]] = {}
        for parent, child in self.limb_tree:
            children.setdefault(parent, []).append(child)
        result: dict[int, list[int]] = {}
        for _parent, child in self.limb_tree:
            stack, found = [child], []
            while stack:
                node = stack.pop()
                found.append(node)
                stack.extend(children.get(node, []))
            result[child] = found
        return result

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "joints": list(self.joints),
            "bones": [list(b) for b in self.bones],
            "flip_pairs": [list(p) for p in self.flip_pairs],
            "limb_tree": [list(e) for e in self.limb_tree],
        }


_COCO_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
_COCO_BONES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
_COCO_FLIP = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
_COCO_LIMBS = ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16))
_ARMS = ((5, 6), (7, 8), (9, 10))
_LEGS = ((11, 12), (13, 14), (15, 16))

COCO17 = Skeleton(
    name="coco17",
    joints=_COCO_NAMES,
    bones=_COCO_BONES,
    flip_pairs=_COCO_FLIP,
    limb_tree=_COCO_LIMBS,
    swap_groups=(_LEGS, _ARMS),
)

HALPE26 = Skeleton(
    name="halpe26",
    joints=_COCO_NAMES
    + (
        "head",
        "neck",
        "hip",
        "left_big_toe",
        "right_big_toe",
        "left_small_toe",
        "right_small_toe",
        "left_heel",
        "right_heel",
    ),
    bones=_COCO_BONES
    + (
        (17, 18),
        (18, 19),
        (15, 20),
        (15, 22),
        (15, 24),
        (20, 22),
        (16, 21),
        (16, 23),
        (16, 25),
        (21, 23),
    ),
    flip_pairs=_COCO_FLIP + ((20, 21), (22, 23), (24, 25)),
    limb_tree=_COCO_LIMBS + ((15, 24), (15, 20), (16, 25), (16, 21)),
    swap_groups=(_LEGS + ((20, 21), (22, 23), (24, 25)), _ARMS),
)

SKELETONS = {s.name: s for s in (COCO17, HALPE26)}
