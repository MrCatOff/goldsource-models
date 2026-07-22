from goldsource.merger import (
    ModelMerger,
    ModelInput,
    BoneStats,
    BoneConflict,
    MergeReport,
    MergeResult,
    TextureReplacement,
    SkinVariant,
    SkinSlot,
    MergeConfig,
)
from goldsource.smd import (
    SMD,
    Node,
    BoneTransform,
    SkeletonFrame,
    Vertex,
    Triangle,
)
from goldsource.hands import (
    HandRig,
    HandMatch,
    HandNormalisation,
    detect_rigs,
    match_hands,
    build_normalised_hand,
    load_reference_hand,
)
from goldsource.skeleton import (
    compute_keep_set,
    remove_bones,
    renumber,
    graft_ancestors,
    world_transforms,
)
from goldsource.compiler import CompileResult, compile_qc, find_studiomdl
from goldsource.pipeline import PipelineResult, ModelPrep, discover_models, run
from goldsource.qc import (
    QC,
    Sequence,
    SequenceEvent,
    BodyGroup,
    BodyGroupEntry,
    Attachment,
    HitBox,
    BoneController,
    TextureRenderMode,
    TextureGroup,
)

__all__ = [
    # Merger
    "ModelMerger",
    "ModelInput",
    "BoneStats",
    "BoneConflict",
    "MergeReport",
    "MergeResult",
    "TextureReplacement",
    "SkinVariant",
    "SkinSlot",
    "MergeConfig",
    # SMD
    "SMD",
    "Node",
    "BoneTransform",
    "SkeletonFrame",
    "Vertex",
    "Triangle",
    # QC
    "QC",
    "Sequence",
    "SequenceEvent",
    "BodyGroup",
    "BodyGroupEntry",
    "Attachment",
    "HitBox",
    "BoneController",
    "TextureRenderMode",
    "TextureGroup",
    # Hands
    "HandRig",
    "HandMatch",
    "HandNormalisation",
    "detect_rigs",
    "match_hands",
    "build_normalised_hand",
    "load_reference_hand",
    # Skeleton
    "compute_keep_set",
    "remove_bones",
    "renumber",
    "graft_ancestors",
    "world_transforms",
    # Compile + pipeline
    "CompileResult",
    "compile_qc",
    "find_studiomdl",
    "PipelineResult",
    "ModelPrep",
    "discover_models",
    "run",
]
