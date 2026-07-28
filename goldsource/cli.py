"""
Command-line interface for merging GoldSource weapon models.

    python -m goldsource merge storage/decompiled/pistols -o storage/build/pistols --compile

Subcommands
-----------
``merge``    run the whole pipeline: normalise hands, prune bones, merge, compile
``analyze``  report bone counts, conflicts and the hand match without writing
``compile``  run studiomdl on an existing QC
``hands``    show how the reference hand maps onto each model's bones
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

from goldsource.compiler import compile_qc, find_studiomdl
from goldsource.config import AppConfig
from goldsource.hands import detect_rigs, load_reference_hand, match_hands
from goldsource.merger import MergeConfig, ModelInput, ModelMerger
from goldsource.pipeline import discover_models, run


DEFAULT_HAND = Path("storage") / "hands" / "default_hand.smd"


# ---------------------------------------------------------------------------
# Shared arguments
# ---------------------------------------------------------------------------

def _add_hand_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hands", metavar="SMD", default=str(DEFAULT_HAND),
        help=f"reference hand mesh to rebind onto every model (default: {DEFAULT_HAND})",
    )
    parser.add_argument(
        "--hand-texture", metavar="BMP", default=None,
        help="texture for the reference hand (default: the .bmp next to --hands, if any)",
    )
    parser.add_argument(
        "--no-hands", dest="normalise", action="store_false",
        help="keep each model's original hand mesh",
    )


def _resolve_hand_texture(args: argparse.Namespace) -> Path | None:
    if getattr(args, "hand_texture", None):
        return Path(args.hand_texture)
    sibling = Path(args.hands).with_suffix(".bmp")
    return sibling if sibling.exists() else None


def _parse_renames(pairs: list[str] | None) -> list[tuple[str, str]]:
    renames: list[tuple[str, str]] = []
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--rename expects FIND=REPLACE, got {pair!r}")
        find, replace = pair.split("=", 1)
        renames.append((find, replace))
    return renames


def _parse_decimate_models(args: argparse.Namespace) -> dict[str, float]:
    """Parse ``--decimate-model MODEL=RATIO`` overrides."""
    out: dict[str, float] = {}
    for pair in getattr(args, "decimate_model", None) or []:
        if "=" not in pair:
            raise SystemExit(f"--decimate-model expects MODEL=RATIO, got {pair!r}")
        model, ratio = pair.rsplit("=", 1)
        try:
            out[model] = float(ratio)
        except ValueError:
            raise SystemExit(f"--decimate-model: {ratio!r} is not a number")
    return out


def _parse_keep_groups(args: argparse.Namespace) -> dict[str, set[str] | str]:
    """
    Collect the bodygroups that stay switchable, from ``--groups`` JSON and
    ``--keep-group MODEL:GROUP``.

    The JSON is ``{"model": ["group", ...]}``; a value of ``"*"`` keeps every
    group that model has.  ``--keep-group`` accepts ``MODEL:*`` for the same.
    """
    keep: dict[str, set[str] | str] = {}

    if getattr(args, "groups", None):
        path = pathlib.Path(args.groups)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"--groups: cannot read {path}: {exc}")
        if not isinstance(loaded, dict):
            raise SystemExit(f"--groups: expected an object at the top level of {path}")
        for model, groups in loaded.items():
            if isinstance(groups, str):
                if groups != "*":
                    raise SystemExit(f"--groups: {model!r} expects a list or \"*\", got {groups!r}")
                keep[model] = "*"
            else:
                keep[model] = set(groups)

    for pair in getattr(args, "keep_group", None) or []:
        if ":" not in pair:
            raise SystemExit(f"--keep-group expects MODEL:GROUP, got {pair!r}")
        model, group = pair.split(":", 1)
        if group == "*":
            keep[model] = "*"
        elif keep.get(model) != "*":
            existing = keep.setdefault(model, set())
            assert isinstance(existing, set)
            existing.add(group)

    return keep


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def cmd_merge(args: argparse.Namespace) -> int:
    if not args.inputs:
        raise SystemExit("merge: no inputs given (pass model directories, or set \"inputs\" in --config)")
    if not args.output:
        raise SystemExit("merge: no output given (pass -o/--output, or set \"output\" in --config)")

    merge_config = MergeConfig(sequence_renames=_parse_renames(args.rename))
    # A skin-oriented AppConfig (models/skin slots) is loaded here as before; an
    # options --config was already folded into the parser defaults in main().
    if args.config:
        raw = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and ("models" in raw or "skin_slots" in raw):
            loaded = AppConfig.load(args.config).build_merge_config()
            loaded.sequence_renames.extend(merge_config.sequence_renames)
            merge_config = loaded
    merge_config.index_sequence_names = getattr(args, "index_sequences", False)

    model_name = args.name
    if not model_name.lower().endswith(".mdl"):
        model_name += ".mdl"

    # --default-hands: offer the two hands in storage/hands/default (male, female)
    # as a shared, selectable hands bodypart.  The male hand doubles as the
    # matching reference, and the hands are used reference-posed (no re-pose).
    hand_variants = None
    if getattr(args, "default_hands", False):
        base = Path("storage") / "hands" / "default"
        hand_variants = [
            (base / "male.smd", base / "male.bmp"),
            (base / "female.smd", base / "female.bmp"),
        ]
        for smd, tex in hand_variants:
            if not smd.exists():
                raise SystemExit(f"--default-hands: missing {smd}")
        args.hands = str(base / "male.smd")
        args.repose_hands = False

    def log(message: str) -> None:
        if not args.quiet:
            print(message)

    result = run(
        inputs=args.inputs,
        output_dir=args.output,
        model_name=model_name,
        hand_smd=args.hands if args.normalise else None,
        hand_texture=_resolve_hand_texture(args) if args.normalise else None,
        normalise=args.normalise,
        prune=args.prune,
        decimate=args.decimate,
        decimate_overrides=_parse_decimate_models(args),
        pack_parts=args.pack_parts,
        vertex_budget=args.vertex_budget,
        keep_hitbox_bones=args.keep_hitbox_bones,
        keep_animated_bones=args.keep_animated_bones,
        share_hands=args.share_hands,
        repose_hands=args.repose_hands,
        keep_hand_mesh=args.keep_hand_mesh,
        hand_match_max_cost=getattr(args, "hand_match_max_cost", 1.0),
        retarget_fingers=getattr(args, "retarget_fingers", False)
                         or getattr(args, "default_hands", False),
        hand_variants=hand_variants,
        unify_skeleton=args.unify_skeleton,
        pool_bones_pass=args.pool_bones,
        flatten_weapon_bones=getattr(args, "flatten_weapon_bones", False),
        short_paths=getattr(args, "short_paths", True),
        bone_target=args.bone_target,
        keep_groups=_parse_keep_groups(args),
        single_group=args.single_group,
        sanitise=args.sanitise,
        max_texture_size=args.max_texture_size,
        exclude=args.exclude,
        merge_config=merge_config,
        compile_model=args.compile,
        studiomdl=args.studiomdl,
        ignore_warnings=args.ignore_warnings,
        write=not args.dry_run,
        log=log,
    )

    print()
    print(result.summary())

    if result.compile is not None and not result.compile.ok:
        output = result.compile.stdout.strip()
        if output:
            print()
            print("studiomdl output:")
            print(output)
        return 1

    if result.exceeds_bodygroup_limits:
        return 1

    if result.merge is not None and result.merge.report.exceeds_limit:
        return 1

    return 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> int:
    def log(message: str) -> None:
        if not args.quiet:
            print(message)

    result = run(
        inputs=args.inputs,
        output_dir=".",
        model_name="analysis.mdl",
        hand_smd=args.hands if args.normalise else None,
        hand_texture=_resolve_hand_texture(args) if args.normalise else None,
        normalise=args.normalise,
        prune=args.prune,
        pack_parts=args.pack_parts,
        keep_hitbox_bones=args.keep_hitbox_bones,
        keep_animated_bones=args.keep_animated_bones,
        exclude=args.exclude,
        write=False,
        log=log,
    )

    print()
    print(result.summary())
    if result.merge is not None:
        print()
        print(result.merge.report.summary())
    return 1 if result.merge is not None and result.merge.report.exceeds_limit else 0


# ---------------------------------------------------------------------------
# hands
# ---------------------------------------------------------------------------

def cmd_hands(args: argparse.Namespace) -> int:
    reference, reference_rigs = load_reference_hand(args.hands)
    print(f"{Path(args.hands).name}: {len(reference.nodes)} bones, "
          f"{len(reference.triangles)} triangles")
    for rig in reference_rigs:
        print(f"  rig {rig.hand} (forearm {rig.forearm}), {len(rig.chains)} finger chains")

    directories: list[Path] = []
    for item in args.inputs:
        directories.extend(discover_models(item))

    for directory in directories:
        model = ModelInput.from_directory(directory.name, directory)
        donor = None
        for bodygroup in model.qc.bodygroups:
            if "hand" not in bodygroup.name.lower():
                continue
            for entry in bodygroup.entries:
                if entry.is_blank:
                    continue
                key = entry.smd.replace("\\", "/").lstrip("./")
                donor = model.smds.get(key) or next(
                    (s for k, s in model.smds.items()
                     if k.split("/")[-1].lower() == key.split("/")[-1].lower()),
                    None,
                )
                if donor is not None:
                    break
            if donor is not None:
                break

        print(f"\n{directory.name}")
        if donor is None:
            print("  no hand bodygroup found")
            continue

        model_rigs = detect_rigs(donor)
        match = match_hands(reference, reference_rigs, donor, model_rigs)
        print(f"  match cost {match.score:.3f} (0 = identical rigs)")
        for reference_hand_bone, model_hand_bone in match.pairs:
            print(f"  {reference_hand_bone} -> {model_hand_bone}")
        for source, target in sorted(match.mapping.items()):
            print(f"    {source:<24} -> {target}")
        if match.unmapped:
            print(f"  UNMAPPED: {', '.join(match.unmapped)}")
    return 0


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

def cmd_compile(args: argparse.Namespace) -> int:
    result = compile_qc(
        args.qc,
        studiomdl=args.studiomdl,
        ignore_warnings=args.ignore_warnings,
        keep_unused_bones=args.keep_bones,
    )
    print(result.stdout.strip())
    if result.ok:
        size = result.output_mdl.stat().st_size
        print(f"\nOK: {result.output_mdl} ({size / 1024:.0f} KB)")
        return 0
    print(f"\nFAILED (exit {result.returncode})")
    return 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _load_merge_config(path: str) -> dict | None:
    """
    Read a ``--config`` JSON.  Returns a dict of merge-option defaults, or ``None``
    when the file is a skin-oriented :class:`AppConfig` (``models``/``skin_slots``),
    which :func:`cmd_merge` loads separately.
    """
    try:
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"--config: cannot read {path}: {exc}")
    if not isinstance(raw, dict):
        raise SystemExit(f"--config: expected a JSON object at the top level of {path}")
    if "models" in raw or "skin_slots" in raw:
        return None
    return raw


def _apply_merge_config(merge_parser: argparse.ArgumentParser, config: dict) -> None:
    """Make *config*'s values the merge parser's defaults; command-line flags still win."""
    valid = {action.dest for action in merge_parser._actions}
    # Keys starting with "_" are treated as free-form comments and ignored.
    unknown = sorted(key for key in config
                     if key not in valid and not key.startswith("_"))
    if unknown:
        print(f"warning: --config: ignoring unknown option(s): {', '.join(unknown)}",
              file=sys.stderr)
    merge_parser.set_defaults(**{k: v for k, v in config.items() if k in valid})


def build_parser(merge_config: dict | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goldsource",
        description="Merge decompiled GoldSource weapon models into one model with submodels.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- merge ---
    merge = subparsers.add_parser(
        "merge", help="normalise hands, prune bones, merge and optionally compile",
    )
    merge.add_argument("inputs", nargs="*", help="model directories, or a directory of them "
                                                  "(may instead come from --config)")
    merge.add_argument("-o", "--output", help="output directory (may instead come from --config)")
    merge.add_argument("-n", "--name", default="merged.mdl", help="output model name")
    _add_hand_arguments(merge)
    merge.add_argument("--no-prune", dest="prune", action="store_false",
                       help="keep bones that carry no geometry")
    merge.add_argument("--decimate", type=float, default=None, metavar="RATIO",
                       help="reduce weapon meshes to RATIO of their vertices "
                            "(e.g. 0.5 = halve; lossy). Fewer triangles and submodels; "
                            "the optimised hand and animations are left untouched")
    merge.add_argument("--decimate-model", action="append", metavar="MODEL=RATIO",
                       help="override --decimate for one model (e.g. v_ak47chimera=0.12 "
                            "to force it into a single submodel); repeatable")
    merge.add_argument("--no-pack", dest="pack_parts", action="store_false",
                       help="keep every always-on mesh in its own bodygroup instead of "
                            "packing them (more bodyparts, harder to view)")
    merge.add_argument("--keep-animated-bones", action="store_true",
                       help="do not fold bones that move; costs bones but keeps "
                            "animation data compressible (use if studiomdl reports "
                            "a sequence over 64K)")
    merge.add_argument("--keep-hitbox-bones", action="store_true",
                       help="let $hbox entries pin bones against pruning "
                            "(costs shared-skeleton collapse; hitboxes are inert on view models)")
    merge.add_argument("--max-texture-size", type=int, default=None, metavar="N",
        help="downscale any texture whose longest side exceeds N pixels "
             "(e.g. 256) to shrink the .mdl; UVs are unaffected")
    merge.add_argument("--vertex-budget", type=int, default=2048, metavar="N",
                       help="vertices allowed per submodel when packing parts (studiomdl MAXSTUDIOVERTS)")
    merge.add_argument("--bone-target", type=int, default=127, metavar="N",
                       help="bones the pool may grow to before it starts re-anchoring "
                            "to reuse a slot; lower trades animation size for bones")
    merge.add_argument("--no-pool-bones", dest="pool_bones", action="store_false",
                       help="do not let models share weapon bone slots "
                            "(costs the sum of every model's bones instead of the largest)")
    merge.add_argument("--flatten-weapon-bones", action="store_true",
                       help="before pooling, root every bone whose parent disagrees "
                            "across models; removes 'illegal parent bone replacement' "
                            "conflicts so more models pool into one skeleton (trades "
                            "animation size for bone count)")
    merge.add_argument("--no-short-paths", dest="short_paths", action="store_false",
                       help="do not shorten nested output SMD paths; studiomdl may then "
                            "truncate long <model>/<model>_anims/<smd> sequence paths")
    merge.add_argument("--all-groups", dest="single_group", action="store_false",
                       help="keep every switchable bodygroup instead of one weapon submodel per model")
    merge.add_argument("--groups", metavar="JSON",
                       help='per-model bodygroups to keep switchable: {"v_x": ["scope"], "v_y": "*"}')
    merge.add_argument("--keep-group", action="append", metavar="MODEL:GROUP",
                       help="keep one bodygroup switchable (MODEL:* for all of a model's)")
    merge.add_argument("--shared-hand", dest="repose_hands", action="store_false",
                       help="use ONE reference-posed hand for every model instead of "
                            "re-posing per model (far less geometry, but stretches models "
                            "whose hands sit far from the reference pose)")
    merge.add_argument("--no-share-hands", dest="share_hands", action="store_false",
                       help="write one hand mesh copy per model instead of sharing one")
    merge.add_argument("--original-hands", dest="keep_hand_mesh", action="store_true",
                       help="keep each model's OWN hand mesh, but still rename its hand "
                            "bones onto the common naming so the skeleton is shared and "
                            "pooled (bone-sharing without hand-sharing)")
    merge.add_argument("--hand-match-max-cost", dest="hand_match_max_cost",
                       type=float, default=1.0, metavar="COST",
                       help="if the reference hand fits a model's rig worse than COST "
                            "(default 1.0; a matching rig scores 0), keep that model's own "
                            "hand mesh instead of reposing the shared one onto it — the "
                            "repose warps badly-matched fingers (v_bhdagger, v_knifedragon). "
                            "Set 0 to always share")
    merge.add_argument("--default-hands", dest="default_hands", action="store_true",
                       help="use the two hands in storage/hands/default (male, female) as a "
                            "shared, selectable hands bodypart for every weapon; "
                            "add the reported stride to a weapon's pev_body to pick female")
    merge.add_argument("--retarget-hands", dest="retarget_fingers", action="store_true",
                       help="bake each weapon's finger animation onto the shared hand's bone "
                            "lengths so one fixed hand mesh curls without stretching (no "
                            "per-weapon re-pose needed). Implied by --default-hands")
    merge.add_argument("--no-unify-skeleton", dest="unify_skeleton", action="store_false",
                       help="write only each weapon's own bones per SMD instead of the full "
                            "merged skeleton in every SMD (leaner, but can hit studiomdl's "
                            "'illegal parent bone replacement' on pooled bones)")
    merge.add_argument("--no-sanitise", dest="sanitise", action="store_false",
                       help="do not rename non-ASCII source filenames")
    merge.add_argument("--exclude", action="append", metavar="NAME",
                       help="skip a model directory by name (repeatable)")
    merge.add_argument("--index-sequences", action="store_true",
        help="rename every sequence to {model}_seq_{i} for easy identification "
             "in Model Viewer (original names kept as models.ini keys)")
    merge.add_argument("--rename", action="append", metavar="FIND=REPLACE",
                       help="sequence name rewrite rule (repeatable)")
    merge.add_argument("--config", metavar="JSON",
                       help="load merge options from a JSON file — every long option is a key "
                            "(e.g. {\"inputs\": [...], \"output\": \"...\", \"exclude\": [...], "
                            "\"flatten_weapon_bones\": true}); flags on the command line override "
                            "it. See storage/configs/knives.json. (An AppConfig JSON with "
                            "\"models\"/skin slots is still read as before.)")
    merge.add_argument("--compile", action="store_true", help="run studiomdl on the result")
    merge.add_argument("--studiomdl", metavar="EXE", help="path to studiomdl")
    merge.add_argument("--ignore-warnings", action="store_true",
                       help="pass -i to studiomdl")
    merge.add_argument("--dry-run", action="store_true", help="analyse without writing files")
    if merge_config:
        _apply_merge_config(merge, merge_config)
    merge.set_defaults(func=cmd_merge)

    # --- analyze ---
    analyze = subparsers.add_parser("analyze", help="report bones and conflicts, write nothing")
    analyze.add_argument("inputs", nargs="+", help="model directories, or a directory of them")
    _add_hand_arguments(analyze)
    analyze.add_argument("--no-prune", dest="prune", action="store_false")
    analyze.add_argument("--keep-hitbox-bones", action="store_true")
    analyze.add_argument("--keep-animated-bones", action="store_true")
    analyze.add_argument("--no-pack", dest="pack_parts", action="store_false")
    analyze.add_argument("--exclude", action="append", metavar="NAME")
    analyze.set_defaults(func=cmd_analyze)

    # --- hands ---
    hands = subparsers.add_parser("hands", help="show the reference-hand bone mapping per model")
    hands.add_argument("inputs", nargs="+", help="model directories, or a directory of them")
    hands.add_argument("--hands", metavar="SMD", default=str(DEFAULT_HAND))
    hands.set_defaults(func=cmd_hands)

    # --- compile ---
    compile_parser = subparsers.add_parser("compile", help="run studiomdl on a QC file")
    compile_parser.add_argument("qc", help="path to the .qc file")
    compile_parser.add_argument("--studiomdl", metavar="EXE")
    compile_parser.add_argument("--ignore-warnings", action="store_true")
    compile_parser.add_argument("--keep-bones", action="store_true",
                                help="pass -k so studiomdl keeps unused bones")
    compile_parser.set_defaults(func=cmd_compile)

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    # A merge --config file provides option defaults; load it first so the real
    # parser can seed its defaults from it (command-line flags still override).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(raw_argv)
    merge_config = _load_merge_config(known.config) if known.config else None
    parser = build_parser(merge_config)
    args = parser.parse_args(raw_argv)
    try:
        return args.func(args)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
