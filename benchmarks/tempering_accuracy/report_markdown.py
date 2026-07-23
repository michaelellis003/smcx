# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Markdown for tempering-accuracy evidence."""

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

_DASH = "—"
_ALL_HEADER = (
    "Cell|Status|Mean RMSE|Cov RMSE|Evidence RMSE|Stages|Pairs|First s|"
    "Steady s|RSS MiB|MPS MiB"
)
_CHALLENGE_HEADER = (
    "Geometry|d|N|Lane|Systematic|Multinomial|Mean RMSE S / M|"
    "Cov RMSE S / M|Evidence RMSE S / M"
)


def _bad(detail: str) -> ValueError:
    return ValueError(f"markdown evidence {detail}")


def _map(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _bad("has an invalid mapping")
    return cast(Mapping[str, Any], value)


def _text(value: object) -> str:
    if value is None:
        return _DASH
    if not isinstance(value, str | int | float) or isinstance(value, bool):
        raise _bad("has an invalid display value")
    result = str(value)
    if any(character in result for character in "\r\n\x00"):
        raise _bad("contains unsafe text")
    return result.replace("\\", "\\\\").replace("|", "\\|")


def _format(value: object, *, integer: bool = False) -> str:
    if value is None:
        return _DASH
    if type(value) not in {int, float} or not math.isfinite(cast(float, value)):
        raise _bad("has an invalid number")
    return f"{value:,.0f}" if integer else f"{value:.6g}"


def _median(value: object, *, divisor: int = 1, integer: bool = False) -> str:
    if value is None:
        return _DASH
    summary = _map(value)
    return _format(summary["median"] / divisor, integer=integer)


def _table(title: str, header: str, rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return f"## {title}\n\nNone."
    columns = header.split("|")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return f"## {title}\n\n" + "\n".join(lines)


def _eligible(item: Mapping[str, Any]) -> bool:
    accuracy = item["accuracy"]
    return bool(
        item["status"] == "eligible"
        and isinstance(accuracy, Mapping)
        and accuracy["correctness_eligible"] is True
    )


def _rmse(item: Mapping[str, Any], name: str) -> str:
    accuracy = item["accuracy"]
    if not isinstance(accuracy, Mapping):
        return _DASH
    return _format(_map(accuracy[f"{name}_loss"])["rmse"])


def _key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    cell = item["cell"]
    return tuple(
        cell[name]
        for name in ("geometry", "dimension", "reference_particles", "lane")
    )


def _label(item: Mapping[str, Any]) -> tuple[str, ...]:
    geometry, dimension, particles, lane = _key(item)
    return geometry, str(dimension), f"{particles:,}", lane


def render_markdown(
    evidence: Mapping[str, Any],
    *,
    report_date: str,
    gate_figure: str | None = None,
    cost_figure: str | None = None,
) -> str:
    """Render one campaign evidence mapping without filesystem access."""
    cells = cast(Sequence[Mapping[str, Any]], evidence["cells"])
    environment = evidence["environment"]
    integrity, host = evidence["integrity"], environment["host"]
    python = environment["python"]
    platform = " ".join(
        str(value)
        for value in (host.get("os"), host.get("macos"), host.get("machine"))
        if value
    )
    figures = " · ".join((
        f"[Gates]({gate_figure})" if gate_figure else "Gates: —",
        f"[Cost]({cost_figure})" if cost_figure else "Cost: —",
    ))
    evidence_rows = (
        ("Execution", f"{evidence['execution']['result_count']:,} / 508"),
        (
            "Source",
            f"`{environment['git_commit'][:12]}`; "
            + ("dirty" if environment["git_dirty"] else "clean"),
        ),
        (
            "Host",
            f"{_text(host.get('cpu_model') or host.get('hardware_model'))}; "
            f"{_text(platform)}",
        ),
        (
            "Python",
            f"{_text(python['implementation'])} {_text(python['version'])}",
        ),
        ("Raw leaves", f"{len(integrity['raw_leaves']):,}"),
    )
    gate_rows = []
    for label, name in (
        ("Centering", "centering"),
        ("Evidence resolution", "evidence_resolution"),
    ):
        count = evidence["gate_counts"][name]
        gate_rows.append((
            label,
            f"{count['registered']:,}",
            f"{count['evaluated']:,}",
            f"{count['passed']:,}",
        ))

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for item in cells[:72]:
        groups.setdefault(_key(item), []).append(item)
    minimum = []
    for candidates in groups.values():
        passing = [
            item["cell"]["sweeps"] for item in candidates if _eligible(item)
        ]
        minimum.append((
            *_label(candidates[0]),
            str(min(passing)) if passing else _DASH,
        ))

    systematic = {
        _key(item): item for item in cells[:72] if item["cell"]["sweeps"] == 20
    }
    challenge = []
    for multi in cells[72:]:
        standard = systematic[_key(multi)]
        challenge.append((
            *_label(multi),
            standard["status"],
            multi["status"],
            *(
                f"{_rmse(standard, name)} / {_rmse(multi, name)}"
                for name in ("mean", "covariance", "evidence")
            ),
        ))

    all_rows = []
    for item in cells:
        work, timing = item["work"], item["timing"]
        stages = pairs = _DASH
        if isinstance(work, Mapping):
            stages = _median(work["stages"], integer=True)
            pairs = _median(work["total_pairs"], integer=True)
        times = [_DASH] * 4
        if _eligible(item) and timing["status"] == "eligible":
            times = [
                _median(timing["first"]),
                _median(timing["steady"]),
                _median(timing["process_rss"], divisor=2**20),
                _median(timing["mps_peak"], divisor=2**20),
            ]
        all_rows.append((
            item["id"],
            item["status"],
            *(_rmse(item, name) for name in ("mean", "covariance", "evidence")),
            stages,
            pairs,
            *times,
        ))

    failure_names = ("ordinal", "phase", "cell_id", "block", "kind")
    failures = [
        tuple(_text(item[name]) for name in failure_names)
        for item in evidence["failures"]
    ]
    attempts = [
        (
            str(item["request_index"]),
            str(item["retry_index"]),
            _text(item["kind"]),
            _text(item["sha256"]),
        )
        for item in evidence["attempts"]
    ]
    exclusions = [
        (item["arm"], item["status"], _text(item.get("tracking_issue")))
        for item in evidence["exclusions"]
    ]
    sections = [
        f"# Tempering accuracy — {report_date}\n\n"
        f"**Verdict:** `{evidence['verdict']}`\n\n{figures}",
        _table("Evidence", "Item|Value", evidence_rows),
        _table("Gates", "Family|Registered|Evaluated|Passed", gate_rows),
        _table(
            "Minimum passing sweep",
            "Geometry|d|N|Lane|Sweeps",
            minimum,
        ),
        _table("Matched challenge", _CHALLENGE_HEADER, challenge),
        _table("All cells", _ALL_HEADER, all_rows),
        _table("Failures", "Ordinal|Phase|Cell|Block|Kind", failures),
        _table("Attempts", "Request|Retry|Kind|SHA-256", attempts),
        _table("Exclusions", "Arm|Status|Issue", exclusions),
    ]
    contract = "; ".join(
        f"{_text(name)}={_text(value)}"
        for name, value in sorted(evidence["algorithm_contract"].items())
    )
    digests = "; ".join(
        f"{name}: `{integrity[f'{name}_sha256']}`"
        for name in ("manifest", "plan", "source", "lock", "raw", "attempts")
    )
    sections.append(
        "## Methods and digests\n\n"
        f"Contract: {contract or _DASH}.\n\n"
        "Timing is shown only for correctness-eligible cells; no "
        f"cross-lane comparison is made.\n\n{digests}"
    )
    return "\n\n".join(sections) + "\n"
