"""Enrich EIN-format PHEME JSON files with timestamps from the raw archive.

The official PHEME archive stores complete Twitter JSON objects, while the
processed EIN files keep only text, users, graph indices, and labels.  This
script joins the two representations without changing EIN's graph indices or
derived stance/state fields.

By default the output must be a different directory from the input.  This is
intentional: verify the generated report before replacing any server data.
"""

from __future__ import annotations

import argparse
import copy
import json
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


TWITTER_TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


class EnrichmentError(ValueError):
    """Raised when a processed node cannot be joined to one raw tweet."""


def normalize_id(value):
    if value is None or isinstance(value, bool):
        raise EnrichmentError("missing tweet/user id")
    return str(value)


def parse_twitter_time(value):
    if not isinstance(value, str) or not value.strip():
        raise EnrichmentError("missing created_at value")
    try:
        return datetime.strptime(value, TWITTER_TIME_FORMAT)
    except ValueError as exc:
        raise EnrichmentError(
            "unsupported PHEME created_at value {!r}".format(value)
        ) from exc


@dataclass(frozen=True)
class RawTweet:
    tweet_id: str
    user_id: str
    text: str
    created_at: str
    parent_tweet_id: Optional[str]
    archive_order: int


@dataclass
class RawEvent:
    source: Optional[RawTweet] = None
    reactions: Optional[List[RawTweet]] = None
    structure_parent: Optional[Dict[str, str]] = None
    structure_order: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if self.reactions is None:
            self.reactions = []
        if self.structure_parent is None:
            self.structure_parent = {}
        if self.structure_order is None:
            self.structure_order = {}


def _tweet_id(tweet):
    value = tweet.get("id_str")
    if value in (None, ""):
        value = tweet.get("id")
    return normalize_id(value)


def _user_id(tweet):
    user = tweet.get("user")
    if not isinstance(user, dict):
        raise EnrichmentError("raw tweet {} has no user object".format(_tweet_id(tweet)))
    value = user.get("id_str")
    if value in (None, ""):
        value = user.get("id")
    return normalize_id(value)


def _raw_tweet(tweet, archive_order):
    parent = tweet.get("in_reply_to_status_id_str")
    if parent in (None, ""):
        parent = tweet.get("in_reply_to_status_id")
    return RawTweet(
        tweet_id=_tweet_id(tweet),
        user_id=_user_id(tweet),
        text=tweet.get("text"),
        created_at=tweet.get("created_at"),
        parent_tweet_id=None if parent in (None, "") else normalize_id(parent),
        archive_order=archive_order,
    )


def _walk_structure(node, parent_id, parent_map, order_map):
    if not isinstance(node, dict):
        return
    for child_id, descendants in node.items():
        child_id = normalize_id(child_id)
        if child_id in parent_map:
            raise EnrichmentError("duplicate tweet {} in structure".format(child_id))
        parent_map[child_id] = parent_id
        order_map[child_id] = len(order_map)
        _walk_structure(descendants, child_id, parent_map, order_map)


def _read_json_member(archive, member):
    file_obj = archive.extractfile(member)
    if file_obj is None:
        raise EnrichmentError("could not read archive member {}".format(member.name))
    try:
        return json.load(file_obj)
    except Exception as exc:
        raise EnrichmentError("invalid JSON in {}".format(member.name)) from exc


def load_raw_events(archive_path, wanted_event_ids):
    """Load only requested events from the official PHEME tar archive."""
    events = {event_id: RawEvent() for event_id in wanted_event_ids}
    archive_order = 0

    # Figshare names the file .tar.bz2, but the published payload is gzip.
    # Automatic detection deliberately handles both the published file and
    # any correctly recompressed local copy.
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            name = member.name.replace("\\", "/")
            basename = name.rsplit("/", 1)[-1]
            if (
                not member.isfile()
                or basename.startswith("._")
                or not name.endswith(".json")
            ):
                continue

            parts = name.split("/")
            if "/source-tweets/" in name or "/reactions/" in name:
                if len(parts) < 3:
                    continue
                event_id = parts[-3]
                if event_id not in events:
                    continue
                tweet = _raw_tweet(_read_json_member(archive, member), archive_order)
                archive_order += 1
                if "/source-tweets/" in name:
                    existing = events[event_id].source
                    if existing is not None and existing.tweet_id != tweet.tweet_id:
                        raise EnrichmentError(
                            "event {} has multiple source tweets".format(event_id)
                        )
                    events[event_id].source = tweet
                else:
                    events[event_id].reactions.append(tweet)
                continue

            if name.endswith("/structure.json"):
                if len(parts) < 2:
                    continue
                event_id = parts[-2]
                if event_id not in events:
                    continue
                structure = _read_json_member(archive, member)
                root = structure.get(event_id)
                if root is None and len(structure) == 1:
                    root = next(iter(structure.values()))
                if root is None:
                    root = {}
                _walk_structure(
                    root,
                    event_id,
                    events[event_id].structure_parent,
                    events[event_id].structure_order,
                )

    missing = [event_id for event_id, event in events.items() if event.source is None]
    if missing:
        raise EnrichmentError(
            "{} event(s) are absent from the raw archive, including {}".format(
                len(missing), missing[:5]
            )
        )
    return events


def _processed_event_id(post, path):
    source = post.get("source")
    if not isinstance(source, dict):
        raise EnrichmentError("{} has no source object".format(path))
    value = source.get("tweet id")
    if value in (None, ""):
        raise EnrichmentError("{} source has no tweet id".format(path))
    return normalize_id(value)


def load_processed_posts(input_dir):
    paths = sorted(Path(input_dir).glob("*.json"))
    if not paths:
        raise FileNotFoundError("no JSON files found in {}".format(input_dir))
    posts = []
    seen = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as file_obj:
            post = json.load(file_obj)
        event_id = _processed_event_id(post, path)
        if event_id in seen:
            raise EnrichmentError(
                "duplicate source tweet {} in {} and {}".format(
                    event_id, seen[event_id], path
                )
            )
        seen[event_id] = path
        posts.append((path, event_id, post))
    return posts


def _raw_parent(event, reaction):
    return event.structure_parent.get(
        reaction.tweet_id, reaction.parent_tweet_id
    )


def _candidate_sort_key(event, reaction):
    structure_position = event.structure_order.get(reaction.tweet_id)
    if structure_position is not None:
        return (0, structure_position, reaction.archive_order)
    return (
        1,
        parse_twitter_time(reaction.created_at).timestamp(),
        int(reaction.tweet_id),
    )


def match_comments(post, event, event_id):
    comments = post.get("comment", [])
    if not isinstance(comments, list):
        raise EnrichmentError("event {} comment is not a list".format(event_id))

    source = event.source
    if source.tweet_id != event_id:
        raise EnrichmentError(
            "event {} raw source id is {}".format(event_id, source.tweet_id)
        )
    source_time = parse_twitter_time(source.created_at)

    # EIN is built from structure.json when it is complete.  Restricting the
    # join to that set prevents disconnected raw reactions from being added.
    use_structure = len(event.structure_parent) == len(comments)
    allowed_ids = set(event.structure_parent) if use_structure else None

    by_key = defaultdict(list)
    for reaction in event.reactions:
        if reaction.tweet_id == event_id:
            continue
        if allowed_ids is not None and reaction.tweet_id not in allowed_ids:
            continue
        by_key[(reaction.user_id, reaction.text)].append(reaction)
    for candidates in by_key.values():
        candidates.sort(key=lambda reaction: _candidate_sort_key(event, reaction))

    comments_by_id = {}
    for comment in comments:
        comment_id = int(comment["comment id"])
        if comment_id in comments_by_id:
            raise EnrichmentError(
                "event {} has duplicate comment id {}".format(event_id, comment_id)
            )
        comments_by_id[comment_id] = comment

    matched = {}
    used_raw_ids = set()
    duplicate_resolutions = 0
    for comment_id in sorted(comments_by_id):
        comment = comments_by_id[comment_id]
        parent_id = int(comment["parent"])
        if parent_id == -1:
            expected_parent = event_id
        else:
            if parent_id not in matched:
                raise EnrichmentError(
                    "event {} comment {} has unresolved/non-topological parent {}".format(
                        event_id, comment_id, parent_id
                    )
                )
            expected_parent = matched[parent_id].tweet_id

        key = (normalize_id(comment.get("user id")), comment.get("content"))
        candidates = [
            reaction
            for reaction in by_key.get(key, [])
            if reaction.tweet_id not in used_raw_ids
            and _raw_parent(event, reaction) == expected_parent
        ]
        if not candidates:
            raise EnrichmentError(
                "event {} comment {} has no raw match for user/content and parent {}".format(
                    event_id, comment_id, expected_parent
                )
            )
        if len(candidates) > 1:
            duplicate_resolutions += 1
        reaction = candidates[0]
        matched[comment_id] = reaction
        used_raw_ids.add(reaction.tweet_id)

        delay_seconds = (parse_twitter_time(reaction.created_at) - source_time).total_seconds()
        if delay_seconds < 0:
            raise EnrichmentError(
                "event {} comment {} precedes its source by {} seconds".format(
                    event_id, comment_id, -delay_seconds
                )
            )

    return matched, use_structure, duplicate_resolutions


def _set_verified(record, field, value, context):
    if field in record and record[field] != value:
        raise EnrichmentError(
            "{} already has conflicting {}: {!r} != {!r}".format(
                context, field, record[field], value
            )
        )
    record[field] = value


def enrich_post(post, event, event_id):
    result = copy.deepcopy(post)
    matched, used_structure, duplicate_resolutions = match_comments(
        result, event, event_id
    )
    source = event.source
    _set_verified(
        result["source"], "created_at", source.created_at, "event {} source".format(event_id)
    )

    for comment in result.get("comment", []):
        comment_id = int(comment["comment id"])
        reaction = matched[comment_id]
        context = "event {} comment {}".format(event_id, comment_id)
        _set_verified(comment, "tweet id", reaction.tweet_id, context)
        _set_verified(comment, "created_at", reaction.created_at, context)
        _set_verified(
            comment,
            "parent tweet id",
            _raw_parent(event, reaction),
            context,
        )

    return result, {
        "comments": len(matched),
        "used_complete_structure": used_structure,
        "duplicate_candidate_resolutions": duplicate_resolutions,
    }


def enrich_directory(input_dir, archive_path, output_dir, overwrite=False):
    input_dir = Path(input_dir).resolve()
    archive_path = Path(archive_path).resolve()
    output_dir = Path(output_dir).resolve()
    if input_dir == output_dir:
        raise EnrichmentError("input and output directories must be different")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    posts = load_processed_posts(input_dir)
    events = load_raw_events(archive_path, {event_id for _, event_id, _ in posts})
    enriched = []
    totals = {
        "events": 0,
        "comments": 0,
        "events_using_complete_structure": 0,
        "events_using_reaction_fallback": 0,
        "duplicate_candidate_resolutions": 0,
    }
    for path, event_id, post in posts:
        result, stats = enrich_post(post, events[event_id], event_id)
        enriched.append((path.name, result))
        totals["events"] += 1
        totals["comments"] += stats["comments"]
        totals["duplicate_candidate_resolutions"] += stats[
            "duplicate_candidate_resolutions"
        ]
        if stats["used_complete_structure"]:
            totals["events_using_complete_structure"] += 1
        else:
            totals["events_using_reaction_fallback"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [output_dir / filename for filename, _ in enriched]
    existing = [path for path in destinations if path.exists()]
    # Keep the report outside the event directory: EIN's loader treats every
    # JSON file in source/raw as an event.
    report_path = output_dir.with_name(
        output_dir.name + "_timestamp_enrichment_summary.json"
    )
    if report_path.exists():
        existing.append(report_path)
    if existing and not overwrite:
        raise FileExistsError(
            "{} output file(s) already exist, including {}; pass --overwrite".format(
                len(existing), existing[:3]
            )
        )

    for (filename, result), destination in zip(enriched, destinations):
        with destination.open("w", encoding="utf-8") as file_obj:
            json.dump(result, file_obj, indent=4, ensure_ascii=False)

    report = {
        "input_dir": str(input_dir),
        "archive": str(archive_path),
        "output_dir": str(output_dir),
        **totals,
    }
    with report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2, ensure_ascii=False)
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore PHEME tweet timestamps in EIN-format JSON files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="EIN-format source or split/raw directory",
    )
    parser.add_argument(
        "--pheme-archive",
        required=True,
        help="official PHEME_veracity.tar.bz2 downloaded from Figshare",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new directory for enriched EIN JSON files",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = enrich_directory(
        args.input_dir,
        args.pheme_archive,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
