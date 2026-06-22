#注意，生成的html元素中的"stance_label": 1,代表对回复的贴子持反对态度
#"state": 1,对源帖子持反对态度
#"label"：1，表示源帖子是谣言
  



import argparse
import html
import json
import os
from collections import defaultdict


ROOT_NODE_ID = "source"


def short_text(text, limit):
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def wrap_text(text, width):
    text = " ".join(str(text).split())
    if not text:
        return [""]
    lines = []
    line = ""
    for char in text:
        candidate = line + char
        if len(candidate) > width:
            lines.append(line)
            line = char
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def build_nodes(post):
    nodes = {
        ROOT_NODE_ID: {
            "id": ROOT_NODE_ID,
            "kind": "source",
            "label": "source",
            "parent": None,
            "content": post.get("source", {}).get("content", ""),
            "meta": post.get("source", {}),
        }
    }

    for comment in post.get("comment", []):
        comment_id = comment.get("comment id")
        node_id = "comment-{}".format(comment_id)
        parent = comment.get("parent")
        parent_id = ROOT_NODE_ID if parent == -1 else "comment-{}".format(parent)
        nodes[node_id] = {
            "id": node_id,
            "kind": "comment",
            "label": "c{}".format(comment_id),
            "parent": parent_id,
            "content": comment.get("content", ""),
            "meta": comment,
        }

    return nodes


def build_children(nodes):
    children = defaultdict(list)
    missing_edges = []

    for node_id, node in nodes.items():
        parent_id = node.get("parent")
        if parent_id is None:
            continue
        if parent_id not in nodes:
            missing_edges.append((parent_id, node_id))
            parent_id = ROOT_NODE_ID
        children[parent_id].append(node_id)

    def sort_key(node_id):
        if node_id == ROOT_NODE_ID:
            return -1
        meta = nodes[node_id]["meta"]
        return meta.get("comment id", 10**12)

    for parent_id in children:
        children[parent_id].sort(key=sort_key)

    return children, missing_edges


def limit_tree(nodes, children, max_nodes):
    if max_nodes is None or len(nodes) <= max_nodes:
        return nodes, children, 0

    kept = set()
    stack = [ROOT_NODE_ID]
    while stack and len(kept) < max_nodes:
        node_id = stack.pop()
        if node_id in kept:
            continue
        kept.add(node_id)
        for child_id in reversed(children.get(node_id, [])):
            stack.append(child_id)

    limited_nodes = {node_id: node for node_id, node in nodes.items() if node_id in kept}
    limited_children = defaultdict(list)
    for parent_id, child_ids in children.items():
        if parent_id not in kept:
            continue
        limited_children[parent_id] = [child_id for child_id in child_ids if child_id in kept]

    return limited_nodes, limited_children, len(nodes) - len(limited_nodes)


def compute_layout(nodes, children, x_gap, y_gap):
    depth = {ROOT_NODE_ID: 0}
    ordered = []

    def dfs(node_id, current_depth):
        depth[node_id] = current_depth
        child_ids = children.get(node_id, [])
        if not child_ids:
            ordered.append(node_id)
            return
        for child_id in child_ids:
            dfs(child_id, current_depth + 1)
        ordered.append(node_id)

    dfs(ROOT_NODE_ID, 0)

    y_by_leaf_order = {}
    leaf_index = 0

    def assign_y(node_id):
        nonlocal leaf_index
        child_ids = children.get(node_id, [])
        if not child_ids:
            y_by_leaf_order[node_id] = leaf_index
            leaf_index += 1
            return y_by_leaf_order[node_id]
        child_positions = [assign_y(child_id) for child_id in child_ids]
        y_by_leaf_order[node_id] = sum(child_positions) / len(child_positions)
        return y_by_leaf_order[node_id]

    assign_y(ROOT_NODE_ID)

    positions = {}
    for node_id in nodes:
        positions[node_id] = (depth.get(node_id, 0) * x_gap, y_by_leaf_order[node_id] * y_gap)

    width = (max((depth.get(node_id, 0) for node_id in nodes), default=0) + 1) * x_gap
    height = max(leaf_index, 1) * y_gap
    return positions, width, height


def json_for_node(node):
    meta = dict(node["meta"])
    meta["node_kind"] = node["kind"]
    meta["node_label"] = node["label"]
    return json.dumps(meta, ensure_ascii=False, indent=2)


def render_svg(nodes, children, positions, args):
    node_width = args.node_width
    node_height = args.node_height
    svg_parts = []

    for parent_id, child_ids in children.items():
        for child_id in child_ids:
            x1, y1 = positions[parent_id]
            x2, y2 = positions[child_id]
            svg_parts.append(
                '<path class="edge" d="M {x1} {y1} C {mx} {y1}, {mx} {y2}, {x2} {y2}" />'.format(
                    x1=x1 + node_width,
                    y1=y1 + node_height / 2,
                    mx=(x1 + x2 + node_width) / 2,
                    x2=x2,
                    y2=y2 + node_height / 2,
                )
            )

    for node_id, node in nodes.items():
        x, y = positions[node_id]
        content = short_text(node["content"], args.node_text_limit)
        lines = wrap_text(content, args.node_line_chars)[: args.node_lines]
        title = html.escape(node["label"] + ": " + str(node["content"]))
        class_name = "node source" if node_id == ROOT_NODE_ID else "node comment"
        attrs = [
            'class="{}"'.format(class_name),
            'data-node-id="{}"'.format(html.escape(node_id)),
            'data-title="{}"'.format(html.escape(node["label"])),
            'data-content="{}"'.format(html.escape(str(node["content"]))),
            'data-meta="{}"'.format(html.escape(json_for_node(node))),
            'transform="translate({} {})"'.format(x, y),
        ]
        svg_parts.append("<g {}>".format(" ".join(attrs)))
        svg_parts.append("<title>{}</title>".format(title))
        svg_parts.append('<rect width="{}" height="{}" rx="6" />'.format(node_width, node_height))
        svg_parts.append('<text class="node-label" x="10" y="20">{}</text>'.format(html.escape(node["label"])))
        for index, line in enumerate(lines):
            svg_parts.append(
                '<text class="node-text" x="10" y="{}">{}</text>'.format(
                    40 + index * 17,
                    html.escape(line),
                )
            )
        svg_parts.append("</g>")

    return "\n".join(svg_parts)


def render_html(post, nodes, children, positions, canvas_width, canvas_height, missing_edges, truncated, args):
    title = "{} ({})".format(
        post.get("source", {}).get("tweet id", os.path.basename(args.input)),
        post.get("source", {}).get("label", "no-label"),
    )
    svg_body = render_svg(nodes, children, positions, args)
    view_width = max(canvas_width + args.node_width + 80, 900)
    view_height = max(canvas_height + args.node_height + 80, 500)
    source_content = html.escape(post.get("source", {}).get("content", ""))
    meta_summary = "nodes: {} | comments: {} | label: {}".format(
        len(nodes),
        max(0, len(nodes) - 1),
        post.get("source", {}).get("label", "unknown"),
    )
    if truncated:
        meta_summary += " | truncated: {}".format(truncated)
    if missing_edges:
        meta_summary += " | missing parents: {}".format(len(missing_edges))

    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{
  margin: 0;
  font-family: Arial, "Microsoft YaHei", sans-serif;
  color: #1f2933;
  background: #f6f8fb;
}}
.app {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  height: 100vh;
}}
.toolbar {{
  position: sticky;
  top: 0;
  z-index: 2;
  display: flex;
  gap: 12px;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #d9e2ec;
  background: #ffffff;
}}
.toolbar h1 {{
  margin: 0;
  font-size: 16px;
  font-weight: 700;
}}
.toolbar span {{
  color: #52616b;
  font-size: 13px;
}}
.graph-wrap {{
  overflow: auto;
}}
svg {{
  display: block;
  background: #fbfdff;
}}
.edge {{
  fill: none;
  stroke: #9fb3c8;
  stroke-width: 1.4;
}}
.node {{
  cursor: pointer;
}}
.node rect {{
  fill: #ffffff;
  stroke: #829ab1;
  stroke-width: 1.2;
}}
.node.source rect {{
  fill: #e3f8ff;
  stroke: #2d9cdb;
  stroke-width: 1.8;
}}
.node:hover rect, .node.active rect {{
  stroke: #ef6c00;
  stroke-width: 2.4;
}}
.node-label {{
  font-size: 13px;
  font-weight: 700;
  fill: #102a43;
}}
.node-text {{
  font-size: 12px;
  fill: #334e68;
}}
.side {{
  border-left: 1px solid #d9e2ec;
  background: #ffffff;
  min-width: 0;
  overflow: auto;
}}
.panel {{
  padding: 16px;
}}
.panel h2 {{
  margin: 0 0 10px;
  font-size: 18px;
}}
.content {{
  white-space: pre-wrap;
  line-height: 1.5;
  padding: 12px;
  background: #f6f8fb;
  border: 1px solid #d9e2ec;
  border-radius: 6px;
}}
pre {{
  white-space: pre-wrap;
  word-break: break-word;
  padding: 12px;
  background: #102a43;
  color: #f0f4f8;
  border-radius: 6px;
  font-size: 12px;
}}
@media (max-width: 900px) {{
  .app {{
    grid-template-columns: 1fr;
    grid-template-rows: minmax(0, 1fr) 45vh;
  }}
  .side {{
    border-left: 0;
    border-top: 1px solid #d9e2ec;
  }}
}}
</style>
</head>
<body>
<div class="app">
  <main class="graph-wrap">
    <div class="toolbar">
      <h1>{title}</h1>
      <span>{meta_summary}</span>
    </div>
    <svg width="{view_width}" height="{view_height}" viewBox="-40 -40 {view_width} {view_height}">
      {svg_body}
    </svg>
  </main>
  <aside class="side">
    <div class="panel">
      <h2 id="node-title">source</h2>
      <div id="node-content" class="content">{source_content}</div>
      <h2>metadata</h2>
      <pre id="node-meta"></pre>
    </div>
  </aside>
</div>
<script>
const titleEl = document.getElementById("node-title");
const contentEl = document.getElementById("node-content");
const metaEl = document.getElementById("node-meta");
const nodes = Array.from(document.querySelectorAll(".node"));

function selectNode(node) {{
  nodes.forEach(n => n.classList.remove("active"));
  node.classList.add("active");
  titleEl.textContent = node.dataset.title || "";
  contentEl.textContent = node.dataset.content || "";
  metaEl.textContent = node.dataset.meta || "";
}}

nodes.forEach(node => {{
  node.addEventListener("click", () => selectNode(node));
}});

const sourceNode = document.querySelector('.node[data-node-id="source"]');
if (sourceNode) {{
  selectNode(sourceNode);
}}
</script>
</body>
</html>
""".format(
        title=html.escape(title),
        meta_summary=html.escape(meta_summary),
        source_content=source_content,
        view_width=int(view_width),
        view_height=int(view_height),
        svg_body=svg_body,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a rumor propagation JSON file as an HTML/SVG reply graph."
    )
    parser.add_argument("input", help="Path to a dataset JSON file.")
    parser.add_argument("-o", "--output", help="Output HTML path.")
    parser.add_argument("--max-nodes", type=int, default=None, help="Only render the first N DFS nodes.")
    parser.add_argument("--x-gap", type=int, default=280, help="Horizontal gap between hop levels.")
    parser.add_argument("--y-gap", type=int, default=92, help="Vertical gap between leaf nodes.")
    parser.add_argument("--node-width", type=int, default=230, help="Node rectangle width.")
    parser.add_argument("--node-height", type=int, default=78, help="Node rectangle height.")
    parser.add_argument("--node-text-limit", type=int, default=70, help="Max text chars shown inside nodes.")
    parser.add_argument("--node-line-chars", type=int, default=24, help="Approx chars per text line inside nodes.")
    parser.add_argument("--node-lines", type=int, default=2, help="Max text lines shown inside nodes.")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as file_obj:
        post = json.load(file_obj)

    nodes = build_nodes(post)
    children, missing_edges = build_children(nodes)
    nodes, children, truncated = limit_tree(nodes, children, args.max_nodes)
    positions, canvas_width, canvas_height = compute_layout(nodes, children, args.x_gap, args.y_gap)
    output = args.output
    if not output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output = os.path.join("experiments", "visualizations", "{}.html".format(base_name))

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    page = render_html(
        post,
        nodes,
        children,
        positions,
        canvas_width,
        canvas_height,
        missing_edges,
        truncated,
        args,
    )
    with open(output, "w", encoding="utf-8") as file_obj:
        file_obj.write(page)

    print("Wrote {}".format(output))
    print("Rendered {} nodes and {} edges.".format(len(nodes), sum(len(v) for v in children.values())))


if __name__ == "__main__":
    main()
