import torch


P2T3_SOURCE_TYPE = 0
P2T3_DEEP_CONVERSATION_TYPE = 1
P2T3_SHALLOW_CONVERSATION_TYPE = 2


def extract_conversation_chains(
    edge_index,
    num_nodes,
    max_chains=None,
    max_chain_length=40,
):
    """Extract root-to-leaf reply chains from a directed propagation tree.

    Node 0 is the source post. Every returned chain begins with a direct reply
    to the source and ends at a leaf. Shared prefixes are intentionally
    repeated, matching the released P2T3 preprocessing and Figure 3 of the
    paper.
    """

    num_nodes = int(num_nodes)
    if num_nodes < 1:
        raise ValueError("P2T3 requires every propagation tree to have a root node.")

    max_chain_length = max(1, int(max_chain_length))
    if max_chains is not None:
        max_chains = max(0, int(max_chains))

    children = [[] for _ in range(num_nodes)]
    if edge_index is not None and edge_index.numel() > 0:
        for source, target in edge_index.detach().cpu().t().tolist():
            source = int(source)
            target = int(target)
            if (
                0 <= source < num_nodes
                and 0 <= target < num_nodes
                and source != target
                and target not in children[source]
            ):
                children[source].append(target)

    for child_list in children:
        child_list.sort()

    chains = []

    def append_chain(path):
        if path:
            chains.append(path)

    def walk(node, path, visited):
        path = path + [node]
        if len(path) >= max_chain_length:
            append_chain(path)
            return

        next_nodes = [
            child for child in children[node] if child not in visited
        ]
        if not next_nodes:
            append_chain(path)
            return

        next_visited = visited | {node}
        for child in next_nodes:
            walk(child, path, next_visited)

    for level_one_node in children[0]:
        walk(level_one_node, [], {0})

    # The released dataset writes every deep conversation first, followed by
    # all one-reply shallow conversations, irrespective of root-child order.
    deep_chains = [chain for chain in chains if len(chain) > 1]
    shallow_chains = [chain for chain in chains if len(chain) == 1]
    ordered_chains = deep_chains + shallow_chains
    if max_chains is not None:
        ordered_chains = ordered_chains[:max_chains]
    return ordered_chains


def _root_depths(edge_index, num_nodes):
    children = [[] for _ in range(num_nodes)]
    if edge_index is not None and edge_index.numel() > 0:
        for source, target in edge_index.detach().cpu().t().tolist():
            source = int(source)
            target = int(target)
            if (
                0 <= source < num_nodes
                and 0 <= target < num_nodes
                and source != target
                and target not in children[source]
            ):
                children[source].append(target)

    depths = [-1] * num_nodes
    depths[0] = 0
    frontier = [0]
    while frontier:
        source = frontier.pop(0)
        for target in children[source]:
            candidate = depths[source] + 1
            if depths[target] < 0 or candidate < depths[target]:
                depths[target] = candidate
                frontier.append(target)
    return depths


def build_p2t3_sequence_metadata(
    edge_index,
    num_nodes,
    max_sequence_length=1000,
    max_chain_length=40,
    max_chain_identifiers=512,
):
    """Build the token metadata consumed by the EIN-compatible P2T3 model."""

    max_sequence_length = max(1, int(max_sequence_length))
    max_chain_identifiers = max(1, int(max_chain_identifiers))
    max_conversations = max_chain_identifiers - 1

    chains = extract_conversation_chains(
        edge_index,
        num_nodes,
        max_chains=max_conversations,
        max_chain_length=max_chain_length,
    )
    depths_by_node = _root_depths(edge_index, int(num_nodes))

    node_ids = [0]
    chain_ids = [0]
    depths = [0]
    type_ids = [P2T3_SOURCE_TYPE]
    level_one_mask = [False]

    for chain_id, chain in enumerate(chains, start=1):
        remaining = max_sequence_length - len(node_ids)
        if remaining <= 0:
            break

        conversation_type = (
            P2T3_DEEP_CONVERSATION_TYPE
            if len(chain) > 1
            else P2T3_SHALLOW_CONVERSATION_TYPE
        )
        clipped_chain = chain[:remaining]
        for offset, node_id in enumerate(clipped_chain):
            node_ids.append(node_id)
            chain_ids.append(chain_id)
            node_depth = depths_by_node[node_id]
            depths.append(node_depth if node_depth >= 0 else offset + 1)
            type_ids.append(conversation_type)
            level_one_mask.append(offset == 0)

    return {
        "p2t3_node_id": torch.tensor(node_ids, dtype=torch.long),
        "p2t3_chain_id": torch.tensor(chain_ids, dtype=torch.long),
        "p2t3_depth": torch.tensor(depths, dtype=torch.long),
        "p2t3_type_id": torch.tensor(type_ids, dtype=torch.long),
        "p2t3_level_one_mask": torch.tensor(
            level_one_mask,
            dtype=torch.bool,
        ),
        "p2t3_sequence_length": torch.tensor(
            [len(node_ids)],
            dtype=torch.long,
        ),
    }


def attach_p2t3_sequence_metadata(data, args):
    metadata = build_p2t3_sequence_metadata(
        getattr(data, "directed_edge_index", data.edge_index),
        data.num_nodes,
        max_sequence_length=getattr(
            args,
            "p2t3_max_sequence_length",
            1000,
        ),
        max_chain_length=getattr(args, "p2t3_max_chain_length", 40),
        max_chain_identifiers=getattr(
            args,
            "p2t3_max_chain_identifiers",
            getattr(args, "p2t3_d_model", 512),
        ),
    )
    for name, value in metadata.items():
        setattr(data, name, value)
    return data
