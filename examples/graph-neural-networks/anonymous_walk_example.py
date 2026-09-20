def anonymize_walk(walk):
    """Replace node IDs with their order of first appearance."""
    first_seen = {}
    pattern = []

    for node in walk:
        if node not in first_seen:
            first_seen[node] = len(first_seen)
        pattern.append(first_seen[node])

    return tuple(pattern)


if __name__ == "__main__":
    result = anonymize_walk([7, 2, 7, 5, 2])
    assert result == (0, 1, 0, 2, 1)
    print(result)
