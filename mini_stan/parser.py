def parse_model(text):

    blocks = {}

    current = None

    for line in text.splitlines():

        line = line.strip()

        if line.endswith("{"):
            current = line[:-1].strip()
            blocks[current] = []
            continue

        if line == "}":
            current = None
            continue

        if current is not None:
            blocks[current].append(line)

    return blocks