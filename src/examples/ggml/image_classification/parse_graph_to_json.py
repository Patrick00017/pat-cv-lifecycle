import json
import re


def parse_compute_graph(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    nodes = []

    # Skip header lines (first 4 lines)
    # Line format: opcode | name | target | args | kwargs
    # Fixed width columns based on the tabular output

    for line in lines[4:]:  # Skip first 4 lines (header)
        line = line.rstrip('\n')
        if not line.strip():
            continue

        # Stop at output node (last valid node before warnings)
        if line.startswith('output'):
            opcode = "output"
            name = "output"
            target = "output"
            args_str = line.split('output')[1].strip() if 'output' in line else ""
            args = []
            if args_str:
                match = re.match(r'\((.*)\)', args_str)
                if match:
                    args = [match.group(1).strip()]
            nodes.append({
                "opcode": opcode,
                "name": name,
                "target": target,
                "args": args,
                "kwargs": {}
            })
            break

        # Parse fixed-width columns based on the tabular format
        # opcode (col 0-14), name (col 15-94), target (col 95-174), args (175-450), kwargs (451+)
        opcode = line[0:15].strip()
        name = line[15:94].strip()
        target = line[95:174].strip()
        args_str = line[175:450].strip() if len(line) > 175 else ""
        kwargs_str = line[450:].strip() if len(line) > 450 else ""

        # Parse args - extract content within parentheses
        args = []
        if args_str:
            # The args are shown as "(arg1, arg2, ...)" format
            # Extract the content between parentheses
            match = re.match(r'\((.*)\)', args_str)
            if match:
                args_content = match.group(1).strip()
                if args_content:
                    # Split by comma but be careful with nested parens
                    args = parse_args(args_content)

        # Parse kwargs - extract content within braces
        kwargs = {}
        if kwargs_str:
            match = re.match(r'\{(.*)\}', kwargs_str)
            if match:
                kwargs_content = match.group(1).strip()
                if kwargs_content:
                    # Parse key=value pairs
                    for pair in kwargs_content.split(','):
                        pair = pair.strip()
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            kwargs[key.strip()] = value.strip()

        node = {
            "opcode": opcode,
            "name": name,
            "target": target,
            "args": args,
            "kwargs": kwargs
        }
        nodes.append(node)

    result = {
        "metadata": {
            "source": input_file,
            "model": "resnet50",
            "description": "FX compute graph from torch.compile"
        },
        "nodes": nodes
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Parsed {len(nodes)} nodes to {output_file}")
    return result


def parse_args(args_content):
    args = []
    depth = 0
    current = ""

    for char in args_content:
        if char == '(':
            depth += 1
            current += char
        elif char == ')':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            if current.strip():
                args.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        args.append(current.strip())

    return args


if __name__ == "__main__":
    parse_compute_graph("compute_graph.txt", "compute_graph.json")