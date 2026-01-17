import random
import sys

def generate_synthetic_obfuscation(ast_node):
    if isinstance(ast_node, list):
        if len(ast_node) > 1 and random.random() < 0.1:
            new_list = ast_node.copy()
            i, j = random.sample(range(len(new_list)), 2)
            new_list[i], new_list[j] = new_list[j], new_list[i]
            return [generate_synthetic_obfuscation(n) for n in new_list]
        return [generate_synthetic_obfuscation(n) for n in ast_node]
    if not isinstance(ast_node, dict):
        return ast_node
    new_node = ast_node.copy()
    node_type = new_node.get("type")
    if node_type == "Identifier":
        new_node["name"] = random.choice("abcdef")
    if "children" in new_node:
        new_node["children"] = [generate_synthetic_obfuscation(c) for c in new_node["children"]]
    return new_node

test_ast = {
    "type": "BlockStatement",
    "children": [
        {"type": "Identifier", "name": "foo"},
        {"type": "Identifier", "name": "bar"}
    ]
}

print("Running test...")
for i in range(100):
    res = generate_synthetic_obfuscation(test_ast)
print("Test complete.")
