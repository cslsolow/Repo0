class Node:
    def __init__(self, name, description: str = ""):
        self.name = name
        self.descriptions = description
        self.edges = []

    def add_edge(self, node):
        self.edges.append(node)
    def __repr__(self):
        return f"Node({self.name})"

class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node_name, description: str = ""):
        if node_name not in self.nodes:
            self.nodes[node_name] = Node(node_name, description)

    def get_node(self, node_name):
        return self.nodes.get(node_name)

    def add_edge(self, from_node_name, to_node_name):
        from_node = self.get_node(from_node_name)
        to_node = self.get_node(to_node_name)
        
        if from_node and to_node:
            from_node.add_edge(to_node)
        else:
            raise ValueError(f"One or both of the nodes {from_node_name}, {to_node_name} do not exist.")

    def has_cycle(self):
        visited = set()
        rec_stack = set()

        def dfs(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)
            for neighbor in node.edges:
                if dfs(neighbor):
                    return True
            rec_stack.remove(node)
            return False

        for node in self.nodes.values():
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def topological_sort(self):
        if self.has_cycle():
            raise ValueError("Graph contains a cycle. Topological sort not possible.")
        
        visited = set()
        result = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in node.edges:
                dfs(neighbor)
            result.append(node)

        for node in self.nodes.values():
            if node not in visited:
                dfs(node)

        return [node.name for node in result[::-1]]

    def print_graph(self):
        lines = []
        for node in self.nodes.values():
            lines.append(f"{node.name} -> {', '.join([neighbor.name for neighbor in node.edges])}")
        return "\n".join(lines)

    def to_dot(self):
        lines = ["digraph G {"]
        # 声明所有节点
        for node in self.nodes.values():
            safe_name = node.name.replace('"', '\\"')
            lines.append(f'    "{safe_name}";')
        # 声明所有有向边
        for node in self.nodes.values():
            from_name = node.name.replace('"', '\\"')
            for neighbor in node.edges:
                to_name = neighbor.name.replace('"', '\\"')
                lines.append(f'    "{from_name}" -> "{to_name}";')
        lines.append("}")
        return "\n".join(lines)