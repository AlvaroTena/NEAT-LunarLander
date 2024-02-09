from typing import List

import NEAT.feed_forward_nn as nn
import networkx as nx


class ForAnimationFormatterFn:
    def __init__(self, genome: nn.gen.Genome):
        self.genome = genome

    def __call__(self):
        return "{}\n{}\n{}\n{}\n".format(
            self.nodes(self.genome.make_input_ids()),
            self.nodes(self.hidden_nodes()),
            self.nodes(self.genome.make_output_ids()),
            self.format_edges(),
        )

    def nodes(self, nodes: List[int]):
        return " ".join(map(str, nodes))

    def format_edges(self):
        edges_str = "\n".join(
            "{} {} {} {}".format(
                link.link_id.input_id,
                link.link_id.output_id,
                link.is_enabled,
                link.weight,
            )
            for link in self.genome.links()
        )
        return edges_str

    def hidden_nodes(self) -> List[int]:
        return [
            neuron.neuron_id
            for neuron in self.genome.neurons()
            if neuron.neuron_id >= self.genome.num_outputs()
        ]

    def to_networkx_graph(self):
        G = nx.DiGraph()

        # Agregar nodos de entrada con layer=0
        for node_id in self.genome.make_input_ids():
            G.add_node(node_id, layer=0)

        # Agregar nodos de salida con layer=2
        for node_id in self.genome.make_output_ids():
            G.add_node(node_id, layer=2)

        # Agregar aristas
        for link in self.genome.links():
            if link.is_enabled:
                G.add_edge(
                    link.link_id.input_id, link.link_id.output_id, weight=link.weight
                )

        # Identificar y agregar nodos ocultos
        hidden_nodes = self.hidden_nodes()
        for node_id in hidden_nodes:
            G.add_node(node_id, layer=None)

        # Inferir las capas ocultas basándose en la distancia desde los nodos de entrada
        for hidden_node in hidden_nodes:
            # Encuentra el camino más largo desde cualquier nodo de entrada al nodo oculto
            max_distance = max(
                (
                    nx.shortest_path_length(G, source=input_node, target=hidden_node),
                    input_node,
                )
                for input_node in self.genome.make_input_ids()
                if nx.has_path(G, input_node, hidden_node)
            )[0]

            # Establece la capa del nodo oculto en base a la distancia máxima + 1
            G.nodes[hidden_node]["layer"] = max_distance + 1

        # Asegúrate de que todos los nodos de salida están en la capa final
        max_hidden_layer = max(G.nodes[node]["layer"] for node in hidden_nodes)
        for output_node in self.genome.make_output_ids():
            G.nodes[output_node]["layer"] = max_hidden_layer + 1

        return G


class DotFormatterFn:
    def __init__(self, genome: nn.gen.Genome, print_zero_links=True):
        self.genome = genome
        self.print_zero_links = print_zero_links

    def __call__(self):
        subgraph_inputs = self.subgraph("inputs", "blue4", self.genome.make_input_ids())
        subgraph_hidden = self.subgraph("hidden", "red2", self.hidden_nodes())
        subgraph_outputs = self.subgraph(
            "outputs", "seagreen2", self.genome.make_output_ids()
        )
        edges = self.format_edges()
        return "digraph G {{\n {} {} {} {} }}".format(
            subgraph_inputs, subgraph_hidden, subgraph_outputs, edges
        )

    def format_edges(self):
        edges_str = "\n".join(
            "{} -> {};".format(link.link_id.input_id, link.link_id.output_id)
            for link in self.genome.links()
            if not (link.weight == 0.0 and not self.print_zero_links)
        )
        return edges_str

    def hidden_nodes(self) -> List[int]:
        return [
            neuron.neuron_id
            for neuron in self.genome.neurons()
            if neuron.neuron_id >= self.genome.num_outputs()
        ]

    def subgraph(self, name: str, node_color: str, nodes: List[int]):
        if not nodes:
            return ""
        nodes_str = " ".join(map(str, nodes))
        return (
            "subgraph {} {{\n"
            "color=white;\n"
            "node [style=solid,color={}, shape=circle];\n"
            "{};\n"
            "}}".format(name, node_color, nodes_str)
        )
