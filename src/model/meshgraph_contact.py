import torch.nn as nn
import torch
from src.utils.utils import decompose_meshgraph_graph, copy_meshgraph_graph
from torch_geometric.data import Data
from torch_scatter import scatter_mean


def instantiate_mlp(input_size, hidden_size, output_size, layers=5, layer_norm=True):
    """
    Creates a Multi-Layer Perceptron (MLP) model.

    Args:
        input_size (int): Number of input features for the MLP.
        hidden_size (int): Number of hidden units in each hidden layer.
        output_size (int): Number of output features of the MLP.
        layers (int): Total number of layers in the MLP, including input and output layers. Default is 5.
        layer_norm (bool): Whether to apply layer normalization to the output layer. Default is True.

    Returns:
        nn.Sequential: The constructed MLP model as a sequential module.
    """

    # Initialize the module list with the first layer (input layer) and ReLU activation
    module = [nn.Linear(input_size, hidden_size), nn.ReLU()]

    # Add hidden layers with ReLU activation
    for _ in range(layers - 2):
        module.append(nn.Linear(hidden_size, hidden_size))
        module.append(nn.ReLU())

    # Add the final layer (output layer)
    module.append(nn.Linear(hidden_size, output_size))

    # Wrap the layers in a sequential container
    module = nn.Sequential(*module)

    # Apply layer normalization if specified
    if layer_norm:
        module = nn.Sequential(*module, nn.LayerNorm(normalized_shape=output_size))

    # Return the constructed MLP model
    return module



class MeshGraphNetContact(nn.Module):
    """
    A neural network module that combines an Encoder, Processor(s), and a Decoder.

    Args:
        message_passing_steps (int): Number of message-passing steps in the Processor stage.
        node_input_size (int): Dimensionality of node input features.
        edge_input_size (int): Dimensionality of edge input features.
        output_size (int): Dimensionality of the Decoder output. Default is 3.
        layers (int): Number of layers in each component (Encoder, Processor, Decoder). Default is 2.
        hidden_size (int): Number of hidden units in the intermediate layers. Default is 128.
        shared_mp (bool): Whether to share the same Processor module across all message-passing steps. Default is False.
    """

    def __init__(self, message_passing_steps, node_input_size, edge_input_size, output_size=3, layers=2,
                 hidden_size=128, shared_mp=False, mssg_flag=False):
        super().__init__()

        self.mssg_flag = mssg_flag

        # Initialize the Encoder with the given input and hidden sizes
        self.encoder = EncoderContact(node_input_size=node_input_size, edge_input_size=edge_input_size,
                               hidden_size=hidden_size, layers=layers)

        # Create a list to hold Processor modules
        processor_list = []
        if shared_mp:
            # Use a single shared Processor for all message-passing steps
            processor = MessagePassingProcessorContact(hidden_size=hidden_size, layers=int(layers))
            for _ in range(message_passing_steps):
                processor_list.append(processor)
        else:
            # Create independent Processors for each message-passing step
            for _ in range(message_passing_steps):
                processor_list.append(MessagePassingProcessorContact(hidden_size=hidden_size, layers=layers))

        # Wrap the list of Processors in a ModuleList to register them as part of the model
        self.processor_list = nn.ModuleList(processor_list)

        # Initialize the Decoder with the given hidden size and output size
        self.decoder = Decoder(hidden_size=hidden_size, output_size=output_size, layers=layers)

    def forward(self, graph):
        """
        Forward pass through the EncoderProcessorDecoder model.

        Args:
            graph (torch_geometric.data.Data): Input graph data with node and edge attributes.

        Returns:
            torch.Tensor: Output tensor after processing the graph.
        """

        # Encode the input graph into a latent representation
        mssg,output_mssg = [], []
        graph = self.encoder(graph)
        if self.mssg_flag:
            mssg.append(graph.x.cpu().detach().numpy()) # Save the latent representation for visualization.
            output_mssg.append(self.decoder(graph).cpu().detach().numpy())

        # Process the latent representation through the sequence of Processors.
        for i, model in enumerate(self.processor_list):
            graph = model(graph)
            if self.mssg_flag:
                mssg.append(graph.x.cpu().detach().numpy())
                output_mssg.append(self.decoder(graph).cpu().detach().numpy())

        # Decode the processed representation into the output format.
        graph_output = self.decoder(graph)

        # Return the final output.
        if self.mssg_flag:
            return graph_output, (mssg, output_mssg)
        else:
            return graph_output
        

class EncoderContact(nn.Module):
    """
    Encodes the input graph into a latent representation by processing node attributes.

    Args:
        node_input_size (int): Dimensionality of the input node features. Default is 128.
        hidden_size (int): Number of hidden units in the intermediate layers. Default is 128.
        layers (int): Number of layers in the MLP used for encoding. Default is 2.
    """

    def __init__(self, node_input_size=1, edge_input_size=3, hidden_size=128, layers=2):
        super(EncoderContact, self).__init__()
        # Instantiate an MLP to process node attributes.
        self.node_encoder = instantiate_mlp(node_input_size, hidden_size, hidden_size, layers=layers)
        self.edge_attr_encoder = instantiate_mlp(edge_input_size * 2, hidden_size, hidden_size, layers=layers)
        self.edge_contact_attr_encoder = instantiate_mlp(edge_input_size, hidden_size, hidden_size, layers=layers)


    def forward(self, graph):
        """
        Forward pass for the Encoder.

        Args:
            graph (torch_geometric.data.Data): Input graph data containing node attributes and edges.

        Returns:
            torch_geometric.data.Data: Graph data with updated node representations.
        """
        # Decompose the graph into its components (node attributes and edge indices).
        node_attr, edge_attr, _, edge_contact_attr, _ = decompose_meshgraph_graph(graph)
        # Apply the MLP to encode node attributes into a latent representation.
        _node_latent = self.node_encoder(node_attr)
        _edge_mesh_attr_latent = self.edge_attr_encoder(edge_attr)
        _edge_cont_attr_latent = self.edge_contact_attr_encoder(edge_contact_attr)
        # Return the updated graph data with latent node attributes.
        return Data(x=_node_latent, edge_attr=_edge_mesh_attr_latent, edge_index=graph.edge_index, 
                    edge_contact_attr=_edge_cont_attr_latent, edge_contact_index=graph.edge_contact_index)
    

class Decoder(nn.Module):
    """
    Decodes the latent representation of the graph into the desired output format.

    Args:
        hidden_size (int): Number of hidden units in the intermediate layers. Default is 128.
        output_size (int): Dimensionality of the output features. Default is 2.
        layers (int): Number of layers in the MLP used for decoding. Default is 2.
    """

    def __init__(self, hidden_size=128, output_size=2, layers=2):
        super(Decoder, self).__init__()
        # Instantiate an MLP to decode the latent representation into the output format.
        self.decoder_model = instantiate_mlp(hidden_size, hidden_size, output_size, layers=layers, layer_norm=False)

    def forward(self, graph):
        """
        Forward pass for the Decoder.

        Args:
            graph (torch_geometric.data.Data): Input graph data containing latent node attributes.

        Returns:
            torch.Tensor: Decoded output tensor.
        """
        # Apply the MLP to decode the latent node attributes into the output features.
        output = self.decoder_model(graph.x)
        # Return the decoded output.
        return output


class MessagePassingProcessorContact(nn.Module):
    """
    Processes the graph by aggregating node information and updating node attributes.

    Args:
        hidden_size (int): Number of hidden units in the intermediate layers. Default is 128.
        layers (int): Number of layers in the MLP used for processing nodes. Default is 2.
    """

    def __init__(self, hidden_size=128, layers=2):
        super(MessagePassingProcessorContact, self).__init__()
        # The input dimension for the processor is twice the hidden size (concatenation of node attributes).
        node_latent_input_dim = 3 * hidden_size
        edge_latent_input_dim = 3 * hidden_size
        # Instantiate an MLP to process the concatenated node attributes.
        node_processor_mlp = instantiate_mlp(node_latent_input_dim, hidden_size, hidden_size, layers=layers)
        edge_mesh_attr_processor_mlp = instantiate_mlp(edge_latent_input_dim, hidden_size, hidden_size, layers=layers)
        edge_contact_attr_processor_mlp = instantiate_mlp(edge_latent_input_dim, hidden_size, hidden_size, layers=layers)

        # Initialize the node processing module with the MLP.
        self.node_processor_module = NodeProcessorContactModule(mlp_model=node_processor_mlp)
        self.edge_mesh_attr_processor_module = EdgeMeshProcessorContactModule(mlp_model=edge_mesh_attr_processor_mlp)
        self.edge_contact_attr_processor_module = EdgeContactProcessorContactModule(mlp_model=edge_contact_attr_processor_mlp)


    def forward(self, graph):
        """
        Forward pass for the Processor.

        Args:
            graph (torch_geometric.data.Data): Input graph data containing latent node attributes.

        Returns:
            torch_geometric.data.Data: Graph data with updated node representations after processing.
        """
        # Create a copy of the graph to retain the previous node attributes.
        graph_last = copy_meshgraph_graph(graph)
        # Update the node and edge attributes Skip-Connections
        graph = self.edge_mesh_attr_processor_module(graph)
        graph = self.edge_contact_attr_processor_module(graph)
        graph = self.node_processor_module(graph)
        # Combine the updated node attributes with the previous attributes (residual connection).
        _node_attr = graph_last.x + graph.x
        _edge_mesh_attr = graph_last.edge_attr + graph.edge_attr
        _edge_contact_attr = graph_last.edge_contact_attr + graph.edge_contact_attr
        # Return the updated graph with the new node attributes.
        return Data(x=_node_attr, edge_attr=_edge_mesh_attr, edge_index=graph.edge_index, 
                    edge_contact_attr=_edge_contact_attr, edge_contact_index=graph.edge_contact_index)


class NodeProcessorContactModule(nn.Module):
    """
    A module to process nodes by aggregating messages from connected nodes and updating attributes.

    Args:
        mlp_model (nn.Module): An MLP model used to update node attributes.
    """

    def __init__(self, mlp_model=None):
        super(NodeProcessorContactModule, self).__init__()
        self.model = mlp_model

    def forward(self, graph):
        """
        Forward pass for the NodeProcessorModule.

        Args:
            graph (torch_geometric.data.Data): Input graph data with node and edge attributes.

        Returns:
            torch_geometric.data.Data: Graph data with updated node attributes.
        """
        # Decompose the graph into node attributes and edge indices.
        node_attr, edge_attr, edge_index, edge_contact_attr, edge_contact_index = decompose_meshgraph_graph(graph)
        node_messages_to_aggregate = []  # List to aggregate information for each node.
        # Mesh sender and receiver indices from edges.
        _, receivers_idx = graph.edge_index  
        # Contact sender and receiver indices from edges.
        _, receivers_contact_idx = graph.edge_contact_index
        num_nodes = graph.num_nodes  # Total number of nodes in the graph.

        # Aggregate messages from edges to nodes receivers
        aggregated_messages_from_edges_mesh = scatter_mean(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)
        aggregated_messages_from_edges_contact = scatter_mean(edge_contact_attr, receivers_contact_idx, dim=0, dim_size=num_nodes)

        # Add the current node attributes to the list of collected information.
        node_messages_to_aggregate.append(node_attr)
        # Add the aggregated messages to process mesh information.
        node_messages_to_aggregate.append(aggregated_messages_from_edges_mesh)
        # Add the aggregated messages to process mesh information.
        node_messages_to_aggregate.append(aggregated_messages_from_edges_contact)

        # Concatenate collected information for each node.
        concatenated_messages = torch.cat(node_messages_to_aggregate, dim=-1)
        # Use the MLP model to update node attributes.
        _node_attr = self.model(concatenated_messages)

        # Return the updated graph with new node attributes.
        return Data(x=_node_attr, edge_attr=edge_attr, edge_index=edge_index, 
                    edge_contact_attr=edge_contact_attr, edge_contact_index=edge_contact_index)


class EdgeMeshProcessorContactModule(nn.Module):
    """
    A module to process nodes by aggregating messages from connected nodes and updating attributes.

    Args:
        mlp_model (nn.Module): An MLP model used to update node attributes.
    """

    def __init__(self, mlp_model=None):
        super(EdgeMeshProcessorContactModule, self).__init__()
        self.model = mlp_model

    def forward(self, graph):
        """
        Forward pass for the NodeProcessorModule.

        Args:
            graph (torch_geometric.data.Data): Input graph data with node and edge attributes.

        Returns:
            torch_geometric.data.Data: Graph data with updated node attributes.
        """
        # Decompose the graph into node attributes and edge indices.
        node_attr, edge_attr, edge_index, edge_contact_attr, edge_contact_index = decompose_meshgraph_graph(graph)
        edge_messages_to_aggregate = []  # List to collect information for each node.
        senders_idx, receivers_idx = graph.edge_index  # Extract sender and receiver indices from edges.

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edge_messages_to_aggregate.append(senders_attr)
        edge_messages_to_aggregate.append(receivers_attr)
        edge_messages_to_aggregate.append(edge_attr)

        edge_messages = torch.cat(edge_messages_to_aggregate, dim=1)

        _edge_mesh_attr = self.model(edge_messages)  # Update

        # Return the updated graph with new node attributes.
        return Data(x=node_attr, edge_attr=_edge_mesh_attr, edge_index=edge_index, 
                    edge_contact_attr=edge_contact_attr, edge_contact_index=edge_contact_index)
    

class EdgeContactProcessorContactModule(nn.Module):
    """
    A module to process nodes by aggregating messages from connected nodes and updating attributes.

    Args:
        mlp_model (nn.Module): An MLP model used to update node attributes.
    """

    def __init__(self, mlp_model=None):
        super(EdgeContactProcessorContactModule, self).__init__()
        self.model = mlp_model

    def forward(self, graph):
        """
        Forward pass for the NodeProcessorModule.

        Args:
            graph (torch_geometric.data.Data): Input graph data with node and edge attributes.

        Returns:
            torch_geometric.data.Data: Graph data with updated node attributes.
        """
        # Decompose the graph into node attributes and edge indices.
        node_attr, _, _, edge_contact_attr, edge_contact_index = decompose_meshgraph_graph(graph)
        edge_messages_to_aggregate = []  # List to collect information for each node.
        senders_idx, receivers_idx = graph.edge_contact_index  # Extract sender and receiver indices from edges.

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edge_messages_to_aggregate.append(senders_attr)
        edge_messages_to_aggregate.append(receivers_attr)
        edge_messages_to_aggregate.append(edge_contact_attr)

        edge_messages = torch.cat(edge_messages_to_aggregate, dim=1)

        _edge_contact_attr = self.model(edge_messages)  # Update

        # Return the updated graph with new node attributes.
        return Data(x=node_attr, edge_attr=graph.edge_attr, edge_index=graph.edge_index, 
                    edge_contact_attr=_edge_contact_attr, edge_contact_index=edge_contact_index)
