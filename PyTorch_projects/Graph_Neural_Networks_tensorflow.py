#bash
#pip install tensorflow tensorflow_gnn

#Define Graph Schema

import tensorflow_gnn as tfgnn

schema = """
node_sets {
  key: "atom"  # or "user" for social networks
  value {
    features {
      key: "features"  # e.g., atomic number or user profile
      value { dtype: DT_FLOAT shape { dim { size: 128 } } }
    }
  }
}
edge_sets {
  key: "bond"  # or "connection" for social networks
  value {
    source: "atom"
    target: "atom"
  }
}
"""
graph_schema = tfgnn.parse_schema(schema)


#Define Graph Convolutional Layers
from tensorflow_gnn import GraphTensor
from tensorflow_gnn.keras.layers import GCNConv, GraphUpdate

def build_gnn_model(graph_schema: tfgnn.GraphTensorSpec):
    inputs = tf.keras.Input(type_spec=graph_schema)
    graph = inputs

    # Initial feature mapping
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn={"atom": tf.keras.layers.Dense(64)})(graph)

    # Graph convolution layers
    for _ in range(3):  # 3 message-passing steps
        graph = GraphUpdate(
            node_sets={
                "atom": GCNConv(
                    units=64,
                    receiver_tag=tfgnn.TARGET,
                    activation="relu"
                )
            }
        )(graph)

    # Pool to graph-level output for property prediction
    pooled = tfgnn.keras.layers.Pool(
        context_name="atom", reduce_type="mean")(graph)
    
    # Prediction head
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(pooled)
    return tf.keras.Model(inputs, outputs)

#Convert Data to GraphTensor Format
python
def create_graph_tensor(node_features, edge_list):
    return tfgnn.GraphTensor.from_pieces(
        node_sets={
            "atom": tfgnn.NodeSet.from_fields(
                sizes=[node_features.shape[0]],
                features={"features": node_features}
            )
        },
        edge_sets={
            "bond": tfgnn.EdgeSet.from_fields(
                sizes=[edge_list.shape[0]],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("atom", edge_list[:, 0]),
                    target=("atom", edge_list[:, 1])
                )
            )
        }
    )


node_features = tf.random.uniform((5, 128))  # Atomic features
edge_list = tf.constant([[0,1], [1,2], [2,3], [3,4]])  # Bond connections
graph_tensor = create_graph_tensor(node_features, edge_list)

# Initialize model
model = build_gnn_model(graph_tensor.spec)
model.compile(optimizer="adam", loss="binary_crossentropy")

# Create dataset (replace with real molecular/social network data)
dataset = tf.data.Dataset.from_tensors(
    (graph_tensor, tf.constant([1.0]))  # Graph + label
).batch(32)


model.fit(dataset, epochs=10)
# Node Classification

def node_classification_model(graph_spec):
    inputs = tf.keras.Input(type_spec=graph_spec)
    graph = inputs

    # Feature transformation
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn={"user": tf.keras.layers.Dense(64)})(graph)

    # GNN layers
    for _ in range(2):
        graph = GraphUpdate(
            node_sets={
                "user": GCNConv(units=64, receiver_tag=tfgnn.TARGET)
            }
        )(graph)

    # Node-level predictions
    user_features = tfgnn.keras.layers.Readout(
        node_set_name="user")(graph)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(user_features)
    
    return tf.keras.Model(inputs, outputs)
#Advanced Techniques

graph_update = GraphUpdate(
    node_sets={
        "user": GCNConv(units=64),
        "post": GCNConv(units=32)  # Additional node type
    },
    edge_sets={
        "follows": tfgnn.keras.layers.SimpleConv(
            sender_edge_feature="weight",  # Edge features
            receiver_tag=tfgnn.TARGET
        )
    }
)

# Create batched dataset
dataset = tf.data.Dataset.from_tensors(graph_tensor).batch(32)

# Enable padding for variable-sized graphs
padded_spec = tfgnn.GraphTensorSpec.from_piece_specs(
    context_spec=tfgnn.ContextSpec.from_field_specs(
        features={"label": tf.TensorSpec([None], tf.float32)}
    )
)
dataset = dataset.padded_batch(32, padded_shapes=padded_spec)