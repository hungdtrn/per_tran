from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from typing import Optional, Sequence, Union
from library.distribution import masked_softmax

def pick_activation_function(activation: Union[str, dict]):
    """Pick an activation function based on its name.

    Arguments:
        activation - Either a string or a dictionary. If a string, it specifies the function name, e.g. 'relu'. If a
            dictionary, it must contain a key named 'name' mapping to the function name, e.g. 'relu', and other
            key-value pairs in the dictionary are arguments for the function. For instance,
            {'name': 'logsoftmax', 'dim': -1}.
    Returns:
        An instantiated object of the respective PyTorch function.
    """
    name_to_fnclass = {
        'identity': torch.nn.Identity,
        'logsigmoid': torch.nn.LogSigmoid,
        'logsoftmax': torch.nn.LogSoftmax,
        'relu': torch.nn.ReLU,
        'leakyrelu': torch.nn.LeakyReLU,
        'prelu': torch.nn.PReLU,
        'sigmoid': torch.nn.Sigmoid,
        'softmax': torch.nn.Softmax,
        'softplus': torch.nn.Softplus,
        'tanh': torch.nn.Tanh,
        'elu': torch.nn.ELU,
    }
    try:
        activation_name = activation.get('name')
    except AttributeError:
        activation_name, kwargs = activation, {}
    else:
        del activation['name']
        kwargs = activation
    activation_obj = name_to_fnclass[activation_name.lower()](**kwargs)
    return activation_obj
    

def build_mlp(dims: Sequence[int], activations: Optional[Sequence[Union[str, dict]]] = None,
              dropout: float = 0.0, bias: bool = True):
    """Build a general Multi-layer Perceptron (MLP).

    Arguments:
        dims - An iterable containing the sequence of input/hidden/output dimensions. For instance, if
            dims = [256, 128, 64], our MLP receives input features of dimension 256, reduces it to 128, and outputs
            features of dimension 64.
        activations - An iterable containing the activations of each layer of the MLP. Each element of the iterable can
            be either a string or a dictionary. If it is a string, it specifies the name of the activation function,
            such as 'relu'; if it is a dictionary, it should contain a name key, and optional keyword arguments for the
            function. For instance, a valid input could be ['relu', {'name': 'logsoftmax', 'dim': -1}]. If activations
            is None, no activation functions are applied to the outputs of the layers of the MLP.
        dropout - Dropout probability.
        bias - Whether to include a bias term in the linear layers or not.
    Returns:
        An MLP as a PyTorch Module.
    """
    if activations is None:
        activations = ['identity'] * (len(dims) - 1)
    if len(dims) - 1 != len(activations):
        raise ValueError('Number of activations must be the same as the number of dimensions - 1.')
    layers = []
    for i, (dim_in, dim_out, activation) in enumerate(zip(dims[:-1], dims[1:], activations)):
        layers.append(nn.Linear(dim_in, dim_out, bias=bias))
        layers.append(pick_activation_function(activation))
            
        if dropout:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def softmax(z, mask=None, dim=1):
    if mask is not None:
        w = masked_softmax(z, mask, dim)
    else:
        w = F.softmax(z, dim=dim)
    
    return w


def attention(query, values, mask=None, attention_fc=None, msg_fc=None):
        """ Perform masked scaled dot product attention

        Args:
            key ([type]): tensor of shape (batch, num_neighbor, num_graph, dim)  or (batch, num_neighbor, dim)
            values ([type]): tensor of shape (batch, num_neighbor, num_graph, dim) or (batch, num_neighbor, dim)
            mask ([type]): tensor of shape (batch, num_neighbor, num_graph) or (batch, num_neighbor)
        """
        batch, num_neighbor, d = values.size(0), values.size(1), values.size(-1)
        num_dim = len(query.size())
        
        if num_dim == 2:
            query = query.unsqueeze(1).repeat(1, num_neighbor, 1)
        elif num_dim == 3:
            query = query.unsqueeze(1).repeat(1, num_neighbor, 1, 1)

        feat = attention_fc(torch.cat([query, values], -1)).squeeze(-1)
        
        if mask is not None:
            mask = masked_softmax(feat, mask, 1)
        else:
            mask = F.softmax(feat, dim=1)
            
        if num_dim == 3:
            w = mask.unsqueeze(-1).repeat(1, 1, 1, d)
        else:
            w = mask.unsqueeze(-1).repeat(1, 1, d)
        
        return msg_fc(torch.sum(w * values, 1)), mask

class HeteGAT(nn.Module):
    def __init__(self, etype, hidden_dim, attn_hidden_dim, attn_act, sem_act='tanh', num_head=1):
        super().__init__()
        
        self.edges = nn.ModuleDict()
        for t in etype:
            src_dim, dst_dim = etype[t]
            self.edges[",".join(list(t))] = MultiheadGat(num_head, t, src_dim, dst_dim, hidden_dim, attn_hidden_dim, attn_act)
        
        act = pick_activation_function(sem_act)        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
    def forward(self, graph):
        msgs = []
        for str_etype, gat in self.edges.items():
            etype = tuple(str_etype.split(","))
            srctype, _, dsttype = etype
            src_feat = torch.cat([graph.nodes[srctype].data["embed_feat"],
                                  graph.nodes[srctype].data["h_gru"]], -1)
            dst_feat = torch.cat([graph.nodes[dsttype].data["embed_feat"],
                                  graph.nodes[dsttype].data["h_gru"]], -1)

            msgs.append(gat(graph, src_feat, dst_feat))
                
        z = torch.stack(msgs, 1)
        w = self.projection(z).mean(0)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)

    

class GAT(nn.Module):
    def __init__(self, etype, src_dim, dst_dim, out_dim, attn_hidden_dim, attn_act, has_self_loop=True, modified=True):
        super().__init__()
        
        self.etype = etype
        srctype, _, dsttype = etype
        self.has_self_loop = has_self_loop
        
        if etype != ('human', 'interacts', 'human') or self.has_self_loop:
            if srctype != dsttype:
                self.src_fc = nn.Linear(src_dim, out_dim, bias=False)
                self.dst_fc = nn.Linear(dst_dim, out_dim, bias=False)
            else:
                self.in_fc = nn.Linear(src_dim, out_dim, bias=False)
            
            if modified:
                attn_dims = [2 * out_dim] + list(attn_hidden_dim) + [1]
                acts = [attn_act for i in range(len(attn_dims) - 1)]

                self.attn_fc = build_mlp(attn_dims, acts)
            else:
                self.attn_fc = build_mlp([2 * out_dim, 1], ["leakyrelu"], bias=False)
        else:
            self.in_fc = nn.Linear(src_dim, out_dim, bias=False)

    
    def get_name(self, name):
        return "{}_{}".format(self.name, name)
    
    def edge_attention_fn(self, mask_fn=None):
        def edge_attention(edges):
            z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
            a = self.attn_fc(z2)
            out = {'e': a}
            if mask_fn is not None:
                out['mask'] = mask_fn(edges)
            
            return out
        return edge_attention

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        out = {'z': edges.src['z'], 'e': edges.data['e']}
        if "mask" in edges.data:
            out["mask"] = edges.data["mask"]
        
        return out
    
    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        mask = None
        if "mask" in nodes.mailbox:
            mask = nodes.mailbox["mask"].unsqueeze(-1)
        
        alpha = softmax(nodes.mailbox['e'], mask=mask, dim=1)

        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}
        
    def forward(self, graph, srch, dsth, mask_fn=None, visualize=False):
        srctype, _, dsttype = self.etype
        
        if srctype != dsttype:
            graph.nodes[srctype].data['z'] = self.src_fc(srch)
            graph.nodes[dsttype].data['z'] = self.dst_fc(dsth)
        else:
            graph.nodes[srctype].data['z'] = self.in_fc(srch)
        
        graph.apply_edges(self.edge_attention_fn(mask_fn), etype=self.etype)
        
        graph.update_all(self.message_func, self.reduce_func, etype=self.etype)

        z = graph.nodes[srctype].data.pop('z')
        if srctype != dsttype:
            graph.nodes[dsttype].data.pop('z')

        return graph.nodes[dsttype].data.pop('h')

class MultiheadGat(nn.Module):
    def __init__(self, num_heads, etype, src_dim, dst_dim, out_dim, attn_hidden_dim, attn_act, has_self_loop=True, merge_type='cat'):
        super().__init__()
        
        self.heads = nn.ModuleList()
        self.etype = etype
        srctype, _, dsttype = etype

        out_dim = out_dim // num_heads
        for i in range(num_heads):
            head = nn.ModuleDict()
            
            if srctype != dsttype:
                head["src_fc"] = nn.Linear(src_dim, out_dim, bias=False)
                head["dst_fc"] = nn.Linear(dst_dim, out_dim, bias=False)
            else:
                head["in_fc"] = nn.Linear(src_dim, out_dim, bias=False)
            
            head["attn_fc"] = build_mlp([2 * out_dim, 1], [attn_act], bias=False)
            
            self.heads.append(head)
        

        
        self.merge_type = merge_type
    
    # def forward(self, graph, srch, dsth, mask_fn=None, visualize=False):
    #     head_outs = [head(graph, srch, dsth, mask_fn, visualize) for head in self.heads]
    #     if self.merge_type == 'cat':
    #         return torch.cat(head_outs, -1)
    #     else:
    #         return torch.mean(torch.stack(head_outs))
    
    def edge_attention_fn(self, mask_fn=None):
        def edge_attention(edges):
            out = {}
            for i, head in enumerate(self.heads):
                z2 = torch.cat([edges.src['z_{}'.format(i)], edges.dst['z_{}'.format(i)]], dim=1)
                a = head["attn_fc"](z2)
                out['e_{}'.format(i)] = a
                if mask_fn is not None:
                    out['mask_{}'.format(i)] = mask_fn(edges)
            
            return out
        return edge_attention

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        out = {}
        for i in range(len(self.heads)):
            out['z_{}'.format(i)] = edges.src['z_{}'.format(i)]
            out['e_{}'.format(i)] = edges.data['e_{}'.format(i)]

            if "mask_{}".format(i) in edges.data:
                out["mask_{}".format(i)] = edges.data["mask_{}".format(i)]
        
        return out

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        hs = []
        for i in range(len(self.heads)):
            mask = None
            if "mask_{}".format(i) in nodes.mailbox:
                mask = nodes.mailbox["mask_{}".format(i)].unsqueeze(-1)
            
            alpha = softmax(nodes.mailbox["e_{}".format(i)], mask=mask, dim=1)

            # equation (4)
            attended = torch.sum(alpha * nodes.mailbox["z_{}".format(i)], dim=1)
            hs.append(attended)
            
        if self.merge_type == 'cat':
            h = torch.cat(hs, -1)
        elif self.merge_type == 'stack':
            h = torch.stack(hs, 1)
            
        return {'h': h}

    
    def forward(self, graph, srch, dsth, mask_fn=None, visualize=False):
        srctype, _, dsttype = self.etype
        
        for i, head in enumerate(self.heads):
            if srctype != dsttype:
                graph.nodes[srctype].data['z_{}'.format(i)] = head["src_fc"](srch)
                graph.nodes[dsttype].data['z_{}'.format(i)] = head["dst_fc"](dsth)
            else:
                graph.nodes[srctype].data['z_{}'.format(i)] = head["in_fc"](srch)
        
        graph.apply_edges(self.edge_attention_fn(mask_fn), etype=self.etype)
        
        graph.update_all(self.message_func, self.reduce_func, etype=self.etype)

        for i, head in enumerate(self.heads):
            graph.nodes[srctype].data.pop('z_{}'.format(i))
            if srctype != dsttype:
                graph.nodes[dsttype].data.pop('z_{}'.format(i))

        return graph.nodes[dsttype].data.pop('h')
