import torch
import torch.nn as nn
import numpy as np
from tensorly.decomposition import tucker

import FCNet
import Attention
from Clasiffier import SimpleClassifier


class BaseModel(nn.Module):
    def __init__(self, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, q):
        """Forward
          v: [batch, T, d]
          q: [batch, q_dim]
          return: logits, not probs
        """
        b_size = len(v)

        q_emb = q
        v = v.reshape(b_size, 16, 128)
        att = self.v_att(v, q).squeeze()

        for batch in range(b_size):
          i = 0
          for region in v[batch]:
            region = att[batch][i] * region
            i += 1
        
        v = v.reshape(b_size, 2048)

        # get core tensors
        # core_list = []
        # for batch in range(b_size):
        #   tensor_dot = np.tensordot(v.detach().numpy()[batch], q.detach().numpy()[batch], 0)
        #   core, factors = tucker(np.array(tensor_dot), rank=[1, 384])
        #   core_list.append(core)

        q = self.q_net(q)
        v = self.v_net(v)
        # joint_repr = q * v
        # logits = self.classifier(torch.from_numpy(np.array(core_list)).float())
        logits = self.classifier(q*v)

        return logits


def build_baseline_model():
    # img [1, 16, 128]
    # que [1, 384]
    v_att = Attention(128, 384, 512).cuda()
    classifier = SimpleClassifier(512, 3000).cuda()
    q_net = FCNet([384, 512]).cuda()
    v_net = FCNet([2048, 512]).cuda()

    return BaseModel(v_att, q_net, v_net, classifier)