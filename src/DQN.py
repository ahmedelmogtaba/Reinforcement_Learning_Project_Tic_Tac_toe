import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, outputs=9, num_embeddings=3, embedding_dim=1):
        super(DQN, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.outputs = outputs
        # self.emb = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)
        #[0, 2, 0, 1, 0, 0, 0, 0, 0] ==> 9x16
        
        self.linear1 = nn.Linear( 9,  64) #16*9 x 16*9
        self.linear2 = nn.Linear( 64, 32)
        self.output = nn.Linear(32, self.outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = self.emb(x)
        # x = x.view(-1,9*self.embedding_dim)
        x = x.float()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.output(x)