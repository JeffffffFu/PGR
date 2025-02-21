import torch

from pdb import set_trace as st

def get_gradient_on_X_v1(X, val_edges, val_edges_false):
	"""
	We want the total loss to be small.
	`X` is the input node embedding matrix. 
	`val_edges` and `val_edges_false` are the set of edges where we want to predict.  
	This function returns the grad on the input node embedding matrix.
	"""
	X = torch.tensor(X)
	X.requires_grad_(True)
	scores = torch.mm(X, torch.t(X))
	scores_array1 = []
	scores_array2 = []
	loss = 0
	for edge in val_edges:
		loss += scores[edge[0]][edge[1]]
		scores_array1.append(scores[edge[0]][edge[1]].item())
	for edge in val_edges_false:
		# loss -= 0.1 * scores[edge[0]][edge[1]]
		scores_array2.append(scores[edge[0]][edge[1]].item())
	loss /= (len(val_edges)+len(val_edges_false))
	loss.backward()
	return X.grad, [loss, reduce(lambda x,y : x+y, scores_array1)/len(scores_array1), reduce(lambda x,y : x+y, scores_array2)/len(scores_array2)]

def get_gradient_on_X_v2(X, edge):
	"""
	We want the score of a specific edge to be small.
	`X` is the input node embedding matrix.
	`edge` is the specific edge we want to attack and `label` indicates whether the groudtruth of this edge is 0 or 1.
	This function returns the grad on the input node embedding matrix.
	"""
	X = torch.tensor(X)
	X.requires_grad_(True)
	scores = torch.mm(X, torch.t(X))
	loss = scores[edge[0]][edge[1]]
	loss.backward()
	return X.grad, [loss, 0, 0]

def get_gradient_on_A(grad_Z, A, MF):
	"""
	`grad_Z` is the gradient on `Z`.
	`A` is a numpy array represention of the adjacency matrix of the graph.
	`MF` is the forward net defined.
	"""
	A = torch.tensor(A.data.numpy(), dtype=torch.float64)
	mask1 = (A != 0)
	Z = MF(A)
	# loss = torch.mul(grad_Z[mask1].double(), Z[mask1]).sum()
	loss = torch.mul(grad_Z.double(), Z).sum()
	loss.backward()
	# st()
	return A.grad, Z