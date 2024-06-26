"""
Description: This file is used to define the Error Passing Network model.
"""
from typing import List, Optional, Tuple

import torch
from torch import cat, eye, matmul, no_grad, sqrt, Tensor, unique
from torch.nn import Linear, MSELoss
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from src.data.processing.datasets import MaskType, PetaleDataset
from src.evaluation.early_stopping import EarlyStopper
from src.models.abstract_models.custom_torch_base import TorchCustomModel
from src.models.wrappers.torch_wrappers import TorchRegressorWrapper
from src.utils.hyperparameters import HP, NumericalContinuousHP, NumericalIntHP
from src.utils.metrics import RootMeanSquaredError


class SimilarityKernel:
    """
    Stores the constant related to mask types
    """
    ATTENTION: str = "attention"
    DOT: str = "dot_product"
    COSINE: str = "cosine"

    def __iter__(self):
        return iter([self.ATTENTION, self.DOT, self.COSINE])


class EPN(TorchCustomModel):
    """
    Error Passing Network model
    """

    def __init__(self,
                 previous_pred_idx: int,
                 pred_mu: float,
                 pred_std: float,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 verbose: bool = False,
                 similarity_kernel: str = SimilarityKernel.ATTENTION,
                 n_neighbors: int = None):

        # We call parent's constructor
        super().__init__(criterion=MSELoss(),
                         criterion_name='MSE',
                         eval_metric=RootMeanSquaredError(),
                         output_size=1,
                         alpha=alpha,
                         beta=beta,
                         num_cont_col=num_cont_col,
                         cat_idx=cat_idx,
                         cat_sizes=cat_sizes,
                         cat_emb_sizes=cat_emb_sizes,
                         additional_input_args=None,
                         verbose=verbose)

        # Index indicating which column of the dataset is associated to previous
        # predictions made by another model
        self._prediction_idx = previous_pred_idx

        # We set variables needed to set back predictions in their original scale
        self._pred_mu = pred_mu
        self._pred_std = pred_std

        # Key and Query projection layers
        self._key_projection = None
        self._query_projection = None

        # Scaling factor
        self._dk = sqrt(Tensor([self._input_size]))

        # Attention map cache
        self._attn_cache = None

        # Number of neighbors and similarity metric
        self._n_neighbors = n_neighbors
        self._similarity_kernel = similarity_kernel
        self._compute_similarity = self._define_similarity_kernel(similarity_kernel)

    @property
    def attn_cache(self) -> Tensor:
        return self._attn_cache

    def _define_similarity_kernel(self, similarity_kernel):
        if similarity_kernel == SimilarityKernel.ATTENTION:
            # Key and Query projection layers
            # We decrease the input size by one because one column contains predicted targets and will be removed
            self._key_projection = Linear(self._input_size - 1, self._input_size)
            self._query_projection = Linear(self._input_size - 1, self._input_size)
            # Use the whole dataset when using attention-based kernel similarity
            self._n_neighbors = None

            def compute_similarity(x1: Tensor, x2: Tensor) -> Tensor:
                return matmul(self._key_projection(x1), self._query_projection(x2).t()) / self._dk
        elif similarity_kernel == SimilarityKernel.DOT:
            def compute_similarity(x1: Tensor, x2: Tensor) -> Tensor:
                return matmul(x1, x2.t())
        elif similarity_kernel == SimilarityKernel.COSINE:
            def compute_similarity(x1: Tensor, x2: Tensor) -> Tensor:
                return matmul(x1 / torch.norm(x1, dim=1, keepdim=True),
                              (x2 / torch.norm(x2, dim=1, keepdim=True)).t())
        else:
            raise ValueError(f'Similarity kernel: {similarity_kernel} not implemented')

        return compute_similarity

    def _execute_valid_step(self,
                            valid_data: Tuple[Optional[DataLoader], PetaleDataset],
                            early_stopper: EarlyStopper) -> bool:
        """
        Executes an inference step on the validation data

        Args:
            valid_data: dataset

        Returns: True if we need to early stop
        """
        if valid_data[0] is None:
            return False

        # We extract the valid dataloader and the complete dataset
        valid_loader, dataset = valid_data

        # Set model for evaluation
        self.eval()
        epoch_loss, epoch_score = 0, 0

        # We extract the data of training and valid set
        x, y, all_idx = dataset[dataset.train_mask + dataset.valid_mask]

        # We execute one inference step on validation set
        with no_grad():

            for _, _, idx in valid_loader:
                # We perform the forward pass
                batch_pos_idx = [all_idx.index(i) for i in idx]
                output = self(x, y, batch_pos_idx)

                # We calculate the loss and the score
                epoch_loss += self.loss(output, y[batch_pos_idx]).item()
                epoch_score += self._eval_metric(output, y[batch_pos_idx])

        # We update evaluations history
        mean_epoch_score = self._update_evaluations_progress(epoch_loss, epoch_score,
                                                             nb_batch=len(valid_loader),
                                                             mask_type=MaskType.VALID)

        # We check early stopping status
        early_stopper(mean_epoch_score, self)

        if early_stopper.early_stop:
            return True

        return False

    def _execute_train_step(self, train_data: Tuple[Optional[DataLoader], PetaleDataset]) -> float:
        """
        Executes one training epoch

        Args:
            train_data: dataset

        Returns: mean epoch loss
        """
        # We set the model for training
        self.train()
        epoch_loss, epoch_score = 0, 0

        # We extract the valid dataloader and the complete dataset
        train_loader, dataset = train_data

        # We extract the features related to all the train mask
        x, y, all_idx = dataset[dataset.train_mask]

        for _, _, idx in train_loader:
            # We clear the gradients
            self._optimizer.zero_grad()

            # We perform the weight update
            batch_pos_idx = [all_idx.index(i) for i in idx]
            pred, loss = self._update_weights([x, y, batch_pos_idx], y[batch_pos_idx])

            # We update the metrics history
            score = self._eval_metric(pred, y[batch_pos_idx])
            epoch_loss += loss
            epoch_score += score

        # We update evaluations history
        mean_epoch_loss = self._update_evaluations_progress(epoch_loss, epoch_score,
                                                            nb_batch=len(train_loader),
                                                            mask_type=MaskType.TRAIN)
        return mean_epoch_loss

    def forward(self,
                x: Tensor,
                y: Tensor,
                test_idx: Optional[List[int]] = None) -> Tensor:
        """
        Executes a forward pass.

        Args:
            x: (N, D) tensor with features
            y: (N, 1) tensor with targets
            test_idx: List of idx associated to test data points for which we want to calculate smooth targets.
                      If None, values are returned for all idx.

        Returns: (N, 1) tensor with smoothed targets
        """

        # We extract previous prediction made by another model
        y_hat = (x[:, self._prediction_idx] * self._pred_std) + self._pred_mu

        # We calculate the errors made on these predictions
        errors = y - y_hat
        # We initialize a list of tensors to concatenate
        new_x = []

        # We extract continuous data
        if len(self._cont_idx) != 0:
            new_x.append(x[:, [i for i in self._cont_idx if i != self._prediction_idx]])

        # We perform entity embeddings on categorical features
        if len(self._cat_idx) != 0:
            new_x.append(self._embedding_block(x))

        # We concatenate all inputs
        x = cat(new_x, 1)

        if test_idx is None:

            # We compute the scaled-dot product
            att = self._compute_similarity(x, x)

            # We set the diagonal to zero
            att = att * (1 - eye(att.shape[0], att.shape[1]))

            # Get the k neighbors
            if self._n_neighbors != None:
                att, indices = torch.topk(att, k=max(round(self._n_neighbors*att.shape[1]), 1), dim=-1)
                errors = errors[indices]

            else:
                # Reshape the dataset to len(x) * number of total nodes
                errors = errors.unsqueeze(1).repeat(1, len(x)).permute(1, 0)

            # We apply the softmax
            att = softmax(att, dim=-1)

        else:

            # We compute the scaled-dot product
            att = self._compute_similarity(x[test_idx, :], x)

            if not self.training:

                # We set the attention given to all test elements to zero
                # Therefore test elements cannot attend to errors of other test elements (including their own)
                att[:, test_idx] = 0

            else:

                # We only make sure that self attention value of test elements are zeroed
                att[range(len(test_idx)), test_idx] = 0

            # Get the k neighbors
            if self._n_neighbors != None:
                att, indices = torch.topk(att, k=max(round(self._n_neighbors*att.shape[1]), 1), dim=1)
                errors = errors[indices]

            else:
                # Reshape the dataset to test_idx * number of total nodes
                errors = errors.unsqueeze(1).repeat(1, len(test_idx)).permute(1, 0)

            # We apply the softmax
            self._attn_cache = att = softmax(att, dim=-1)

            # We only keep predictions previously made for test idx
            y_hat = y_hat[test_idx]
        return (matmul(att, errors.t())[:, 0] + y_hat).squeeze(dim=-1)

    def predict(self,
                dataset: PetaleDataset,
                mask: Optional[List[int]] = None) -> Tensor:
        """
        Returns the real-valued predictions for all samples
        in a particular set (default = test)

        Args:
            dataset: PetaleDataset which its items are tuples (x, y, idx) where
                     - x : (N,D) tensor with D-dimensional samples
                     - y : (N,) tensor with classification labels
                     - idx : (N,) tensor with idx of samples according to the whole dataset
            mask: list of dataset idx for which we want to predict target

        Returns: (N,) tensor
        """

        # Set mask value if not provided
        if mask is None:
            mask = dataset.test_mask

        # Set model for evaluation
        self.eval()

        # We look for the idx that are not in the training set
        added_idx = [i for i in mask if i not in dataset.train_mask]

        # Execute a forward pass and apply a softmax
        with no_grad():
            x, y, idx = dataset[dataset.train_mask + added_idx]
            if len(added_idx) > 0:
                return self(x, y, [idx.index(i) for i in added_idx])
            else:
                return self(x, y)


class PetaleEPN(TorchRegressorWrapper):
    """
    Error Passing Network wrapper for the Petale framework
    """

    def __init__(self,
                 previous_pred_idx: int,
                 pred_mu: float,
                 pred_std: float,
                 alpha: float = 0,
                 beta: float = 0,
                 num_cont_col: Optional[int] = None,
                 cat_idx: Optional[List[int]] = None,
                 cat_sizes: Optional[List[int]] = None,
                 cat_emb_sizes: Optional[List[int]] = None,
                 lr: float = 0.05,
                 rho: float = 0,
                 batch_size: Optional[int] = None,
                 max_epochs: int = 200,
                 patience: int = 15,
                 verbose: bool = False,
                 similarity_kernel: str = SimilarityKernel.ATTENTION,
                 n_neighbors: int = None):
        # Creation of the model
        model = EPN(previous_pred_idx=previous_pred_idx,
                    pred_mu=pred_mu,
                    pred_std=pred_std,
                    alpha=alpha,
                    beta=beta,
                    num_cont_col=num_cont_col,
                    cat_idx=cat_idx,
                    cat_sizes=cat_sizes,
                    cat_emb_sizes=cat_emb_sizes,
                    verbose=verbose,
                    similarity_kernel=similarity_kernel,
                    n_neighbors=n_neighbors)

        # Call of parent's constructor
        # Valid batch size is set to None to use full batch at once
        super().__init__(model=model,
                         train_params=dict(lr=lr,
                                           rho=rho,
                                           batch_size=batch_size,
                                           valid_batch_size=None,
                                           patience=patience,
                                           max_epochs=max_epochs,
                                           include_dataset=True))

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model

        Returns: list of hyperparameters
        """
        return list(EPNHP())


class EPNHP:
    """
    EPN hyperparameters
    """
    ALPHA = NumericalContinuousHP("alpha")
    BATCH_SIZE = NumericalIntHP("batch_size")
    BETA = NumericalContinuousHP("beta")
    LR = NumericalContinuousHP("lr")
    RHO = NumericalContinuousHP("rho")
    NEIGHBORS = NumericalIntHP("n_neighbors")

    def __iter__(self):
        return iter([self.ALPHA, self.BATCH_SIZE, self.BETA, self.LR, self.RHO, self.NEIGHBORS])
