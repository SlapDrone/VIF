import marimo

__generated_with = "0.14.12"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Real World Application
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Packages
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import math
    import torch
    from matplotlib import pyplot as plt
    import time

    # GPytorch
    import gpytorch
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import ScaleKernel, MaternKernel, InducingPointKernel
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import VariationalStrategy

    # GPBoost
    import gpboost as gpb

    # DKL-GP
    import viva
    from viva import VIVACpp as VIVA, my_train_cpp as my_train

    # Train-Test-Split
    from sklearn.model_selection import train_test_split, KFold

    # Batch
    from torch.utils.data import TensorDataset, DataLoader

    # Notebook
    import tqdm
    import tqdm.notebook as tqdm

    # For CRPS & Log-Score
    from scipy.stats import norm
    from scipy.special import erf


    from sklearn.cluster import KMeans
    return (
        ApproximateGP,
        CholeskyVariationalDistribution,
        ConstantMean,
        InducingPointKernel,
        KFold,
        MaternKernel,
        MultivariateNormal,
        ScaleKernel,
        VIVA,
        VariationalStrategy,
        gpb,
        gpytorch,
        my_train,
        norm,
        np,
        pd,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Functions
        """
    )
    return


@app.cell
def _(norm, np, torch):
    # Scale
    def Scale(X_train, X_test, y_train, y_test):
        mins = X_train.min(0)
        maxs = X_train.max(0)
        X_train = (X_train-mins)/(maxs-mins)
        X_test = (X_test-mins)/(maxs-mins)

        mean = y_train.mean()
        sd = np.sqrt(y_train.var())

        y_train = (y_train-mean)/sd
        y_test = (y_test-mean)/sd
    
        return X_train, X_test, y_train, y_test

    # Find too close points
    def find_unique_X(X):
        n = len(X)
        mask = torch.ones(n, dtype=torch.bool)
        with torch.no_grad():
            for i in range(n - 1, 0, -1):
                if torch.cdist(X[i:i + 1], X[:i]).min() < 0.001:
                    mask[i] = False
        return mask

    # CRPS
    def crps_norm_vectorized(observations, means, sigmas):
        """
        Compute CRPS for multiple normal distributions.
    
        Parameters:
        observations (array-like): Observed values
        means (array-like): Means of the forecast distributions
        sigmas (array-like): Variance of the forecast distributions
    
        Returns:
        array-like: CRPS values
        """
        observations = np.asarray(observations)
        means = np.asarray(means)
        sigmas = np.sqrt(np.asarray(sigmas))
    
        z = (observations - means) / sigmas
        crps = sigmas * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        return np.mean(crps)

    # Log-Score
    def log_score_norm_vectorized(observations, means, sigmas):
        """
        Compute Log Score for multiple normal distributions.
    
        Parameters:
        observations (array-like): Observed values
        means (array-like): Means of the forecast distributions
        sigmas (array-like): Variance of the forecast distributions
    
        Returns:
        array-like: Log Score values
        """
        observations = np.asarray(observations)
        means = np.asarray(means)
        sigmas = np.sqrt(np.asarray(sigmas))
    
        return np.mean(-norm.logpdf(observations, means, sigmas))
    return (
        Scale,
        crps_norm_vectorized,
        find_unique_X,
        log_score_norm_vectorized,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Data
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Choose dataset:
        """
    )
    return


@app.cell
def _(pd):
    data = pd.read_csv('https://raw.githubusercontent.com/TimGyger/VIF/refs/heads/main/Real%20World/Gaussian/3dRoad/3droad.txt', sep=" ", header=None)
    return (data,)


@app.cell
def _(data):
    data.head(3)
    return


@app.cell
def _(data, find_unique_X, np, torch):
    # Features & Response
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    X = X.values  # Selecting multiple features
    y = y.values  # Selecting the target variable

    mins = X.min(0)
    maxs = X.max(0)
    X1 = (X-mins)/(maxs-mins)
    X1_tensor = torch.tensor(X1, dtype=torch.float32)
    mask_all = find_unique_X(X1_tensor)
    X = X1[mask_all]
    y = y[mask_all]
    # Compute column-wise standard deviation
    stds = np.std(X1, axis=0)

    # Filter columns based on standard deviation
    mask = stds > 0.01  # Keep columns with std > 0.01
    X = X[:, mask]

    d = X.shape[1] # Features
    return X, d, y


@app.cell
def _(KFold, X):
    # Generate 5 disjoint test sets
    kf = KFold(n_splits=5, shuffle=True, random_state=100)
    test_indices_sets = []
    train_indices_sets = []
    for i, (_, test_indices) in enumerate(kf.split(X)):
        print(f"Fold {i+1} Test indices: {test_indices}")
        test_indices_sets.append(set(test_indices))
        train_indices_sets.append(set(range(len(X))) - test_indices_sets[i])
    return test_indices_sets, train_indices_sets


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Models
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### SKIP
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Parameters
        """
    )
    return


@app.cell
def _():
    num_inducing_points = 1000
    training_iterations = 100
    return (num_inducing_points,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Model
        """
    )
    return


@app.cell
def _(
    ConstantMean,
    MaternKernel,
    MultivariateNormal,
    ScaleKernel,
    d,
    gpytorch,
    num_inducing_points,
    torch,
):
    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(
                gpytorch.kernels.GridInterpolationKernel(MaternKernel(batch_shape=torch.Size([train_x.size(-1)]),nu=3/2), 
                                        grid_size=round(num_inducing_points/d), num_dims=1)
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            univariate_covars = self.covar_module(x.mT.unsqueeze(-1))
            covar_x = univariate_covars.prod(dim=-3)
            return MultivariateNormal(mean_x, covar_x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Training & Prediction
        """
    )
    return


app._unparsable_cell(
    r"""
    # Define row and column names
    row_names = ['MAE', 'RMSE', 'CRPS', 'LS','Time']
    col_names = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5']

    # Create an empty DataFrame with the specified row and column names
    dat_SKIP = pd.DataFrame(np.zeros((len(row_names), len(col_names))), 
                          index=row_names, columns=col_names)

    for j in range(5):

        # Data
        X_test = np.array([X[idx] for idx in test_indices_sets[j]])
        X_train = np.array([X[idx] for idx in train_indices_sets[j]])
        y_test = np.array([y[idx] for idx in test_indices_sets[j]])
        y_train = np.array([y[idx] for idx in train_indices_sets[j]])
        X_train, X_test, y_train, y_test = Scale(X_train, X_test, y_train, y_test)
        print(X_train.shape)
        print(y_train.shape)
    
        # Convert the numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        mask_all = find_unique_X(X_train_tensor)
        X_train_tensor = X_train_tensor[mask_all]
        y_train_tensor = y_train_tensor[mask_all]
        print(X_train.shape)
        print(y_train.shape)
    
        print(f\"Split {j+1} Test Response: {y_test}\")
    
        # Model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model_SKIP = GPRegressionModel(X_train_tensor, y_train_tensor, likelihood)
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model_SKIP.parameters(), lr=0.05)
        if torch.cuda.is_available():
            model_SKIP = model_SKIP.cuda()
            likelihood = likelihood.cuda()
        # \"Loss\" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_SKIP)
    
        # Train
        def train():
            #iterator = tqdm.tqdm(range(training_iterations), desc=\"Train\")
            print(\"Training started.\")
            for i in range(training_iterations):
                optimizer.zero_grad()
                with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(100):
                    # Get output from model
                    output = model_SKIP(X_train_tensor)
                    # Calc loss and backprop derivatives
                    loss = -mll(output, y_train_tensor)
                    loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
        
                if (i+1) % 10 == 0:
                    print(f\"Iteration {i+1}/{training_iterations}\")
                    print(f\"Loss: {loss.item()}\n\")
        

            print(\"Training completed.\")
    
        start_time = time.time()
        %time train()
        runtime_SKIP = time.time() - start_time

        # Prediction
        print(\"Prediction started.\")
        model_SKIP.eval()
        likelihood.eval()
        with gpytorch.settings.max_preconditioner_size(100), torch.no_grad():
            with gpytorch.settings.max_root_decomposition_size(100), gpytorch.settings.fast_pred_var():
                preds_SKIP = model_SKIP(X_test_tensor)
                variances_SKIP = preds_SKIP.variance
            MAE_SKIP = torch.mean(torch.abs(preds_SKIP.mean - y_test_tensor))
            RMSE_SKIP = torch.sqrt(torch.mean(torch.square(preds_SKIP.mean - y_test_tensor)))
            LS_SKIP = log_score_norm_vectorized(y_test_tensor,preds_SKIP.mean,variances_SKIP)
            CRPS_SKIP = crps_norm_vectorized(y_test_tensor,preds_SKIP.mean,variances_SKIP)
            dat_SKIP.loc[row_names[0], col_names[j]] = MAE_SKIP.item()
            dat_SKIP.loc[row_names[1], col_names[j]] = RMSE_SKIP.item()
            dat_SKIP.loc[row_names[2], col_names[j]] = CRPS_SKIP.item()
            dat_SKIP.loc[row_names[3], col_names[j]] = LS_SKIP.item()
            dat_SKIP.loc[row_names[4], col_names[j]] = runtime_SKIP
            print('Test MAE: {}'.format(MAE_SKIP))
            print('Test RMSE: {}'.format(RMSE_SKIP))
            print('Test CRPS: {}'.format(CRPS_SKIP))
            print('Test LS: {}'.format(LS_SKIP))
            print('Runtime: {} seconds'.format(runtime_SKIP))
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Sparse Gaussian Process Regression (SGPR)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Parameters
        """
    )
    return


@app.cell
def _():
    num_inducing_points_1 = 1000
    training_iterations_1 = 500
    return (num_inducing_points_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Model
        """
    )
    return


@app.cell
def _(
    ConstantMean,
    InducingPointKernel,
    MaternKernel,
    MultivariateNormal,
    ScaleKernel,
    d,
    gpytorch,
    num_inducing_points_1,
):
    class SGPR(gpytorch.models.ExactGP):

        def __init__(self, train_x, train_y, likelihood):
            super(SGPR, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.base_covar_module = ScaleKernel(MaternKernel(nu=3 / 2, ard_num_dims=d))
            self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:num_inducing_points_1, :].clone(), likelihood=likelihood)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Training & Prediction
        """
    )
    return


app._unparsable_cell(
    r"""
    # Define row and column names
    row_names = ['MAE', 'RMSE', 'CRPS', 'LS','Time']
    col_names = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5']

    # Create an empty DataFrame with the specified row and column names
    dat_SGPR = pd.DataFrame(np.zeros((len(row_names), len(col_names))), 
                          index=row_names, columns=col_names)

    for j in range(5):

        # Data
        X_test = np.array([X[idx] for idx in test_indices_sets[j]])
        X_train = np.array([X[idx] for idx in train_indices_sets[j]])
        y_test = np.array([y[idx] for idx in test_indices_sets[j]])
        y_train = np.array([y[idx] for idx in train_indices_sets[j]])
        X_train, X_test, y_train, y_test = Scale(X_train, X_test, y_train, y_test)
        print(X_train.shape)
        print(y_train.shape)
    
        # Convert the numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        mask_all = find_unique_X(X_train_tensor)
        X_train_tensor = X_train_tensor[mask_all]
        y_train_tensor = y_train_tensor[mask_all]
        print(X_train.shape)
        print(y_train.shape)
    
        print(f\"Split {j+1} Test Response: {y_test}\")
    
        # Model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model_SGPR = SGPR(X_train_tensor, y_train_tensor, likelihood)
        # Use the adam optimizer
        #optimizer = torch.optim.LBFGS(model_SGPR.parameters(), lr=0.1)
        optimizer = torch.optim.Adam(model_SGPR.parameters(), lr=0.1)
        if torch.cuda.is_available():
            model_SGPR = model_SGPR.cuda()
            likelihood = likelihood.cuda()
        # \"Loss\" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_SGPR)
    
        # Train
        def train():
            #iterator = tqdm.tqdm(range(training_iterations), desc=\"Train\")
            print(\"Training started.\")
            for i in range(training_iterations):
                def closure():
                    optimizer.zero_grad()
                    output = model_SGPR(X_train_tensor)
                    loss = -mll(output, y_train_tensor)
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
        
                if (i+1) % 10 == 0:
                    print(f\"Iteration {i+1}/{training_iterations}\")
                    print(f\"Loss: {loss.item()}\n\")

            print(\"Training completed.\")
    
        start_time = time.time()
        %time train()
        runtime_SGPR = time.time() - start_time

        # Prediction
        print(\"Prediction started.\")
        model_SGPR.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds_SGPR = model_SGPR.likelihood(model_SGPR(X_test_tensor))
            variances_SGPR = preds_SGPR.variance
            MAE_SGPR = torch.mean(torch.abs(preds_SGPR.mean - y_test_tensor))
            RMSE_SGPR = torch.sqrt(torch.mean(torch.square(preds_SGPR.mean - y_test_tensor)))
            LS_SGPR = log_score_norm_vectorized(y_test_tensor,preds_SGPR.mean,variances_SGPR)
            CRPS_SGPR = crps_norm_vectorized(y_test_tensor,preds_SGPR.mean,variances_SGPR)
            dat_SGPR.loc[row_names[0], col_names[j]] = MAE_SGPR.item()
            dat_SGPR.loc[row_names[1], col_names[j]] = RMSE_SGPR.item()
            dat_SGPR.loc[row_names[2], col_names[j]] = CRPS_SGPR.item()
            dat_SGPR.loc[row_names[3], col_names[j]] = LS_SGPR.item()
            dat_SGPR.loc[row_names[4], col_names[j]] = runtime_SGPR
            print('Test MAE: {}'.format(MAE_SGPR))
            print('Test RMSE: {}'.format(RMSE_SGPR))
            print('Test CRPS: {}'.format(CRPS_SGPR))
            print('Test LS: {}'.format(LS_SGPR))
            print('Runtime: {} seconds'.format(runtime_SGPR))
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Stochastic Variational Gaussian Processes (SVGP)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Parameters
        """
    )
    return


@app.cell
def _():
    num_inducing_points_2 = 1000
    training_iterations_2 = 500
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Model
        """
    )
    return


@app.cell
def _(
    ApproximateGP,
    CholeskyVariationalDistribution,
    MaternKernel,
    VariationalStrategy,
    d,
    gpytorch,
):
    class SVGP(ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
            super(SVGP, self).__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(MaternKernel(nu=3/2,ard_num_dims = d))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Training & Prediction
        """
    )
    return


app._unparsable_cell(
    r"""
    # Define row and column names
    row_names = ['MAE', 'RMSE', 'CRPS', 'LS','Time']
    col_names = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5']

    # Create an empty DataFrame with the specified row and column names
    dat_SVGP = pd.DataFrame(np.zeros((len(row_names), len(col_names))), 
                          index=row_names, columns=col_names)

    for j in range(5):

        # Data
        X_test = np.array([X[idx] for idx in test_indices_sets[j]])
        X_train = np.array([X[idx] for idx in train_indices_sets[j]])
        y_test = np.array([y[idx] for idx in test_indices_sets[j]])
        y_train = np.array([y[idx] for idx in train_indices_sets[j]])
        X_train, X_test, y_train, y_test = Scale(X_train, X_test, y_train, y_test)
        print(X_train.shape)
        print(y_train.shape)
    
        # Convert the numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        # Remove close points
        mask_all = find_unique_X(X_train_tensor)
        X_train_tensor = X_train_tensor[mask_all]
        y_train_tensor = y_train_tensor[mask_all]
        print(X_train.shape)
        print(y_train.shape)
        # Batches
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
        print(f\"Split {j+1} Test Response: {y_test}\")
    
        # Model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        inducing_points = X_train_tensor[:num_inducing_points, :]
        model_SVGP = SVGP(inducing_points=inducing_points)
        if torch.cuda.is_available():
            model_SVGP = model_SVGP.cuda()
            likelihood = likelihood.cuda()
        # Use the adam optimizer
        model_SVGP.train()
        likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': model_SVGP.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)
        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(likelihood, model_SVGP, num_data=y_train_tensor.size(0))
    
        # Train
        def train():
            print(\"Training started.\")
            #epochs_iter = tqdm.tqdm(range(training_iterations), desc=\"Epoch\")
            for i in range(training_iterations):
                # Within each iteration, we will go over each minibatch of data
                #minibatch_iter = tqdm.tqdm(train_loader, desc=\"Minibatch\", leave=False)
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    output = model_SVGP(x_batch)
                    loss = -mll(output, y_batch)
                    #minibatch_iter.set_postfix(loss=loss.item())
                    loss.backward()
                    optimizer.step()
                if (i+1) % 10 == 0:
                    print(f\"Iteration {i + 1}/{training_iterations}\")
                    print(f\"Loss: {loss.item()}\n\")
                
            print(\"Training completed.\")
    
        start_time = time.time()
        %time train()
        runtime_SVGP = time.time() - start_time

        # Prediction
        print(\"Prediction started.\")
        model_SVGP.eval()
        likelihood.eval()
        means_SVGP = torch.tensor([0.])
        variances_SVGP = torch.tensor([0.])
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                preds = likelihood(model_SVGP(x_batch))
                means_SVGP = torch.cat([means_SVGP, preds.mean.cpu()])
                variances_SVGP = torch.cat([variances_SVGP, preds.variance.cpu()])
            means_SVGP = means_SVGP[1:]
            variances_SVGP = variances_SVGP[1:]
        
            MAE_SVGP = torch.mean(torch.abs(means_SVGP - y_test_tensor))
            RMSE_SVGP = torch.sqrt(torch.mean(torch.square(means_SVGP - y_test_tensor)))
            LS_SVGP = log_score_norm_vectorized(y_test_tensor,means_SVGP,variances_SVGP)
            CRPS_SVGP = crps_norm_vectorized(y_test_tensor,means_SVGP,variances_SVGP)
            dat_SVGP.loc[row_names[0], col_names[j]] = MAE_SVGP.item()
            dat_SVGP.loc[row_names[1], col_names[j]] = RMSE_SVGP.item()
            dat_SVGP.loc[row_names[2], col_names[j]] = CRPS_SVGP.item()
            dat_SVGP.loc[row_names[3], col_names[j]] = LS_SVGP.item()
            dat_SVGP.loc[row_names[4], col_names[j]] = runtime_SVGP
            print('Test MAE: {}'.format(MAE_SVGP))
            print('Test RMSE: {}'.format(RMSE_SVGP))
            print('Test CRPS: {}'.format(CRPS_SVGP))
            print('Test LS: {}'.format(LS_SVGP))
            print('Runtime: {} seconds'.format(runtime_SVGP))
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Double-Kullback-Leibler-optimal Gaussian-process approximation (DKL-GP)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Parameters
        """
    )
    return


@app.cell
def _():
    classify = False
    rho = 1.5
    n_Epoch=35
    return classify, n_Epoch, rho


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Train & Prediction
        """
    )
    return


@app.cell
def _(
    MaternKernel,
    Scale,
    VIVA,
    X,
    classify,
    crps_norm_vectorized,
    d,
    find_unique_X,
    gpytorch,
    log_score_norm_vectorized,
    my_train,
    n_Epoch,
    np,
    pd,
    rho,
    test_indices_sets,
    time,
    torch,
    train_indices_sets,
    y,
):
    row_names = ['MAE', 'RMSE', 'CRPS', 'LS', 'Time']
    col_names = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5']
    dat_DKL = pd.DataFrame(np.zeros((len(row_names), len(col_names))), index=row_names, columns=col_names)
    for j in range(5):
        X_test = np.array([X[idx] for idx in test_indices_sets[j]])
        X_train = np.array([X[idx] for idx in train_indices_sets[j]])
        y_test = np.array([y[idx] for idx in test_indices_sets[j]])
        y_train = np.array([y[idx] for idx in train_indices_sets[j]])
        X_train, X_test, y_train, y_test = Scale(X_train, X_test, y_train, y_test)
        print(X_train.shape)
        print(y_train.shape)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        mask_all_1 = find_unique_X(X_train_tensor)
        X_train_tensor = X_train_tensor[mask_all_1]
        y_train_tensor = y_train_tensor[mask_all_1]
        print(X_train.shape)
        print(y_train.shape)
        X_tensor = torch.cat((X_train_tensor, X_test_tensor), dim=0)
        y_tensor = torch.cat((y_train_tensor, y_test_tensor), dim=0)
        print(f'Split {j + 1} Test Response: {y_test}')
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        K = gpytorch.kernels.ScaleKernel(MaternKernel(nu=3 / 2, ard_num_dims=d))
        model_DKL = VIVA(X_tensor, y_tensor, K, likelihood, rho, n_test=y_test.size, classify=classify, use_ic0=True)
        start_time = time.time()
        my_train(model_DKL, n_Epoch=n_Epoch)
        runtime_DKL = time.time() - start_time
        print('Prediction started.')
        preds_DKL, variances_DKL = model_DKL.predict()
        MAE_DKL = torch.mean(torch.abs(preds_DKL - y_test_tensor))
        RMSE_DKL = torch.sqrt(torch.mean(torch.square(preds_DKL - y_test_tensor)))
        CRPS_DKL = crps_norm_vectorized(y_test_tensor, preds_DKL, variances_DKL)
        LS_DKL = log_score_norm_vectorized(y_test_tensor, preds_DKL, variances_DKL)
        dat_DKL.loc[row_names[0], col_names[j]] = MAE_DKL.item()
        dat_DKL.loc[row_names[1], col_names[j]] = RMSE_DKL.item()
        dat_DKL.loc[row_names[2], col_names[j]] = CRPS_DKL.item()
        dat_DKL.loc[row_names[3], col_names[j]] = LS_DKL.item()
        dat_DKL.loc[row_names[4], col_names[j]] = runtime_DKL
        print('Test MAE: {}'.format(MAE_DKL))
        print('Test RMSE: {}'.format(RMSE_DKL))
        print('Test CRPS: {}'.format(CRPS_DKL))
        print('Test LS: {}'.format(LS_DKL))
        print('Runtime: {} seconds'.format(runtime_DKL))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### VIF Approximation
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Parameters
        """
    )
    return


@app.cell
def _():
    likelihood_1 = 'gaussian'
    num_inducing_points_3 = 200
    num_Vecchia_neighbors = 30
    return likelihood_1, num_Vecchia_neighbors, num_inducing_points_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Training & Prediction
        """
    )
    return


@app.cell
def _(
    Scale,
    X,
    crps_norm_vectorized,
    find_unique_X,
    gpb,
    likelihood_1,
    log_score_norm_vectorized,
    np,
    num_Vecchia_neighbors,
    num_inducing_points_3,
    pd,
    test_indices_sets,
    time,
    torch,
    train_indices_sets,
    y,
):
    row_names_1 = ['MAE', 'RMSE', 'CRPS', 'LS', 'Time']
    col_names_1 = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5']
    dat_vif_corr = pd.DataFrame(np.zeros((len(row_names_1), len(col_names_1))), index=row_names_1, columns=col_names_1)
    dat_vif_eucl = pd.DataFrame(np.zeros((len(row_names_1), len(col_names_1))), index=row_names_1, columns=col_names_1)
    for j_1 in range(5):
        X_test_1 = np.array([X[idx] for idx in test_indices_sets[j_1]])
        X_train_1 = np.array([X[idx] for idx in train_indices_sets[j_1]])
        y_test_1 = np.array([y[idx] for idx in test_indices_sets[j_1]])
        y_train_1 = np.array([y[idx] for idx in train_indices_sets[j_1]])
        X_train_1, X_test_1, y_train_1, y_test_1 = Scale(X_train_1, X_test_1, y_train_1, y_test_1)
        print(X_train_1.shape)
        print(y_train_1.shape)
        X_train_tensor_1 = torch.tensor(X_train_1, dtype=torch.float32)
        y_train_tensor_1 = torch.tensor(y_train_1, dtype=torch.float32)
        mask_all_2 = find_unique_X(X_train_tensor_1)
        X_train_tensor_1 = X_train_tensor_1[mask_all_2]
        y_train_tensor_1 = y_train_tensor_1[mask_all_2]
        X_train_1 = np.asarray(X_train_tensor_1)
        y_train_1 = np.asarray(y_train_tensor_1)
        print(X_train_1.shape)
        print(y_train_1.shape)
        model_vif_corr = gpb.GPModel(gp_coords=X_train_1, cov_function='matern_ard', cov_fct_shape=1.5, likelihood=likelihood_1, vecchia_ordering='random', num_neighbors=num_Vecchia_neighbors, num_ind_points=num_inducing_points_3, ind_points_selection='kmeans++', matrix_inversion_method='cholesky', gp_approx='full_scale_vecchia_correlation_based')
        model_vif_eucl = gpb.GPModel(gp_coords=X_train_1, cov_function='matern_ard', cov_fct_shape=1.5, likelihood=likelihood_1, vecchia_ordering='random', num_neighbors=num_Vecchia_neighbors, num_ind_points=num_inducing_points_3, ind_points_selection='kmeans++', matrix_inversion_method='cholesky', gp_approx='full_scale_vecchia')
        model_vif_corr.set_optim_params(params={'optimizer_cov': 'lbfgs', 'trace': True})
        start_time_1 = time.time()
        model_vif_corr.fit(y=y_train_1)
        runtime_vif_corr = time.time() - start_time_1
        model_vif_eucl.set_optim_params(params={'optimizer_cov': 'lbfgs', 'trace': True})
        start_time_1 = time.time()
        model_vif_eucl.fit(y=y_train_1)
        runtime_vif_eucl = time.time() - start_time_1
        pred_vif_corr = model_vif_corr.predict(gp_coords_pred=X_test_1, predict_var=True)
        MAE_vif_corr = np.mean(np.abs(pred_vif_corr['mu'] - y_test_1))
        RMSE_vif_corr = np.sqrt(np.mean(np.square(pred_vif_corr['mu'] - y_test_1)))
        CRPS_vif_corr = crps_norm_vectorized(y_test_1, pred_vif_corr['mu'], pred_vif_corr['var'])
        LS_vif_corr = log_score_norm_vectorized(y_test_1, pred_vif_corr['mu'], pred_vif_corr['var'])
        print('Test MAE: {}'.format(MAE_vif_corr))
        print('Test RMSE: {}'.format(RMSE_vif_corr))
        print('Test CRPS: {}'.format(CRPS_vif_corr))
        print('Test LS: {}'.format(LS_vif_corr))
        print('Runtime: {} seconds'.format(runtime_vif_corr))
        dat_vif_corr.loc[row_names_1[0], col_names_1[j_1]] = MAE_vif_corr.item()
        dat_vif_corr.loc[row_names_1[1], col_names_1[j_1]] = RMSE_vif_corr.item()
        dat_vif_corr.loc[row_names_1[2], col_names_1[j_1]] = CRPS_vif_corr.item()
        dat_vif_corr.loc[row_names_1[3], col_names_1[j_1]] = LS_vif_corr.item()
        dat_vif_corr.loc[row_names_1[4], col_names_1[j_1]] = runtime_vif_corr
        pred_vif_eucl = model_vif_eucl.predict(gp_coords_pred=X_test_1, predict_var=True)
        MAE_vif_eucl = np.mean(np.abs(pred_vif_eucl['mu'] - y_test_1))
        RMSE_vif_eucl = np.sqrt(np.mean(np.square(pred_vif_eucl['mu'] - y_test_1)))
        CRPS_vif_eucl = crps_norm_vectorized(y_test_1, pred_vif_eucl['mu'], pred_vif_eucl['var'])
        LS_vif_eucl = log_score_norm_vectorized(y_test_1, pred_vif_eucl['mu'], pred_vif_eucl['var'])
        print('Test MAE: {}'.format(MAE_vif_eucl))
        print('Test RMSE: {}'.format(RMSE_vif_eucl))
        print('Test CRPS: {}'.format(CRPS_vif_eucl))
        print('Test LS: {}'.format(LS_vif_eucl))
        print('Runtime: {} seconds'.format(runtime_vif_eucl))
        dat_vif_eucl.loc[row_names_1[0], col_names_1[j_1]] = MAE_vif_eucl.item()
        dat_vif_eucl.loc[row_names_1[1], col_names_1[j_1]] = RMSE_vif_eucl.item()
        dat_vif_eucl.loc[row_names_1[2], col_names_1[j_1]] = CRPS_vif_eucl.item()
        dat_vif_eucl.loc[row_names_1[3], col_names_1[j_1]] = LS_vif_eucl.item()
        dat_vif_eucl.loc[row_names_1[4], col_names_1[j_1]] = runtime_vif_eucl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### VIF Approximation (estimate shape parameter)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Parameters
        """
    )
    return


@app.cell
def _():
    likelihood_2 = 'gaussian'
    num_inducing_points_4 = 200
    num_Vecchia_neighbors_1 = 30
    return likelihood_2, num_Vecchia_neighbors_1, num_inducing_points_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Training & Prediction
        """
    )
    return


@app.cell
def _(
    Scale,
    X,
    crps_norm_vectorized,
    find_unique_X,
    gpb,
    likelihood_2,
    log_score_norm_vectorized,
    np,
    num_Vecchia_neighbors_1,
    num_inducing_points_4,
    pd,
    test_indices_sets,
    time,
    torch,
    train_indices_sets,
    y,
):
    row_names_2 = ['MAE', 'RMSE', 'CRPS', 'LS', 'nu', 'Time']
    col_names_2 = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5']
    dat_vif_corr_shape = pd.DataFrame(np.zeros((len(row_names_2), len(col_names_2))), index=row_names_2, columns=col_names_2)
    for j_2 in range(5):
        X_test_2 = np.array([X[idx] for idx in test_indices_sets[j_2]])
        X_train_2 = np.array([X[idx] for idx in train_indices_sets[j_2]])
        y_test_2 = np.array([y[idx] for idx in test_indices_sets[j_2]])
        y_train_2 = np.array([y[idx] for idx in train_indices_sets[j_2]])
        X_train_2, X_test_2, y_train_2, y_test_2 = Scale(X_train_2, X_test_2, y_train_2, y_test_2)
        print(X_train_2.shape)
        print(y_train_2.shape)
        X_train_tensor_2 = torch.tensor(X_train_2, dtype=torch.float32)
        y_train_tensor_2 = torch.tensor(y_train_2, dtype=torch.float32)
        mask_all_3 = find_unique_X(X_train_tensor_2)
        X_train_tensor_2 = X_train_tensor_2[mask_all_3]
        y_train_tensor_2 = y_train_tensor_2[mask_all_3]
        X_train_2 = np.asarray(X_train_tensor_2)
        y_train_2 = np.asarray(y_train_tensor_2)
        print(X_train_2.shape)
        print(y_train_2.shape)
        model_vif_corr_1 = gpb.GPModel(gp_coords=X_train_2, cov_function='matern_ard_estimate_shape', cov_fct_shape=1.5, likelihood=likelihood_2, vecchia_ordering='random', num_neighbors=num_Vecchia_neighbors_1, num_ind_points=num_inducing_points_4, ind_points_selection='kmeans++', matrix_inversion_method='cholesky', gp_approx='full_scale_vecchia_correlation_based')
        model_vif_corr_1.set_optim_params(params={'optimizer_cov': 'lbfgs', 'trace': True})
        start_time_2 = time.time()
        model_vif_corr_1.fit(y=y_train_2)
        runtime_vif_corr_1 = time.time() - start_time_2
        pred_vif_corr_1 = model_vif_corr_1.predict(gp_coords_pred=X_test_2, predict_var=True)
        MAE_vif_corr_1 = np.mean(np.abs(pred_vif_corr_1['mu'] - y_test_2))
        RMSE_vif_corr_1 = np.sqrt(np.mean(np.square(pred_vif_corr_1['mu'] - y_test_2)))
        CRPS_vif_corr_1 = crps_norm_vectorized(y_test_2, pred_vif_corr_1['mu'], pred_vif_corr_1['var'])
        LS_vif_corr_1 = log_score_norm_vectorized(y_test_2, pred_vif_corr_1['mu'], pred_vif_corr_1['var'])
        shape_vif_corr = model_vif_corr_1.get_cov_pars().iloc[0, -1]
        print('Test MAE: {}'.format(MAE_vif_corr_1))
        print('Test RMSE: {}'.format(RMSE_vif_corr_1))
        print('Test CRPS: {}'.format(CRPS_vif_corr_1))
        print('Test LS: {}'.format(LS_vif_corr_1))
        print('Test nu: {}'.format(shape_vif_corr))
        print('Runtime: {} seconds'.format(runtime_vif_corr_1))
        dat_vif_corr_shape.loc[row_names_2[0], col_names_2[j_2]] = MAE_vif_corr_1.item()
        dat_vif_corr_shape.loc[row_names_2[1], col_names_2[j_2]] = RMSE_vif_corr_1.item()
        dat_vif_corr_shape.loc[row_names_2[2], col_names_2[j_2]] = CRPS_vif_corr_1.item()
        dat_vif_corr_shape.loc[row_names_2[3], col_names_2[j_2]] = LS_vif_corr_1.item()
        dat_vif_corr_shape.loc[row_names_2[4], col_names_2[j_2]] = shape_vif_corr.item()
        dat_vif_corr_shape.loc[row_names_2[5], col_names_2[j_2]] = runtime_vif_corr_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### VIF Approximation (Linear Regression + GP)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Parameters
        """
    )
    return


@app.cell
def _():
    likelihood_3 = 'gaussian'
    num_inducing_points_5 = 200
    num_Vecchia_neighbors_2 = 30
    return likelihood_3, num_Vecchia_neighbors_2, num_inducing_points_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Training & Prediction
        """
    )
    return


@app.cell
def _(
    Scale,
    X,
    crps_norm_vectorized,
    find_unique_X,
    gpb,
    likelihood_3,
    log_score_norm_vectorized,
    np,
    num_Vecchia_neighbors_2,
    num_inducing_points_5,
    pd,
    test_indices_sets,
    time,
    torch,
    train_indices_sets,
    y,
):
    row_names_3 = ['MAE', 'RMSE', 'CRPS', 'LS', 'Time']
    col_names_3 = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5']
    dat_vif_corr_LR = pd.DataFrame(np.zeros((len(row_names_3), len(col_names_3))), index=row_names_3, columns=col_names_3)
    for j_3 in range(5):
        X_test_3 = np.array([X[idx] for idx in test_indices_sets[j_3]])
        X_train_3 = np.array([X[idx] for idx in train_indices_sets[j_3]])
        y_test_3 = np.array([y[idx] for idx in test_indices_sets[j_3]])
        y_train_3 = np.array([y[idx] for idx in train_indices_sets[j_3]])
        X_train_3, X_test_3, y_train_3, y_test_3 = Scale(X_train_3, X_test_3, y_train_3, y_test_3)
        print(X_train_3.shape)
        print(y_train_3.shape)
        X_train_tensor_3 = torch.tensor(X_train_3, dtype=torch.float32)
        y_train_tensor_3 = torch.tensor(y_train_3, dtype=torch.float32)
        mask_all_4 = find_unique_X(X_train_tensor_3)
        X_train_tensor_3 = X_train_tensor_3[mask_all_4]
        y_train_tensor_3 = y_train_tensor_3[mask_all_4]
        X_train_3 = np.asarray(X_train_tensor_3)
        y_train_3 = np.asarray(y_train_tensor_3)
        new_column = np.ones((X_train_3.shape[0], 1))
        X_train_intercept = np.hstack([new_column, X_train_3])
        new_column = np.ones((X_test_3.shape[0], 1))
        X_test_intercept = np.hstack([new_column, X_test_3])
        print(X_train_3.shape)
        print(y_train_3.shape)
        model_vif_corr_2 = gpb.GPModel(gp_coords=X_train_3, cov_function='matern_ard', cov_fct_shape=1.5, likelihood=likelihood_3, vecchia_ordering='random', num_neighbors=num_Vecchia_neighbors_2, num_ind_points=num_inducing_points_5, ind_points_selection='kmeans++', matrix_inversion_method='cholesky', gp_approx='full_scale_vecchia_correlation_based')
        model_vif_corr_2.set_optim_params(params={'optimizer_cov': 'lbfgs', 'trace': True})
        start_time_3 = time.time()
        model_vif_corr_2.fit(y=y_train_3, X=X_train_intercept)
        runtime_vif_corr_2 = time.time() - start_time_3
        pred_vif_corr_2 = model_vif_corr_2.predict(X_pred=X_test_intercept, gp_coords_pred=X_test_3, predict_var=True)
        MAE_vif_corr_2 = np.mean(np.abs(pred_vif_corr_2['mu'] - y_test_3))
        RMSE_vif_corr_2 = np.sqrt(np.mean(np.square(pred_vif_corr_2['mu'] - y_test_3)))
        CRPS_vif_corr_2 = crps_norm_vectorized(y_test_3, pred_vif_corr_2['mu'], pred_vif_corr_2['var'])
        LS_vif_corr_2 = log_score_norm_vectorized(y_test_3, pred_vif_corr_2['mu'], pred_vif_corr_2['var'])
        print('Test MAE: {}'.format(MAE_vif_corr_2))
        print('Test RMSE: {}'.format(RMSE_vif_corr_2))
        print('Test CRPS: {}'.format(CRPS_vif_corr_2))
        print('Runtime: {} seconds'.format(runtime_vif_corr_2))
        dat_vif_corr_LR.loc[row_names_3[0], col_names_3[j_3]] = MAE_vif_corr_2.item()
        dat_vif_corr_LR.loc[row_names_3[1], col_names_3[j_3]] = RMSE_vif_corr_2.item()
        dat_vif_corr_LR.loc[row_names_3[2], col_names_3[j_3]] = CRPS_vif_corr_2.item()
        dat_vif_corr_LR.loc[row_names_3[3], col_names_3[j_3]] = LS_vif_corr_2.item()
        dat_vif_corr_LR.loc[row_names_3[4], col_names_3[j_3]] = runtime_vif_corr_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### VIF Approximation (GPBoost)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Parameters
        """
    )
    return


@app.cell
def _():
    likelihood_4 = 'gaussian'
    num_inducing_points_6 = 200
    num_Vecchia_neighbors_3 = 30
    return likelihood_4, num_Vecchia_neighbors_3, num_inducing_points_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Training & Prediction
        """
    )
    return


@app.cell
def _(
    Scale,
    X,
    crps_norm_vectorized,
    find_unique_X,
    gpb,
    likelihood_4,
    log_score_norm_vectorized,
    np,
    num_Vecchia_neighbors_3,
    num_inducing_points_6,
    pd,
    test_indices_sets,
    time,
    torch,
    train_indices_sets,
    y,
):
    row_names_4 = ['MAE', 'RMSE', 'CRPS', 'LS', 'Time']
    col_names_4 = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5']
    dat_vif_corr_GPBoost = pd.DataFrame(np.zeros((len(row_names_4), len(col_names_4))), index=row_names_4, columns=col_names_4)
    for j_4 in range(5):
        X_test_4 = np.array([X[idx] for idx in test_indices_sets[j_4]])
        X_train_4 = np.array([X[idx] for idx in train_indices_sets[j_4]])
        y_test_4 = np.array([y[idx] for idx in test_indices_sets[j_4]])
        y_train_4 = np.array([y[idx] for idx in train_indices_sets[j_4]])
        X_train_4, X_test_4, y_train_4, y_test_4 = Scale(X_train_4, X_test_4, y_train_4, y_test_4)
        print(X_train_4.shape)
        print(y_train_4.shape)
        X_train_tensor_4 = torch.tensor(X_train_4, dtype=torch.float32)
        y_train_tensor_4 = torch.tensor(y_train_4, dtype=torch.float32)
        mask_all_5 = find_unique_X(X_train_tensor_4)
        X_train_tensor_4 = X_train_tensor_4[mask_all_5]
        y_train_tensor_4 = y_train_tensor_4[mask_all_5]
        X_train_4 = np.asarray(X_train_tensor_4)
        y_train_4 = np.asarray(y_train_tensor_4)
        print(X_train_4.shape)
        print(y_train_4.shape)
        search_space = {'learning_rate': [0.001, 10], 'min_data_in_leaf': [1, 100], 'max_depth': [-1, -1], 'num_leaves': [100, 1024], 'lambda_l2': [0, 1000], 'max_bin': [100, 1000], 'line_search_step_length': [True, False]}
        metric = 'mse'
        model_vif_corr_3 = gpb.GPModel(gp_coords=X_train_4, cov_function='matern_ard', cov_fct_shape=1.5, likelihood=likelihood_4, vecchia_ordering='random', num_neighbors=num_Vecchia_neighbors_3, num_ind_points=num_inducing_points_6, ind_points_selection='kmeans++', matrix_inversion_method='cholesky', gp_approx='full_scale_vecchia_correlation_based')
        opt_params = gpb.tune_pars_TPE_algorithm_optuna(X=X_train_4, y=y_train_4, search_space=search_space, nfold=5, cv_seed=4, gp_model=model_vif_corr_3, metric=metric, tpe_seed=1, max_num_boost_round=1000, n_trials=100, early_stopping_rounds=20)
        print('Best parameters: ' + str(opt_params['best_params']))
        print('Best number of iterations: ' + str(opt_params['best_iter']))
        print('Best score: ' + str(opt_params['best_score']))
        data_train = gpb.Dataset(data=X_train_4, label=y_train_4)
        start_time_4 = time.time()
        bst = gpb.train(params=opt_params['best_params'], train_set=data_train, gp_model=model_vif_corr_3, num_boost_round=opt_params['best_iter'])
        runtime_vif_corr_3 = time.time() - start_time_4
        pred_vif_corr_3 = bst.predict(data=X_test_4, predict_var=True, gp_coords_pred=X_test_4)
        MAE_vif_corr_3 = np.mean(np.abs(pred_vif_corr_3['response_mean'] - y_test_4))
        RMSE_vif_corr_3 = np.sqrt(np.mean(np.square(pred_vif_corr_3['response_mean'] - y_test_4)))
        CRPS_vif_corr_3 = crps_norm_vectorized(y_test_4, pred_vif_corr_3['response_mean'], pred_vif_corr_3['response_var'])
        LS_vif_corr_3 = log_score_norm_vectorized(y_test_4, pred_vif_corr_3['response_mean'], pred_vif_corr_3['response_var'])
        print('Test MAE: {}'.format(MAE_vif_corr_3))
        print('Test RMSE: {}'.format(RMSE_vif_corr_3))
        print('Test CRPS: {}'.format(CRPS_vif_corr_3))
        print('Test LS: {}'.format(LS_vif_corr_3))
        print('Runtime: {} seconds'.format(runtime_vif_corr_3))
        dat_vif_corr_GPBoost.loc[row_names_4[0], col_names_4[j_4]] = MAE_vif_corr_3.item()
        dat_vif_corr_GPBoost.loc[row_names_4[1], col_names_4[j_4]] = RMSE_vif_corr_3.item()
        dat_vif_corr_GPBoost.loc[row_names_4[2], col_names_4[j_4]] = CRPS_vif_corr_3.item()
        dat_vif_corr_GPBoost.loc[row_names_4[3], col_names_4[j_4]] = LS_vif_corr_3.item()
        dat_vif_corr_GPBoost.loc[row_names_4[4], col_names_4[j_4]] = runtime_vif_corr_3
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
