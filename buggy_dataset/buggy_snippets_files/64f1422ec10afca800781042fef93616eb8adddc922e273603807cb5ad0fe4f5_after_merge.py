    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None):
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        interp_term = torch.triangular_solve(induc_data_covar.double(), L, upper=False)[0].to(full_inputs.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = torch.matmul(
            interp_term.transpose(-1, -2),
            (inducing_values - self.prior_distribution.mean).unsqueeze(-1)
        ).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLazyTensor(variational_inducing_covar, middle_term)
        predictive_covar = SumLazyTensor(
            data_data_covar.add_jitter(1e-4),
            MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term)
        )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)