    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param epsilon: Overshoot parameter.
        :type epsilon: `float`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        assert self.set_params(**kwargs)
        clip_min, clip_max = self.classifier.clip_values
        x_adv = x.copy()
        preds = self.classifier.predict(x, logits=True)

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        for j, val in enumerate(x_adv):
            xj = val[None, ...]
            f = preds[j]
            grd = self.classifier.class_gradient(xj, logits=True)[0]
            fk_hat = np.argmax(f)

            for _ in range(self.max_iter):
                grad_diff = grd - grd[fk_hat]
                f_diff = f - f[fk_hat]

                # Choose coordinate and compute perturbation
                norm = np.linalg.norm(grad_diff.reshape(self.classifier.nb_classes, -1), axis=1) + tol
                value = np.abs(f_diff) / norm
                value[fk_hat] = np.inf
                l = np.argmin(value)
                r = (abs(f_diff[l]) / (pow(np.linalg.norm(grad_diff[l]), 2) + tol)) * grad_diff[l]

                # Add perturbation and clip result
                xj = np.clip(xj + r, clip_min, clip_max)

                # Recompute prediction for new xj
                f = self.classifier.predict(xj, logits=True)[0]
                grd = self.classifier.class_gradient(xj, logits=True)[0]
                fk_i_hat = np.argmax(f)

                # Stop if misclassification has been achieved
                if fk_i_hat != fk_hat:
                    break

            # Apply overshoot parameter
            x_adv[j] = np.clip(x[j] + (1 + self.epsilon) * (xj[0] - x[j]), clip_min, clip_max)

        preds = np.argmax(preds, axis=1)
        preds_adv = np.argmax(self.classifier.predict(x_adv), axis=1)
        logger.info('Success rate of DeepFool attack: %.2f%%', (np.sum(preds != preds_adv) / x.shape[0]))

        return x_adv