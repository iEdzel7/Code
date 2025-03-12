    def process_output(self, output, train=False):
        """Reduces output according to the training mode.

        Separates loss from logging and progress bar metrics
        """
        # --------------------------
        # handle single scalar only
        # --------------------------
        # single scalar returned from a xx_step
        if isinstance(output, torch.Tensor):
            progress_bar_metrics = {}
            log_metrics = {}
            callback_metrics = {}
            hiddens = None
            return output, progress_bar_metrics, log_metrics, callback_metrics, hiddens

        # ---------------
        # EXTRACT CALLBACK KEYS
        # ---------------
        # all keys not progress_bar or log are candidates for callbacks
        callback_metrics = {}
        for k, v in output.items():
            if k not in ['progress_bar', 'log', 'hiddens']:
                callback_metrics[k] = v

        if train and (self.use_dp or self.use_ddp2):
            num_gpus = self.num_gpus
            callback_metrics = self.reduce_distributed_output(callback_metrics, num_gpus)

        # ---------------
        # EXTRACT PROGRESS BAR KEYS
        # ---------------
        try:
            progress_output = output['progress_bar']

            # reduce progress metrics for progress bar when using dp
            if train and (self.use_dp or self.use_ddp2):
                num_gpus = self.num_gpus
                progress_output = self.reduce_distributed_output(progress_output, num_gpus)

            progress_bar_metrics = progress_output
        except Exception:
            progress_bar_metrics = {}

        # ---------------
        # EXTRACT LOGGING KEYS
        # ---------------
        # extract metrics to log to experiment
        try:
            log_output = output['log']

            # reduce progress metrics for progress bar when using dp
            if train and (self.use_dp or self.use_ddp2):
                num_gpus = self.num_gpus
                log_output = self.reduce_distributed_output(log_output, num_gpus)

            log_metrics = log_output
        except Exception:
            log_metrics = {}

        # ---------------
        # EXTRACT LOSS
        # ---------------
        # if output dict doesn't have the keyword loss
        # then assume the output=loss if scalar
        loss = None
        if train:
            try:
                loss = output['loss']
            except Exception:
                if isinstance(output, torch.Tensor):
                    loss = output
                else:
                    raise RuntimeError(
                        'No `loss` value in the dictionary returned from `model.training_step()`.'
                    )

            # when using dp need to reduce the loss
            if self.use_dp or self.use_ddp2:
                loss = self.reduce_distributed_output(loss, self.num_gpus)

        # ---------------
        # EXTRACT HIDDEN
        # ---------------
        hiddens = output.get('hiddens')

        # use every metric passed in as a candidate for callback
        callback_metrics.update(progress_bar_metrics)
        callback_metrics.update(log_metrics)

        # detach all metrics for callbacks to prevent memory leaks
        # no .item() because it will slow things down
        callback_metrics = recursive_detach(callback_metrics)

        return loss, progress_bar_metrics, log_metrics, callback_metrics, hiddens