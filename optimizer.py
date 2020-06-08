


class Trans_Optim():
    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        # init optimizer
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.lr = 0

    def step_and_update_lr(self):
        # optimizer.step + update_learning
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model**(-0.5)) * min(n_steps**(-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        
        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
    
    def _print_lr(self):
        return self._optimizer.param_groups[0]['lr']

    def _print_step(self):
        return self.n_steps

    def reset_step(self, step):
        self.n_steps = step